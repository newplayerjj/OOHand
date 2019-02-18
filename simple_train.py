import argparse
import logging
import os
import time
import pdb

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pose_dataset import get_dataflow_batch, OpenOoseHand

from networks import get_network

import common

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
parser.add_argument('--datapath', type=str, default='D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max-epoch', type=int, default=30)
# --lr give initial learning rate
parser.add_argument('--lr', type=str, default='0.01')
parser.add_argument('--modelpath', type=str, default='D:/wzchen/PythonProj/cwz_handpose/tf-openpose-models-2018-2-18/')
parser.add_argument('--logpath', type=str, default='D:/wzchen/PythonProj/cwz_handpose/tf-openpose-models-2018-2-18/')
parser.add_argument('--checkpoint', type=str, default='D:/wzchen/PythonProj/cwz_handpose/tf-openpose-models-2018-2-18/mobilenet_thin_batch_8_lr_0.01_gpus_1_184x184_/')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()

if __name__ == '__main__':
    output_w, output_h = common.network_w // common.network_scale, common.network_h // common.network_scale
    logger.info('define model+')

    # get placeholders
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(common.batchsize, common.network_h, common.network_w, 3), name='image')
        heatmap_node = tf.placeholder(tf.float32, shape=(common.batchsize, output_h, output_w, common.num_hand_parts), name='heatmap')
        
    # get dataflow
    # df = get_dataflow_batch('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', True, common.batchsize)

    # get network output
    net, pretrain_path, last_layer = get_network(common.model, input_node)

    # get loss
    losses = []
    last_loss = []
    l1s = net.loss_l1()
    for idx, l1 in enumerate(l1s):
        # l1 = [stage1_L1_5 stage2_L1_5 ... ]
        # l2 = [stage1_L1_5 stage2_L1_5 ... ]
        loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - heatmap_node, name='loss_l1_stage%d_tower%d' % (idx, 0))
        losses.append(tf.reduce_mean([loss_l1]))
    last_loss.append(loss_l1)

    # get training and update ops
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        # define loss
        total_loss = tf.reduce_sum(losses) / common.batchsize
        total_loss_ll = tf.reduce_sum(last_loss) / common.batchsize

        # define optimizer
        step_per_epoch = common.total_training_data // common.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       decay_steps=step_per_epoch, decay_rate=0.95, staircase=True)
                                                        # decay_steps=step_per_epoch, decay_rate=0.33, staircase=True)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            # boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            boundaries = [step_per_epoch * i for i in range(0,len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    merged_summary_op = tf.summary.merge_all()

    ##################
    # start training #
    ##################
    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        training_name = '{}_batch_{}_lr_{}_gpus_{}_{}x{}_{}'.format(
            common.model,
            common.batchsize,
            args.lr,
            args.gpus,
            common.network_w, common.network_h,
            args.tag
        )

        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint %s...Done' % tf.train.latest_checkpoint(args.checkpoint))
        elif pretrain_path:
            logger.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(pretrain_path, sess, False)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(args.logpath + training_name, sess.graph)

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)
        # get dataflow
        df = get_dataflow_batch('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', True, common.batchsize)

        for epoch in range(0, args.max_epoch):
            
            #
            loss_history = []
            loss_history_ll = []

            for step, dp in enumerate(df.get_data()):
                inp = dp[0].astype(np.float32)
                heat = dp[1].astype(np.float32)

                pred = sess.run([net.loss_last()], {'image:0': inp,
                                                    'heatmap:0': heat})
                # for batch in range(0, common.batchsize):
                #     # OpenOoseHand.display_image(inp[batch], pred[0][batch].astype(np.float32))
                #     OpenOoseHand.display_image(inp[batch], heat[batch])

                _, gs_num = sess.run([train_op, global_step], {'image:0': inp,
                                                               'heatmap:0': heat})
                if (gs_num - last_gs_num) % 5 == 0:                                            
                    train_loss, train_loss_ll = sess.run([total_loss, total_loss_ll],  {'image:0': inp, 'heatmap:0': heat})
                    loss_history.append(train_loss)
                    loss_history_ll.append(train_loss_ll)
                if gs_num - last_gs_num >= 100:
                    lr_val, summary = sess.run([learning_rate, merged_summary_op], 
                                                                          {'image:0': inp, 'heatmap:0': heat})
                    train_loss = np.mean(loss_history)
                    train_loss_ll = np.mean(loss_history_ll)
                    # loss_history = []
                    # loss_history_ll = []
                    # log of training loss / accuracy
                    batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                    logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g' % (gs_num / step_per_epoch, gs_num, batch_per_sec * common.batchsize, lr_val, train_loss, train_loss_ll))
                    last_gs_num = gs_num

                    file_writer.add_summary(summary, gs_num)

                if gs_num - last_gs_num2 >= 1000:
                    # save weights
                    save_path = os.path.join(args.modelpath, training_name, 'model')
                    saver.save(sess, save_path, global_step=global_step)
                    last_gs_num2 = gs_num
                    logger.info('save weights. %s' % save_path)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
        logger.info('optimization finished. %f' % (time.time() - time_started))
