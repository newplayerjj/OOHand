import os
from os.path import dirname, abspath

import tensorflow as tf

from openpose_mob_thin import MobilenetNetworkThin
from openpose_hand_model import OpenPoseHand

import common


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        # net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.50, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'D:/wzchen/PythonProj/tf-openpose/models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_thin_full':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=1.0, conv_width2=1.0, trainable=trainable)
        pretrain_path = 'D:/wzchen/PythonProj/tf-openpose/models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'openposehand':
        net = OpenPoseHand({'image': placeholder_input}, trainable=trainable)
        pretrain_path = './openposehand_pretrain/'
        last_layer = 'Mconv7_stage6'
    else:
        raise Exception('Invalid Mode.')

    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
    if sess_for_load is not None:
        if type == 'cmu' or type == 'vgg':
            if not os.path.isfile(pretrain_path_full):
                raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        else:
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
            }
            ckpt_path = os.path.join(_get_base_path(), ckpts[type])
            loader = tf.train.Saver()
            try:
                loader.restore(sess_for_load, ckpt_path)
            except Exception as e:
                raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))

    return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': 'graph/cmu/graph_opt.pb',
        'mobilenet_thin': 'graph/mobilenet_thin/graph_opt.pb'
    }

    base_data_dir = dirname(dirname(abspath(__file__)))
    if os.path.exists(os.path.join(base_data_dir, 'models')):
        base_data_dir = os.path.join(base_data_dir, 'models')
    else:
        base_data_dir = os.path.join(base_data_dir, 'tf_pose_data')

    graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
    if os.path.isfile(graph_path):
        return graph_path

    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)

if __name__ == '__main__':
    import logging
    logging.getLogger("requests").setLevel(logging.WARNING)
    logger = logging.getLogger('pose_dataset')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    import numpy as np
    from pose_dataset import get_dataflow_batch, DataFlowToQueue, OpenOoseHand
    output_w = common.network_w // common.network_scale
    output_h = common.network_h // common.network_scale 
    # # input_wh, scale is sync using common.py (to avoid multithread bug)
    # set_network_input_wh(args.input_width, args.input_height)
    # set_network_scale(scale)

    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(common.batchsize, common.network_h, common.network_w, 3), name='image')
        heatmap_node = tf.placeholder(tf.float32, shape=(common.batchsize, output_h, output_w, common.num_hand_parts), name='heatmap')
        
        df = get_dataflow_batch('./hand143_panopticdb/', True, common.batchsize)
        
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node], queue_size=2)
        q_inp, q_heat = enqueuer.dequeue()

    net, pretrain_path, last_layer = get_network(common.model, q_inp)
    net_out_heat = net.loss_last()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        # 餵入資料的shape若是不對 會死在這裡
        enqueuer.start()

        loss = net_out_heat - q_heat

        loss_val, out_heat, inp, heat = sess.run([loss, net_out_heat, q_inp, q_heat])
        print('loss_val= %s' % str(np.sum(loss_val)))

        # for batch in range(0, common.batchsize):
        #     OpenOoseHand.display_image(inp[batch], heat[batch].astype(np.float32))
    
        coord.request_stop()