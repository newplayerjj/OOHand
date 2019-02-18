import json

import logging
import math
import multiprocessing
import struct
import sys
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import random
import cv2
import numpy as np
import time

import tensorflow as tf

from tensorpack.dataflow import MultiThreadMapData
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData, RepeatedData
from tensorpack.dataflow.parallel import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from pose_augment import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale, hand_random_scale, crop_hand_roi_big, crop_hand_roi

import common

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mplset = False

class OOHandMataData:
    __hand_parts = common.num_hand_parts
    def __init__(self, idx, img_url, img_meta, annotations, sigma):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        self.joint_list = []
        for x, y, v in annotations:
            self.joint_list.append([int(round(x)), int(round(y))] if v == 1 else (-1000, -1000))

    def get_heatmap(self, target_size):
        heatmap = np.zeros((OOHandMataData.__hand_parts, self.height, self.width), dtype=np.float32)

        for idx, point in enumerate(self.joint_list):
            if point[0] < 0 or point[1] < 0:
                continue
            OOHandMataData.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        # th = 2.3026
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

class MPIIModified(RNGDataFlow):
    def __init__(self, root_path, is_train):
        self.is_train = is_train
        self.root_path = root_path
        self.test_data_path = os.path.join(self.root_path,'manual_test')
        self.train_data_path = os.path.join(self.root_path,'manual_train')

        self.meta_list = data['root']
        self.prefix_zeros = 8

class OpenOoseHand(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, as_numpy=False):
        global mplset
        # if as_numpy and not mplset:
        #     import matplotlib as mpl
        #     mpl.use('Agg')
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(OpenOoseHand.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(OpenOoseHand.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, root_path, is_train):
        self.is_train = is_train
        self.root_path = root_path
        jsonPath = os.path.join(self.root_path,'hands_v143_14817.json')
        with open(jsonPath) as f:
            data = json.load(f)
        self.meta_list = data['root']
        self.prefix_zeros = 8
        # peek self.size()
        # import pdb; pdb.set_trace()

    def size(self):
        return len(self.meta_list)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass

        for idx in idxs:
            meta_data = self.meta_list[idx]
            img_path = os.path.join(self.root_path, meta_data['img_paths'])
            img_meta = {'height': meta_data['img_height'],
                        'width': meta_data['img_width']}

            meta = OOHandMataData(idx, img_path, img_meta, meta_data['joint_self'], sigma=8.0)

            total_keypoints = len(meta_data['joint_self'])
            if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                continue

            yield [meta]

def read_image_url(metas):
    for meta in metas:
        img_str = open(meta.img_url, 'rb').read()

        if not img_str:
            logger.warning('image not read, path=%s' % meta.img_url)
            raise Exception()
        nparr = np.fromstring(img_str, np.uint8)
        meta.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return metas

def get_dataflow(path, is_train, img_path=None):
    ds = OpenOoseHand(path, is_train)       # read data from lmdb
    # if is_train:
    #     ''' 
    #         ds is a DataFlow Object which must implement get_data() function
            
    #         MapData(ds, func) will apply func to data returned by ds.get_data()
    #         1. create an obj
    #         2. obj.ds = ds
    #         2. obj.get_data():
    #             data = self.ds.get_data()
    #             yield self.func(data)

    #         MapDataComponent(ds, func) will do similar thing
    #             the main different is the target of MapDataComponent(...)
    #             is the returned data of get_data()
    #     '''

    #     ds = MapData(ds, read_image_url)
    #     ds = MapDataComponent(ds, pose_random_scale)
    #     ds = MapDataComponent(ds, pose_rotation)
    #     ds = MapDataComponent(ds, pose_flip)
    #     ds = MapDataComponent(ds, pose_resize_shortestedge_random)
    #     ds = MapDataComponent(ds, pose_crop_random)
    #     # use joint_list to draw two point and vector heatmap
    #     ds = MapData(ds, pose_to_img)
    #     # augs = [
    #     #     imgaug.RandomApplyAug(imgaug.RandomChooseAug([
    #     #         imgaug.GaussianBlur(max_size=3)
    #     #     ]), 0.7)
    #     # ]
    #     # ds = AugmentImageComponent(ds, augs)
    #     # ds = PrefetchData(ds, 10, multiprocessing.cpu_count() * 4)
    #     ds = PrefetchData(ds, 2, 1)
    # else:
    #     ds = MultiThreadMapData(ds, nr_thread=16, map_func=read_image_url, buffer_size=1000)
    #     ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
    #     ds = MapDataComponent(ds, pose_crop_center)
    #     ds = MapData(ds, pose_to_img)
    #     ds = PrefetchData(ds, 10, multiprocessing.cpu_count() // 4)
    if is_train:
        ''' 
            ds is a DataFlow Object which must implement get_data() function
            
            MapData(ds, func) will apply func to data returned by ds.get_data()
            1. create an obj
            2. obj.ds = ds
            2. obj.get_data():
                data = self.ds.get_data()
                yield self.func(data)

            MapDataComponent(ds, func) will do similar thing
                the main different is the target of MapDataComponent(...)
                is the returned data of get_data()
        '''

        ds = MapData(ds, read_image_url)
        ds = MapDataComponent(ds, crop_hand_roi_big)
        ds = MapDataComponent(ds, hand_random_scale)
        ds = MapDataComponent(ds, pose_rotation)
        ds = MapDataComponent(ds, pose_flip)
        ds = MapDataComponent(ds, crop_hand_roi)
        # ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        # ds = MapDataComponent(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 20, 1)
    else:
        ds = MultiThreadMapData(ds, nr_thread=1, map_func=read_image_url, buffer_size=5)
        ds = MapDataComponent(ds, crop_hand_roi_big)
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 2, 2)

    return ds


def _get_dataflow_onlyread(path, is_train, img_path=None):
    ds = OpenOoseHand(path, is_train)  # read data from lmdb
    ds = MapData(ds, read_image_url)
    ds = MapDataComponent(ds, crop_hand_roi_big)
    ds = MapDataComponent(ds, hand_random_scale)
    ds = MapDataComponent(ds, pose_rotation)
    ds = MapDataComponent(ds, pose_flip)
    ds = MapDataComponent(ds, crop_hand_roi)
    # ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
    # ds = MapDataComponent(ds, pose_crop_random)
    ds = MapData(ds, pose_to_img)
    # ds = PrefetchData(ds, 10, 2)
    ds = RepeatedData(ds, -1)
    return ds


def get_dataflow_batch(path, is_train, batchsize, img_path=None):
    logger.info('dataflow img_path=%s' % img_path)
    ds = get_dataflow(path, is_train, img_path=img_path)
    ds = BatchData(ds, batchsize)
    ds = RepeatedData(ds, -1)
    if is_train:
        ds = PrefetchData(ds, batchsize*2, 1)
    else:
        ds = PrefetchData(ds, batchsize*2, 1)

    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=2):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()

def test_enqueuer():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_width = common.network_w
    args.input_height = common.network_h
    args.batchsize = common.batchsize
    scale = common.network_scale
    output_w = args.input_width // scale
    output_h = args.input_height // scale 
    # # input_wh, scale is sync using common.py (to avoid multithread bug)
    # set_network_input_wh(args.input_width, args.input_height)
    # set_network_scale(scale)

    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, common.num_hand_parts), name='heatmap')
        
        df = get_dataflow_batch('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', True, args.batchsize)
        
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node], queue_size=2)
        q_inp, q_heat = enqueuer.dequeue()


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        # 餵入資料的shape若是不對 會死在這裡
        enqueuer.start()

        inp, heat = sess.run([q_inp, q_heat])
        print('inp.shape = %s' % str(inp.shape))
        print('heat.shape = %s' % str(heat.shape))

        for batch in range(0, common.batchsize):
            OpenOoseHand.display_image(inp[batch], heat[batch].astype(np.float32))
    
        coord.request_stop()
    
def test_data_flow():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # df = get_dataflow('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', True)
    df = _get_dataflow_onlyread('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', True)
    # df = get_dataflow('D:/wzchen/PythonProj/cwz_handpose/hand143_panopticdb/', False)

    # from tensorpack.dataflow.common import TestDataSpeed
    # TestDataSpeed(df).start()
    # sys.exit(0)

    with tf.Session() as sess:
        df.reset_state()
        t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logger.info('%d dp shape={}'.format(d.shape))
            print(time.time() - t1)
            t1 = time.time()
 
            OpenOoseHand.display_image(dp[0], dp[1].astype(np.float32))
            print(dp[1].shape)
            pass

    logger.info('done')

if __name__ == '__main__':
     test_data_flow()