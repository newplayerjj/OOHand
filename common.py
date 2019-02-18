from enum import Enum

import tensorflow as tf
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

num_hand_parts = 22

batchsize = 8
network_w = 184
network_h = 184

model = 'mobilenet_thin'
network_scale = 8

total_training_data = 14817
