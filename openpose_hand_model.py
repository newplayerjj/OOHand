from __future__ import absolute_import

import tensorflow as tf

import network_base

import common

class OpenPoseHand(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=None):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        padding = 'SAME'
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        
        (self.feed('image')

            .conv(3, 3, 64, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv1_1')
            .conv(3, 3, 64, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv1_2')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool1_stage1')

            .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv2_1')
            .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv2_2')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool2_stage1')

            .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv3_1')
            .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv3_2')
            .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv3_3')
            .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv3_4')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool3_stage1')

            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv4_1')
            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv4_2')
            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv4_3')
            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv4_4')
            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv5_1')
            .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv5_2')

            .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='conv5_3_CPM')

            .conv(1, 1, 512, 1, 1, padding='VALID', kernel_name='kernel', bias_name='bias', name='conv6_1_CPM')
            .conv(1, 1, 22, 1, 1, padding='VALID', kernel_name='kernel', bias_name='bias', relu=False, name='conv6_2_CPM')
        )

        self.feed('conv6_2_CPM', 'conv5_3_CPM')
        for stage in range(2, 7):
            (self.concat(3, name='concat_stage{}'.format(stage))
                .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='Mconv1_stage{}'.format(stage))
                .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='Mconv2_stage{}'.format(stage))
                .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='Mconv3_stage{}'.format(stage))
                .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='Mconv4_stage{}'.format(stage))
                .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', bias_name='bias', name='Mconv5_stage{}'.format(stage))

                .conv(1, 1, 128, 1, 1, padding='VALID', kernel_name='kernel', bias_name='bias', name='Mconv6_stage{}'.format(stage))
                .conv(1, 1, 22, 1, 1, padding='VALID', kernel_name='kernel', bias_name='bias', relu=False, name='Mconv7_stage{}'.format(stage))
            )
            self.feed('Mconv7_stage{}'.format(stage), 'conv5_3_CPM')

        self.feed('Mconv7_stage{}'.format(6))

    # def setup(self):
    #     min_depth = 8
    #     padding = 'SAME'
    #     depth = lambda d: max(int(d * self.conv_width), min_depth)
    #     depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        
    #     (self.feed('image')

    #         .conv(3, 3, 64, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv1_1').batch_normalization(relu=True)
    #         .conv(3, 3, 64, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv1_2').batch_normalization(relu=True)
    #         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1_stage1')

    #         .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv2_1').batch_normalization(relu=True)
    #         .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv2_2').batch_normalization(relu=True)
    #         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2_stage1')

    #         .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv3_1').batch_normalization(relu=True)
    #         .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv3_2').batch_normalization(relu=True)
    #         .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv3_3').batch_normalization(relu=True)
    #         .conv(3, 3, 256, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv3_4').batch_normalization(relu=True)
    #         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3_stage1')

    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv4_1').batch_normalization(relu=True)
    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv4_2').batch_normalization(relu=True)
    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv4_3').batch_normalization(relu=True)
    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv4_4').batch_normalization(relu=True)
    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv5_1').batch_normalization(relu=True)
    #         .conv(3, 3, 512, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv5_2').batch_normalization(relu=True)

    #         .conv(3, 3, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='conv5_3_CPM').batch_normalization(relu=True)

    #         .conv(1, 1, 512, 1, 1, padding='VALID', kernel_name='kernel', biased=False, relu=False, name='conv6_1_CPM').batch_normalization(relu=True)
    #         .conv(1, 1, 22, 1, 1, padding='VALID', kernel_name='kernel', biased=True, bias_name='bias', name='conv6_2_CPM')
    #     )

    #     self.feed('conv6_2_CPM', 'conv5_3_CPM')
    #     for stage in range(2, 7):
    #         (self.concat(3, name='concat_stage{}'.format(stage))
    #             .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='Mconv1_stage{}'.format(stage)).batch_normalization(relu=True)
    #             .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='Mconv2_stage{}'.format(stage)).batch_normalization(relu=True)
    #             .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='Mconv3_stage{}'.format(stage)).batch_normalization(relu=True)
    #             .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='Mconv4_stage{}'.format(stage)).batch_normalization(relu=True)
    #             .conv(7, 7, 128, 1, 1, padding=padding, kernel_name='kernel', biased=False, relu=False, name='Mconv5_stage{}'.format(stage)).batch_normalization(relu=True)

    #             .conv(1, 1, 128, 1, 1, padding='VALID', kernel_name='kernel', biased=False, relu=False, name='Mconv6_stage{}'.format(stage)).batch_normalization(relu=True)
    #             .conv(1, 1, 22, 1, 1, padding='VALID', kernel_name='kernel', biased=True, bias_name='bias', relu=False, name='Mconv7_stage{}'.format(stage))
    #         )
    #         self.feed('Mconv7_stage{}'.format(stage), 'conv5_3_CPM')

    #     self.feed('Mconv7_stage{}'.format(6))

    def loss_l1(self):
        l1s = []
        for layer_name in sorted(self.layers.keys()):
            if 'Mconv7_stage' in layer_name or 'conv6_2_CPM' in layer_name:
                l1s.append(self.layers[layer_name])

        return l1s

    def loss_last(self):
        return self.get_output('Mconv7_stage{}'.format(6))

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'conv' in v.op.name and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        return vs
