from basicnetwork import BaseNetwork


import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class MobilenetThin(BaseNetwork):
    def setup(self):
        input_name = 'image'
        num_class = 21
        min_filters = 8
        filters_multiplier = 1.0
        filters_multiplier2 = 1.0
        scale_f = lambda filters: max(int(filters * filters_multiplier), min_filters)
        scale_f2 = lambda filters: max(int(filters * filters_multiplier2), min_filters)

        with slim.arg_scope([self.separable_conv, self.conv], 
                            activation=tf.nn.relu6,
                            padding='SAME',
                            use_bn=True,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005)):

            with tf.variable_scope(None, 'MobilenetV1'):
                (self.feed(input_name)
                .conv(scale_f(32), 3, (2,2), name='Conv2d_0')
                .separable_conv(scale_f(64), 3, (1,1), name='Conv2d_1')
                .separable_conv(scale_f(128), 3, (2,2), name='Conv2d_2')
                .separable_conv(scale_f(128), 3, (1,1), name='Conv2d_3')
                .separable_conv(scale_f(256), 3, (2,2), name='Conv2d_4')
                .separable_conv(scale_f(256), 3, (1,1), name='Conv2d_5')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_6')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_7')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_8')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_9')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_10')
                .separable_conv(scale_f(512), 3, (1,1), name='Conv2d_11')
                # .separable_conv(scale_f(1024), 3, (2,2), name='Conv2d_12')
                # .separable_conv(scale_f(1024), 3, (1,1), name='Conv2d_13')
                )
            
            (self.feed('Conv2d_3').max_pool((2,2), (2,2), name='Conv2d_3_pool'))

            (self.feed('Conv2d_3_pool', 'Conv2d_7', 'Conv2d_11')
            .concat(3, name='feat_concat'))

            feature_lv = 'feat_concat'
            with tf.variable_scope(None, 'Openpose'):
                prefix = 'MConv_Stage1'
                (self.feed(feature_lv)
                .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_1')
                .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_2')
                .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_3')
                .separable_conv(scale_f2(512), 1, (1,1), name=prefix + '_L1_4')
                .separable_conv(num_class, 1, (1,1), activation=None, name=prefix + '_L1_5'))

                for stage_id in range(5):
                    prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
                    prefix = 'MConv_Stage%d' % (stage_id + 2)
                    (self.feed(prefix_prev + '_L1_5',
                               feature_lv)
                    .concat(3, name=prefix + '_concat')
                    .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_1')
                    .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_2')
                    .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_3')
                    .separable_conv(scale_f2(128), 3, (1,1), name=prefix + '_L1_4')
                    .separable_conv(num_class, 1, (1,1), activation=None, name=prefix + '_L1_5'))

def non_eager_mode_test():
    inp1 = tf.placeholder(tf.float32, shape=(8,256,256,3))
    model = MobilenetThin({'image':inp1}, ['MConv_Stage6_L1_5'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run([model.get_output('MConv_Stage6_L1_5')], 
                   {
                        inp1: np.random.rand(8,256,256,3), 
                   })
    print(res)

def eager_mode_test():
    tf.enable_eager_execution()
    model = MobilenetThin(['image'], ['MConv_Stage6_L1_5'])
    res = model({
                    'image':tf.cast(np.random.rand(8,256,256,3), tf.float32), 
               })
    print(res)

if __name__ == "__main__":
    eager_mode_test()