import sys
import abc

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# def make_var(name, shape, trainable=True, initializer=None):
#         '''Creates a new TensorFlow variable.'''
#         return tf.get_variable(name, shape, trainable=trainable, initializer=initializer)

# def batch_normalization(self, input, name, scale_offset=True, relu=False):
#     # NOTE: Currently, only inference is supported
#     with tf.variable_scope(name) as scope:
#         shape = [input.get_shape()[-1]]
#         if scale_offset:
#             scale = make_var('scale', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
#             offset = make_var('offset', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
#         else:
#             scale, offset = (None, None)
#         output = tf.nn.batch_normalization(
#             input,
#             mean=make_var('mean', shape=shape, initializer=tf.contrib.layers.xavier_initializer()),
#             variance=make_var('variance', shape=shape, initializer=tf.contrib.layers.xavier_initializer()),
#             offset=offset,
#             scale=scale,
#             # TODO: This is the default Caffe batch norm eps
#             # Get the actual eps from parameters
#             variance_epsilon=1e-5,
#             name=name)
#         if relu:
#             output = tf.nn.relu(output)
#         return output

def is_str(inp):
    try:
        is_str = isinstance(inp, basestring)
    except NameError:
        is_str = isinstance(inp, str)
    return is_str

def layer(op):
    '''
    Decorator for composable network layers.
    '''
    @slim.add_arg_scope
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        layer_input = None
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]

            if tf.executing_eagerly():
                layer_input = [layer_input]
        else:
            layer_input = list(self.terminals)

        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.eager_exec_list.append(name)
        # for eager mode layer_input must be an instance of list
        self.layer_inputs[name] = layer_input
        self.layer_ops[name] = layer_output
        
        # This output is now the input for the next layer.
        if tf.executing_eagerly():
            # in eager mode feeds name string
            self.feed(name)
        else:
            self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class BaseNetwork(tf.keras.Model):
    def __init__(self, inputs, eager_output_names, trainable=True):
        '''
        Arguments:
            tf {[type]} -- [description]
            inputs:
                [in graph mode]
                    inputs should be a dictionary maps key to placeholder tensors
                [in eager mode]
                    inputs should be a list of name string
        
        Keyword Arguments:
            eager_output_names {list}
                [only used in eager mode]
            trainable {bool}
                Control if batch norm trainable
        '''
    
        super(BaseNetwork, self).__init__()

        if tf.executing_eagerly():
            self.layer_ops = {}
            for inp_name in inputs:
                self.layer_ops[inp_name] = None
        else:
            # The input nodes for this network (static mode)
            self.inputs = inputs
            # Mapping from layer names to layers
            self.layer_ops = dict(inputs)
            # Switch variable for dropout
            self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        # list of name sorted in execution order
        self.eager_exec_list = []
        self.eager_output_names = eager_output_names
        # temporary keeper while model get executed
        self.eager_name_to_output = {}
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to it's inputs
        self.layer_inputs = {}
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()
            
    @abc.abstractmethod
    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def call(self, inputs):
        '''
            execute the network. (only for eager mode) 
        
            inputs must be a dictionary which maps from name to input value
        '''
        for key in inputs.keys():
            value = inputs[key]

            if value.dtype != 'float32':
                raise RuntimeWarning("Input type (%s) is not float32, it will cause Batchnormalization crash." % value.dtype)
                
            self.eager_name_to_output[key] = value

        for lname in self.eager_exec_list:
            inp_vals = []
            for inp_name in self.layer_inputs[lname]:
                inp_vals.append(self.eager_name_to_output[inp_name])

            if len(inp_vals) == 1:
                self.eager_name_to_output[lname] = self.layer_ops[lname](inp_vals[0])
            else:
                self.eager_name_to_output[lname] = self.layer_ops[lname](inp_vals)

        outputs = []
        try:
            for out_name in self.eager_output_names:
                outputs.append(self.eager_name_to_output[out_name])
        except KeyError:
            raise KeyError('output name "%s" is not found in the graph, neither an input node' % out_name)
        

        return outputs

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        def check_if_str(inp):
            if is_str(inp):
                if inp not in self.layer_ops:
                    raise KeyError('Unknown layer name fed: %s' % inp)
            else:
                raise RuntimeError('Feed non string in eager mode')
        ####
        assert len(args) != 0
        self.terminals = []
        if tf.executing_eagerly():
            for fed_layer in args:
                if isinstance(fed_layer, list):
                    for layer in fed_layer:
                        check_if_str(layer)
                else:
                    check_if_str(fed_layer)
                self.terminals.append(fed_layer)
        else:
            for fed_layer in args:
                if is_str(fed_layer):
                    try:
                        fed_layer = self.layer_ops[fed_layer]
                    except KeyError:
                        raise KeyError('Unknown layer name fed: %s' % fed_layer)
                self.terminals.append(fed_layer)
        return self

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layer_ops.items()) + 1
        return '%s_%d' % (prefix, ident)

    def get_output(self, name=None):
        '''(for graph mode) Returns the current network output.'''
        if not name:
            return self.terminals[-1]
        else:
            return self.layer_ops[name]

    def op_executor(self, op, inputs):
        if tf.executing_eagerly():
            return op
        else:
            return op(inputs)

    @layer
    def add(self, inputs, name):
        def add(inps): return tf.add_n(inps, name=name)
        return self.op_executor(add, inputs)

    @layer
    def concat(self, inputs, axis, name):
        def concat(inps): return tf.concat(axis=axis, values=inps, name=name)
        return self.op_executor(concat, inputs)

    @layer
    def relu(self, inputs, name):
        def relu(inps): return tf.nn.relu(inps, name=name)
        return self.op_executor(relu, inputs)

    @layer
    def relu6(self, inputs, name):
        def relu6(inps): return tf.nn.relu6(inps, name=name)
        return self.op_executor(relu6, inputs)

    @layer
    def batchnorm(self, inputs, name, axis=-1, momentum=0.99, epsilon=0.001):
        def batchnorm(inps):
            return tf.keras.layers.BatchNormalization(axis=axis,
                                                      momentum=momentum,
                                                      epsilon=epsilon,
                                                      center=True,
                                                      scale=True,
                                                      trainable=self.trainable,
                                                      name=name)
        return self.op_executor(batchnorm, inputs)

    @layer
    def upsample(self, inputs, factor, name):
        def upsample(inps):
            return tf.image.resize_bilinear(inps, [int(inps.shape[1]) * factor, int(inps.shape[2]) * factor], name=name)
        return self.op_executor(upsample, inputs)
            
    @layer
    def softmax(self, inputs, name):
        def softmax(inps): return tf.nn.softmax(inps, name=name)
        return self.op_executor(softmax, inputs)

    @layer
    def max_pool(self, inputs, ksize, strides, name, padding='SAME'):
        '''
        Arguments:
            inputs {[type]} -- input
            ksize {[type]} -- ksize should be (kh, kw) or [kh, kw]
            strides {[type]} -- strides should be (sh, sw) or [sh, sw]
            name {[type]} -- name
        
        Keyword Arguments:
            padding {str} -- [description] (default: {'SAME'})
        '''
        def max_pool(inps):
            return tf.nn.max_pool(inps,
                                  ksize=[1, ksize[0], ksize[1], 1],
                                  strides=[1, strides[0], strides[1], 1],
                                  padding=padding,
                                  name=name)
        return self.op_executor(max_pool, inputs)

    @layer
    def avg_pool(self, inputs, ksize, strides, name, padding='SAME'):
        '''
        Arguments:
            inputs {[type]} -- input
            ksize {[type]} -- ksize should be (kh, kw) or [kh, kw]
            strides {[type]} -- strides should be (sh, sw) or [sh, sw]
            name {[type]} -- name
        
        Keyword Arguments:
            padding {str} -- [description] (default: {'SAME'})
        '''
        def avg_pool(inps):
            return tf.nn.avg_pool(inps,
                                  ksize=[1, ksize[0], ksize[1], 1],
                                  strides=[1, strides[0], strides[1], 1],
                                  padding=padding,
                                  name=name)
        return self.op_executor(avg_pool, inputs)

    @layer
    def conv(self,
             inputs,
             filters,
             kernel_size,
             strides=(1, 1),
             padding='SAME',
             dilation_rate=(1, 1),
             activation=tf.nn.relu6,
             use_bn=True,
             kernel_initializer=None,
             kernel_regularizer=None,
             name=None):
        '''
            [use_bn]
                True -> use batch norm
                False -> use bias
                None -> both not
            [activation]
                only support activation implemented in "BaseNetwork"
                - self.relu
                - self.relu6
                    .
                    .
                    .
        '''
        use_bias = (not use_bn) and (use_bn is not None)

        convol = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilation_rate,
                                        activation=None,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        name='Conv2D')

        bn = None
        if use_bn:
            bn = tf.keras.layers.BatchNormalization(axis=-1,
                                                    momentum=0.99,
                                                    epsilon=0.001,
                                                    center=True,
                                                    scale=True,
                                                    trainable=self.trainable,
                                                    name='BatchNorm')

        def conv(inps):
            res = convol(inps)
            if bn is not None:
                res = bn(res)
            if activation is not None:
                res = activation(res)
            return res

        return self.op_executor(conv, inputs)

    @layer
    def separable_conv(self,
                       inputs,
                       filters,
                       kernel_size,
                       strides=(1, 1),
                       padding='SAME',
                       activation=tf.nn.relu6,
                       use_bn=True,
                       kernel_initializer=None,
                       kernel_regularizer=None,
                       name=None):
        use_bias = (not use_bn) and (use_bn is not None)

        depth_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding=padding,
                                                        depth_multiplier=1,
                                                        activation=None,
                                                        use_bias=use_bias,
                                                        depthwise_initializer=kernel_initializer,
                                                        depthwise_regularizer=kernel_regularizer,
                                                        name='Depthwise2D')
        depth_bn = None
        if use_bn:
            depth_bn = tf.keras.layers.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            trainable=self.trainable,
                                                            name='Depthwise2D_BatchNorm')

        conv = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=1, 
                                        strides=(1,1),
                                        padding=padding,
                                        dilation_rate=(1,1),
                                        activation=None,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        name='Pointwise2D')

        conv_bn = None
        if use_bn:
            conv_bn = tf.keras.layers.BatchNormalization(axis=-1,
                                                            momentum=0.99,
                                                            epsilon=0.001,
                                                            center=True,
                                                            scale=True,
                                                            trainable=self.trainable,
                                                            name='Pointwise2D_BatchNorm')

        def separable_conv(inps):
            res = depth_conv(inps)
            if depth_bn is not None:
                res = depth_bn(res)
            if activation is not None:
                res = activation(res)

            res = conv(res)
            if conv_bn is not None:
                res = conv_bn(res)
            if activation is not None:
                res = activation(res)
            return res

        return self.op_executor(separable_conv, inputs)

class TestNetwork(BaseNetwork):
    def setup(self):
        '''[summary]
        input not float32 will cause batchnorm crash in eager mode
        input must be converted to a tensor object (numpy array is not allowed)
        '''

        self.feed('input1', 'input2').concat(3, name='input_concat')
        with slim.arg_scope([self.separable_conv, self.conv], 
                            activation=tf.nn.relu6,
                            padding='SAME',
                            use_bn=True,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005)):
            self.feed('input1').separable_conv(128, 3, name='a')
            self.feed('input2').separable_conv(128, 3, name='b')
            self.feed('input_concat').separable_conv(256, 3, strides=(2,2), name='down_2')
            self.max_pool((3, 3), (2, 2), name='down_4')
            self.avg_pool((3, 2), (2, 2), name='down_8')

            self.feed('a', 'b').add(name='a_plus_b')

            self.feed('down_8').upsample(8, name='c')

            self.feed('a', 'b', 'a_plus_b', 'c').concat(3, name='concat')
            self.conv(128, 3, activation=None, name='final_conv')
            self.softmax(name='result')

def non_eager_mode_test():
    inp1 = tf.placeholder(tf.float32, shape=(8,128,128,32))
    inp2 = tf.placeholder(tf.float32, shape=(8,128,128,64))
    model = TestNetwork({'input1':inp1, 'input2':inp2}, ['result'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run([model.get_output('result')], 
                   {
                        inp1:np.random.rand(8,128,128,32), 
                        inp2:np.random.rand(8,128,128,64),
                   })
    print(res)
