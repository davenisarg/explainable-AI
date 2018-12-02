# common.py
# Description: helper function file for each model training
#            : Contains functions/methods for batch normalization, flatten, max pool, avg pool, fully connected, spatial Convolution etc.

# Metrics:
# Date/Time           : 2018-12-2 17:57:18
# Hostname            : ambrose
# OS                  : Ubuntu 18.04.1 LTS
# Kernel              : 4.15.0-39-generic
# RAM                 : 12 GB
# CPU model           : Intel® Core i5-3337U CPU @ 1.80GHz
# CPU/Core count      : 4
# Author information  : Nisarg Dave (nisargd@mtu.edu)
# Source code license : GPL v3 (https://www.gnu.org/licenses/gpl-3.0.txt)
# Software/Language   : Python 3.7.0 (https://www.python.org/downloads/release/python-370/)
# Version             : 3.7.0 (Ubuntu 18.04.1 LTS)
# Pre-req/Dependency  : Python 3.x.x (with following modules/packages/libraries)
# Modules             : tensorflow
# Compilation command : NA (Compiled by Internal call)
# Compilation time    : NA (depends of size of data)
# Execution command   : NA (Executed by Internal call)
# Execution time      : NA (depends of size of data)


# Headers
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
SAVE_VARIABLES = 'save_variables'

# variable initialization
def _get_variable(name,
                  shape,
                  initializer,
                  regularizer= None,
                  dtype= 'float',
                  trainable= True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
      var = tf.get_variable(name,
                           shape= shape,
                           initializer= initializer,
                           dtype= dtype,
                           regularizer= regularizer,
                           collections= collections,
                           trainable= trainable)

    return var

# Batch Normalization layer
def batchNormalization(x, is_training, decay= 0.9, epsilon= 0.001, inference_only= False):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer= tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer= tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer= tf.zeros_initializer,
                                trainable= False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer= tf.ones_initializer,
                                    trainable= False)

    # While training


    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, decay)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS , update_moving_variance)
    return tf.cond(is_training, lambda: tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon), lambda: tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon))
    #return tf.contrib.layers.batch_norm(x, decay= decay, epsilon= epsilon, is_training= is_training)

# Flatten Layer
def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1,len(shape)):
      dim*=shape[i]
    return tf.reshape(x, [-1, dim])

# Threshold Layer (ReLU)
def treshold(x, treshold):
    return tf.cast(x > treshold, x.dtype) * x

# Fully Connected layer
def fullyConnected(x, num_units_out, wd= 0.0, weight_initializer= None, bias_initializer= None, inference_only= False):
    num_units_in = x.get_shape()[1]

    stddev = 1./tf.sqrt(tf.cast(num_units_out, tf.float32))
    if weight_initializer is None:
      weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
    if bias_initializer is None:
      bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)

    weights = _get_variable('weights',
                            [num_units_in, num_units_out], weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    biases = _get_variable('biases',
                           [num_units_out], bias_initializer)

    return tf.nn.xw_plus_b(x, weights, biases)

# Convolution Layer
def spatialConvolution(x, ksize, stride, filters_out, wd= 0.0, weight_initializer= None, bias_initializer= None, inference_only= False):
    filters_in = x.get_shape()[-1]
    stddev = 1./tf.sqrt(tf.cast(filters_out, tf.float32))
    if weight_initializer is None:
      weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
    if bias_initializer is None:
      bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)

    shape = [ksize, ksize, filters_in, filters_out]
    weights = _get_variable('weights',
                            shape, weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding= 'SAME')
    biases = _get_variable('biases', [filters_out],  bias_initializer)

    return tf.nn.bias_add(conv, biases)

# Max Pooling Layer
def maxPool(x, ksize, stride):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

# Average Pooling Layer
def avgPool(x, ksize, stride, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding= padding)