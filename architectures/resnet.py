# resnet.py
# Description: ResNet Implementation (Deep Residual Network with Depth 18,34,50,101,152)

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
# Pre-req Scripts     : .common, .model
# Compilation command : NA (Compiled by Internal call)
# Compilation time    : NA (depends of size of data)
# Execution command   : NA (Executed by Internal call)
# Execution time      : NA (depends of size of data)

# Headers
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from .common import *
from .model import *

# ResNet Stack
def resnetStack(net, num_blocks, stack_stride, block_filters_internal, bottleneck, wd= 0.0, is_training= True):

  for n in range(num_blocks):
    s = stack_stride if n == 0 else 1
    block_stride = s

    with tf.variable_scope('block%d' % (n + 1)):
      net = resnetBlock(net, bottleneck, block_filters_internal, block_stride, wd= wd, is_training= is_training)

  return net

# ResNet Block
def resnetBlock(net, bottleneck, block_filters_internal, block_stride, wd= 0.0, is_training= True):

  filters_in = net.get_shape()[-1]

  m = 4 if bottleneck else 1
  filters_out = m * block_filters_internal

  shortcut = net  # branch 1

  conv_filters_out = block_filters_internal

  conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)

  if bottleneck:

    with tf.variable_scope('a'):
      net = spatialConvolution(net, 1, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('b'):
      net = spatialConvolution(net, 3, 1, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('c'):
      net = spatialConvolution(net, 1, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)

  else:

    with tf.variable_scope('A'):
      net = spatialConvolution(net, 3, block_stride, conv_filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('B'):
      net = spatialConvolution(net, 3, 1, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      net = batchNormalization(net, is_training= is_training)

  with tf.variable_scope('shortcut'):

    if filters_out != filters_in or block_stride != 1:
      shortcut = spatialConvolution(shortcut, 1, block_stride, filters_out, weight_initializer= conv_weight_initializer, wd= wd)
      shortcut = batchNormalization(shortcut, is_training= is_training)

  return tf.nn.relu(net + shortcut)


# ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
# Build the model based on the depth specified
def resnet(net, num_classes, wd, is_training, transfer_mode, depth):

  num_blockes= []

  bottleneck= False

  if depth == 18:
    num_blocks= [2, 2, 2, 2]

  elif depth == 34:
    num_blocks= [3, 4, 6, 3]

  elif depth == 50:
    num_blocks= [3, 4, 6, 3]
    bottleneck= True

  elif depth == 101:
    num_blocks= [3, 4, 23, 3]
    bottleneck= True

  elif depth == 152:
    num_blocks= [3, 8, 36, 3]
    bottleneck= True

  return getModel(net, num_classes, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= transfer_mode)

# Helper fn
def getModel(net, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
            bottleneck= True, transfer_mode= False):

  conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)

  fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)

  with tf.variable_scope('scale1'):
    net = spatialConvolution(net, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('scale2'):
    net = maxPool(net, 3, 2)
    net = resnetStack(net, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale3'):
    net = resnetStack(net, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale4'):
    net = resnetStack(net, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)

  with tf.variable_scope('scale5'):
    net = resnetStack(net, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)

  net = tf.reduce_mean(net, reduction_indices= [1, 2], name= "avg_pool")

  with tf.variable_scope('output'):
    net = fullyConnected(net, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)

  return net