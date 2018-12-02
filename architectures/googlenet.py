# googlenet.py
# Description: GoogleNet Implementation (Deep Convolution Network)

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
# Pre-req Scripts     : .common
# Compilation command : NA (Compiled by Internal call)
# Compilation time    : NA (depends of size of data)
# Execution command   : NA (Executed by Internal call)
# Execution time      : NA (depends of size of data)

# Headers
import tensorflow as tf
from .common import *

# GoogleNet, Going Deeper with Convolutions (By Google Inc.)
def inception(net, conv1_size, conv3_size, conv5_size, pool1_size, wd, is_training):

  with tf.variable_scope("conv_1"):
    conv1 = spatialConvolution(net, 1, 1, conv1_size, wd= wd)
    conv1 = batchNormalization(conv1, is_training= is_training)
    conv1 = tf.nn.relu(conv1)

  with tf.variable_scope("conv_3_1"):
    conv3 = spatialConvolution(net, 1, 1, conv3_size[0], wd= wd)
    conv3 = batchNormalization(conv3, is_training= is_training)
    conv3 = tf.nn.relu(conv3)

  with tf.variable_scope("conv_3_2"):
    conv3 = spatialConvolution(conv3, 3, 1, conv3_size[1], wd= wd)
    conv3 = batchNormalization(conv3, is_training= is_training)
    conv3 = tf.nn.relu(conv3)

  with tf.variable_scope("conv_5_1"):
    conv5 = spatialConvolution(net, 1, 1, conv5_size[0], wd= wd)
    conv5 = batchNormalization(conv5, is_training= is_training)
    conv5 = tf.nn.relu(conv5)

  with tf.variable_scope("conv_5_2"):
    conv5 = spatialConvolution(conv5, 5, 1, conv5_size[1], wd= wd)
    conv5 = batchNormalization(conv5, is_training= is_training)
    conv5 = tf.nn.relu(conv5)

  with tf.variable_scope("pool_1"):
    pool1= maxPool(net, 3, 1)
    pool1 = spatialConvolution(pool1, 1, 1, pool1_size, wd= wd)
    pool1 = batchNormalization(pool1, is_training= is_training)
    pool1 = tf.nn.relu(pool1)

  return tf.concat([conv1, conv3, conv5, pool1], 3)

def googlenet(net, num_output, wd, dropout_rate, is_training):

    with tf.variable_scope('conv1'):
      net = spatialConvolution(net, 7, 2, 64, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu (net)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('conv2'):
      net = spatialConvolution(net, 1, 1, 64, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    with tf.variable_scope('conv3'):
      net = spatialConvolution(net, 3, 1, 192, wd= wd)
      net = batchNormalization(net, is_training= is_training)
      net = tf.nn.relu(net)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('inception3a'):
      net = inception( net, 64, [96, 128], [16, 32], 32, wd= wd, is_training= is_training)

    with tf.variable_scope('inception3b'):
      net = inception( net, 128, [128, 192], [32, 96], 64, wd= wd, is_training= is_training)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('inception4a'):
      net = inception( net, 192, [96, 208], [16, 48], 64, wd= wd, is_training= is_training)

    with tf.variable_scope('inception4b'):
      net = inception( net, 160, [112, 224], [24, 64], 64, wd= wd, is_training= is_training)

    with tf.variable_scope('inception4c'):
      net = inception( net, 128, [128, 256], [24, 64], 64, wd= wd, is_training= is_training)

    with tf.variable_scope('inception4d'):
      net = inception( net, 112, [144, 288], [32, 64], 64, wd= wd, is_training= is_training)

    with tf.variable_scope('inception4e'):
      net = inception( net, 256, [160, 320], [32, 128], 128, wd= wd, is_training= is_training)

    net = maxPool(net, 3, 2)

    with tf.variable_scope('inception5a'):
      net= inception(net, 256, [160, 320], [32, 128], 128, wd= wd, is_training= is_training)

    with tf.variable_scope('inception5b'):
      net= inception(net, 384, [192, 384], [48, 128], 128, wd= wd, is_training= is_training)

    net= avgPool(net, 7, 1)

    net= flatten(net)

    net= tf.nn.dropout(net, dropout_rate)

    with tf.variable_scope('output'):
      net= fullyConnected(net, num_output, wd= wd)

    return net