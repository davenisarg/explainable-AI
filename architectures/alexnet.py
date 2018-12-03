# alexnet.py
# Description: Alexnet(Caffenet) Implementation (Deep Convolution Network)

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
from .common import *
from .model import *

#AlexNet architecture
def alexnet(net, num_classes, wd, dropout_rate, is_training):

  with tf.variable_scope('conv1'):
    net = spatialConvolution(net, 11, 4, 64, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu (net)
    #common.activation_summary(net)

  net = maxPool(net, 3, 2)

  with tf.variable_scope('conv2'):
    net = spatialConvolution(net, 5, 1, 192, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)
    #common.activation_summary(net)

  net = maxPool(net, 3, 2)

  with tf.variable_scope('conv3'):
    net = spatialConvolution(net, 3, 1, 384, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)
    #common.activation_summary(net)

  with tf.variable_scope('conv4'):
    net = spatialConvolution(net, 3, 1, 256, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('conv5'):
    net = spatialConvolution(net, 3, 1, 256, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  net = maxPool(net, 3, 2)

  net = flatten(net)

  with tf.variable_scope('fc1'):
    net = tf.nn.dropout(net, dropout_rate)
    net = fullyConnected(net, 4096, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('fc2'):
    net = tf.nn.dropout(net, dropout_rate)
    net = fullyConnected(net, 4096, wd= wd)
    net = batchNormalization(net, is_training= is_training)
    net = tf.nn.relu(net)

  with tf.variable_scope('output'):
    net = fullyConnected(net, num_classes, wd= wd)

  return net
