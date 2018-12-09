# model.py
# Description: Helper file with necessary methods/functions for simulating model building, training and validation.

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
# Modules             : NA
# Pre-req Scripts     : .common, .googlenet, .resnet, .alexnet
# Compilation command : NA (Compiled by Internal call)
# Compilation time    : NA (depends of size of data)
# Execution command   : NA (Executed by Internal call)
# Execution time      : NA (depends of size of data)


# Headers
from .alexnet import *
from .resnet import *
from .googlenet import *
from .common import SAVE_VARIABLES

# Simulates specified model
class model:

  """
  Args:
    inputs: input tensor
    labels: label tensor
    loss: a measure of loss
    optimizer: a SGD optimizer to calculate gradients
    wd: a weight decay policy
    architecture: deep learning architecture
    depth: further determining the details of architecture
    num_classes: number of classes
    is_trainig: a placeholder for determing either the model is in the training or testing modes.
    transfer_mode: determine the type of transfer learning
    top_n: top-n accuracies to report
    max_to_keep: determines how many snapshot we want to store
    num_gpus: number of available GPUs
  """
  def __init__(self, inputs, labels, loss, optimizer, wd, architecture, depth, num_classes, is_training, transfer_mode, top_n= 5, max_to_keep= 5, num_gpus=1):
    self.architecture= architecture
    self.depth= depth
    self.num_classes= num_classes
    self.is_training= is_training
    self.transfer_mode= transfer_mode
    self.saver= None
    self.pretrained_loader= None
    self.loss= loss
    self.optimizer= optimizer
    self.wd= wd
    self.inputs= inputs
    self.labels= labels
    self.top_n= top_n
    self.max_to_keep= max_to_keep
    self.num_gpus= num_gpus

  """
  Calculate the average gradient for each shared variable across all the GPUs.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
  """
  def average_gradients(self, tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      grads = []
      for g, _ in grad_and_vars:
        expanded_g = tf.expand_dims(g, 0)

        grads.append(expanded_g)

      grad = tf.concat(axis= 0, values= grads)
      grad = tf.reduce_mean(grad, 0)

      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  """
    This methods build a graph for one tower.
    Args:
      device: a specefic device to put the tower on
    Returns:
     several operations for calculating gradients, doing batch norm uodates, and calculating accuracies
  """
  def get_grads(self, device):

    with tf.device(device):
      logits= self.inference()
      probabilities= tf.nn.softmax(logits, name='output')

      # Top-1 accuracy
      top1acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32))
      # Top-n accuracy
      topnacc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, self.top_n), tf.float32))

      cross_entropy_mean = self.loss(logits, self.labels)

      # Get all the regularization lesses and add them
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      reg_loss = tf.add_n(regularization_losses)

      #Add summary
      tf.summary.scalar('Regularization Loss', reg_loss)

      # Compute the total loss
      total_loss = tf.add(cross_entropy_mean, reg_loss)

      tf.summary.scalar('Total Loss', total_loss)
      tf.summary.scalar('Top-1 Accuracy', top1acc)
      tf.summary.scalar('Top-' + str(self.top_n) + ' Accuracy', topnacc)

      # Gather batch normaliziation update operations
      batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

      last_grads = self.optimizer.compute_gradients(total_loss,var_list= tf.get_collection(SAVE_VARIABLES, scope='output'))
      grads = self.optimizer.compute_gradients(total_loss)

    return grads, last_grads, batchnorm_updates, cross_entropy_mean, top1acc, topnacc

  """
    Args:
      nothing.
    Returns:
      averaged gradients and other operations to run batchnorms and calculating accuracies.
  """
  def multigpu_grads(self):

    # Calculate the gradients for each model.
    tower_grads = []
    tower_last_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(self.num_gpus):
          with tf.name_scope('Tower_%d' % i) as scope:

            grads, last_grads, batchnorm_updates, cross_entropy_mean, top1acc, topnacc = self.get_grads('/gpu:%d' % i)

            tf.get_variable_scope().reuse_variables()

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            tower_last_grads.append(last_grads)
            tower_grads.append(grads)
    return self.average_gradients(tower_grads), self.average_gradients(tower_last_grads), batchnorm_updates, cross_entropy_mean, top1acc, topnacc

  """
    Calculate the average gradient for each shared variable across all the GPUs.
    Args:
      nothing.
    Returns:
      the necessary operations to train the network
  """
  def train_ops(self):
    if self.num_gpus==1:
      grads, last_grads, batchnorm_updates, cross_entropy_mean, top1acc, topnacc = self.get_grads('/gpu:0')
    else:
      grads, last_grads, batchnorm_updates, cross_entropy_mean, top1acc, topnacc = self.multigpu_grads()


    # Setup the training
    if self.transfer_mode[0]==1:
      train_op = self.optimizer.apply_gradients(last_grads)
    if self.transfer_mode[0]==2 or self.transfer_mode[0]==0:
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      train_op = tf.group(self.optimizer.apply_gradients(grads), batchnorm_updates_op)
    if self.transfer_mode[0]==3:
      train_op = tf.cond(tf.less(tf.get_collection(SAVE_VARIABLES, scope="epoch_number")[0],self.transfer_mode[1]),
              lambda: tf.group(self.optimizer.apply_gradients(last_grads),*batchnorm_updates), lambda: tf.group(self.optimizer.apply_gradients(grads),*batchnorm_updates))

    return [train_op, cross_entropy_mean, top1acc, topnacc]

  """
    This method build a specefic architecture.
    Args:
      nothing.
    Returns:
      The architecture
  """
  def inference(self):
    if self.architecture.lower()=='alexnet':
      return alexnet(self.inputs, self.num_classes, self.wd, tf.where(self.is_training, 0.5, 1.0), self.is_training)
    elif self.architecture.lower()=='resnet':
      return resnet(self.inputs, self.num_classes, self.wd, self.is_training, self.transfer_mode, self.depth)
    elif self.architecture.lower()=='googlenet':
        return googlenet(self.inputs, self.num_classes, self.wd, tf.where(self.is_training, 0.4, 1.0), self.is_training)

  """
    evaluating the network.
    Args:
      loss: a loss measure
      inference_only: determines evaulating or just predicting
    Returns:
     List of operations
  """
  def evaluate_ops(self, inference_only):
    with tf.device('/gpu:0'):
      logits = self.inference()

      topn = tf.nn.top_k(tf.nn.softmax(logits), self.top_n)
      topnind= topn.indices
      topnval= topn.values
      if not inference_only:
        cross_entropy_mean = self.loss(logits, self.labels)

        top1acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32))

        topnacc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, self.top_n), tf.float32))

        return [cross_entropy_mean, top1acc, topnacc, topnind, topnval]
      else:
        return [topnind,topnval]

  """
    This method saves the current parameters file
    Args:
      sess: the active session of the graph
      path: path to save
      global_step: a number specifying different snapshots
    Returns:
      nothing
  """
  def save(self, sess, path, global_step):
    # Create a saver.
    if self.saver is None:
      self.saver = tf.train.Saver(tf.get_collection(SAVE_VARIABLES), max_to_keep= self.max_to_keep)
    self.saver.save(sess, path, global_step)

  """ This method loads a set of pretrained parameters saved file.
    Args:
      sess: an active Tensorflow session
      path: path to load the snapshot
    Returns:
      nothing
  """
  def load(self, sess, path):
    ckpt = tf.train.get_checkpoint_state(path)
    if self.pretrained_loader is None:
      if self.transfer_mode[0]==0:
        self.pretrained_loader = tf.train.Saver(tf.get_collection(SAVE_VARIABLES))
      else:
        self.pretrained_loader = tf.train.Saver(var_list= self.exclude())
    self.pretrained_loader.restore(sess, ckpt.model_checkpoint_path)

  """
     for transfer learning
  """
  def exclude(self):
    var_list = tf.get_collection(SAVE_VARIABLES)
    to_remove = []
    for var in var_list:
      if var.name.find("output")>=0 or var.name.find("epoch_number")>=0:
        to_remove.append(var)
    print(to_remove)
    for x in to_remove:
      var_list.remove(x)
    return var_list
