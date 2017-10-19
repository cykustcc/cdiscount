#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cdiscount_densenet.py
# Author: Jiawei Chen <jwchen.maria@gmail.com>
r"""
File to run densenet model on cdiscount data.
reference: https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py

Example Usage: (assume you should in ../src)
python -m src.cdiscount_densenet --gpu=0,1,2,3 --densenet_depth=50

python -m src.cdiscount_densenet \
    --densenet_depth=50 \
    --pred_test=True \
    --model_path_for_pred=./train_log/train_log/cdiscount-densenet-d18/model-20000
"""
import numpy as np
import tensorflow as tf
import argparse
import os
import multiprocessing
import tqdm
import csv

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from google.apputils import app
import gflags

import socket

from .cdiscount_data import *
from .cdiscount_resnet_utils import *
from .common_utils import *

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('densenet_depth', 18,
                      'depth of densenet, should be one of [18, 34, 50, 101].')

gflags.DEFINE_integer('densenet_growth_rate', 12,
                      'growth rate of densenet, should be one of [12, 24, 32, 40].')

gflags.DEFINE_string('load', None,
                     'load model.')

gflags.DEFINE_string('gpu', None,
                     'specify which gpu(s) to be used.')

gflags.DEFINE_bool('pred_train', False,
                   'If true, run prediction on rain set without training.'
                   '(using existed model.)')

gflags.DEFINE_bool('pred_test', False,
                   'If true, run prediction on test set without training.'
                   '(using existed model.)')

gflags.DEFINE_bool('apply_augmentation', False,
                   'If true, Apply image augmentation. For training and'
                   'testing, we apply different augmentation')

gflags.DEFINE_string('model_path_for_pred', "",
                     'model path for prediction on test set.')

gflags.DEFINE_string('log_dir_name_suffix', "",
                     'suffix of the model checkpoint folder name.')

# BATCH_SIZE = 64
if socket.gethostname() == "ESC8000":
  BATCH_SIZE = 512
else:
  BATCH_SIZE = 32
INPUT_SHAPE = 180


class Model(ModelDesc):
  def __init__(self, depth, growth_rate):
    super(Model, self).__init__()
    self.N = int((depth - 4)  / 3)
    self.growthRate = growth_rate

  def _get_inputs(self):
    return [InputDesc(tf.float16, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, input_vars):
    image, label = input_vars
    # image = image / 128.0 - 1
    image = image_preprocess(image, bgr=False)

    def conv(name, l, channel, stride):
      return Conv2D(name, l, channel, 3, stride=stride,
              nl=tf.identity, use_bias=False,
              W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

    def add_layer(name, l):
      shape = l.get_shape().as_list()
      in_channel = shape[3]
      with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c)
        c = conv('conv1', c, self.growthRate, 1)
        l = tf.concat([c, l], 3)
      return l

    def add_transition(name, l):
      shape = l.get_shape().as_list()
      in_channel = shape[3]
      with tf.variable_scope(name) as scope:
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
      return l

    def densenet_block(name):
      with tf.variable_scope(name) as scope:

        for i in range(self.N):
          l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition(name + '-transition', l)
      return l

    def densenet_backbone(image, num_blocks, block_func):
      with argscope(Conv2D, nl=tf.identity, use_bias=False,
                    W_init=variance_scaling_initializer(mode='FAN_OUT')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(densenet_block, 'group0', block_func, 64, num_blocks[0], 1, first=True)
                  .apply(densenet_block, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(densenet_block, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(densenet_block, 'group3', block_func, 512, num_blocks[3], 2)
                  .BNReLU('bnlast')
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 5270, nl=tf.identity)())
      return logits

    def dense_net(name):
      l = conv('conv0', image, 16, 1)
      with tf.variable_scope('block1') as scope:

        for i in range(self.N):
          l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition1', l)

      with tf.variable_scope('block2') as scope:

        for i in range(self.N):
          l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition2', l)

      with tf.variable_scope('block3') as scope:

        for i in range(self.N):
          l = add_layer('dense_layer.{}'.format(i), l)
      l = BatchNorm('bnlast', l)
      l = tf.nn.relu(l)
      l = GlobalAvgPooling('gap', l)
      logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)

      return logits

    logits = dense_net("dense_net")
    loss = compute_loss_and_error(logits, label)
    wd_loss = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
    add_moving_summary(loss, wd_loss)
    self.cost = tf.add_n([loss, wd_loss], name='cost')

  def _get_optimizer(self):
      lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
      tf.summary.scalar('learning_rate', lr)
      return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, batch):
  isTrain = train_or_test == 'train'
  ds = Cdiscount(FLAGS.datadir, FLAGS.img_list_file, train_or_test,
                 shuffle=isTrain)
  if isTrain:
      ds = PrefetchDataZMQ(ds, min(20, multiprocessing.cpu_count()))
  ds = BatchData(ds, batch, remainder=not isTrain)
  return ds


def get_config(model):
  nr_tower = max(get_nr_gpu(), 1)
  batch = BATCH_SIZE // nr_tower
  logger.info("Running on {} towers. Batch size per tower:{}".format(nr_tower,
                                                                     batch))
  # prepare dataset
  dataset_train = get_data('train', batch)
  dataset_val = get_data('val', batch)
  infs = [ClassificationError('wrong-top1', 'val-error-top1'),
          ClassificationError('wrong-top5', 'val-error-top5')]
  steps_per_epoch = dataset_train.size() // 3
  callbacks=[
    ModelSaver(),
    ScheduledHyperParamSetter('learning_rate',
                              [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]),
    HumanHyperParamSetter('learning_rate'),
  ]
  if nr_tower == 1:
    # single-GPU inference with queue prefetch
    callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
  else:
    # multi-GPU inference (with mandatory queue prefetch)
    callbacks.append(DataParallelInferenceRunner(
        dataset_val, infs, list(range(nr_tower))))

  return TrainConfig(
    dataflow=dataset_train,
    model=model,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    max_epoch=110,
    nr_tower=nr_tower
  )


def main(argv):
  if FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  model = Model(FLAGS.densenet_depth, FLAGS.densenet_growth_rate)
  model_name = ('cdiscount-densenet-d' + str(FLAGS.densenet_depth) + '-gr' +
        str(FLAGS.densenet_growth_rate) +
        str(FLAGS.log_dir_name_suffix))

  if FLAGS.pred_train:
    make_pred(model, model_name, 'train', FLAGS.model_path_for_pred,
        PRED_BATCH_SIZE, FLAGS.apply_augmentation)
  elif FLAGS.pred_test:
    make_pred(model, model_name, 'test', FLAGS.model_path_for_pred,
        PRED_BATCH_SIZE, FLAGS.apply_augmentation)
  else:
    logger.set_logger_dir(
        os.path.join('train_log', model_name))
    config = get_config(model)
    if FLAGS.load:
      config.session_init = get_model_loader(FLAGS.load)
    SyncMultiGPUTrainerParameterServer(config).train()
    #SimpleTrainer(config).train()


if __name__ == '__main__':
  app.run()

