#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_inceptionv3.py
# Author: Yukun Chen <cykustc@gmail.com>

import cv2
import argparse
import os
import tensorflow as tf
import multiprocessing
import socket

from google.apputils import app
import gflags

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.dataflow import dataset

from .cdiscount_data import *
from .common_utils import *

FLAGS = gflags.FLAGS

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

if socket.gethostname() == "ESC8000":
  BATCH_SIZE={
    'cdiscount-inceptionv3' : 142, #1792
  }
  PRED_BATCH_SIZE = 300
else: #home, 8GB gpu memory.
  BATCH_SIZE={
    'cdiscount-inceptionv3' : 128, #1792
  }
  PRED_BATCH_SIZE = 192
INPUT_SHAPE = 180

LEARNING_RATE={
  'cdiscount-inceptionv3' : [(5, 0.01), (9, 0.08), (12, 0.006),
														 (17, 0.003), (22, 1e-3), (36, 2e-4),
														 (41, 8e-5), (48, 1e-5), (53, 2e-6)],
}


class Model(ModelDesc):
  def _get_inputs(self):
    return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, inputs):
    image, label = inputs
    image = image / 255.0   # ?

    def proj_kk(l, k, ch_r, ch, stride=1):
      l = Conv2D('conv{0}{0}r'.format(k), l, ch_r, 1)
      return Conv2D('conv{0}{0}'.format(k), l, ch, k, stride=stride,
                    padding='VALID' if stride > 1 else 'SAME')

    def proj_233(l, ch_r, ch, stride=1):
      l = Conv2D('conv233r', l, ch_r, 1)
      l = Conv2D('conv233a', l, ch, 3)
      return Conv2D('conv233b', l, ch, 3, stride=stride,
                    padding='VALID' if stride > 1 else 'SAME')

    def pool_proj(l, ch, pool_type):
      if pool_type == 'max':
          l = MaxPooling('maxpool', l, 3, 1)
      else:
          l = AvgPooling('maxpool', l, 3, 1, padding='SAME')
      return Conv2D('poolproj', l, ch, 1)

    def proj_77(l, ch_r, ch):
      return (LinearWrap(l)
              .Conv2D('conv77r', ch_r, 1)
              .Conv2D('conv77a', ch_r, [1, 7])
              .Conv2D('conv77b', ch, [7, 1])())

    def proj_277(l, ch_r, ch):
      return (LinearWrap(l)
              .Conv2D('conv277r', ch_r, 1)
              .Conv2D('conv277aa', ch_r, [7, 1])
              .Conv2D('conv277ab', ch_r, [1, 7])
              .Conv2D('conv277ba', ch_r, [7, 1])
              .Conv2D('conv277bb', ch, [1, 7])())

    with argscope(Conv2D, nl=BNReLU, use_bias=False),\
          argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
      l = (LinearWrap(image)
           .Conv2D('conv0', 32, 3, stride=2, padding='VALID')  # 299
           .Conv2D('conv1', 32, 3, padding='VALID')  # 149
           .Conv2D('conv2', 64, 3, padding='SAME')  # 147
           .MaxPooling('pool2', 3, 2)
           .Conv2D('conv3', 80, 1, padding='SAME')  # 73
           .Conv2D('conv4', 192, 3, padding='VALID')  # 71
           .MaxPooling('pool4', 3, 2)())  # 35

      with tf.variable_scope('incep-35-256a'):
        l = tf.concat([
            Conv2D('conv11', l, 64, 1),
            proj_kk(l, 5, 48, 64),
            proj_233(l, 64, 96),
            pool_proj(l, 32, 'avg')
        ], 3, name='concat')
      with tf.variable_scope('incep-35-288a'):
        l = tf.concat([
            Conv2D('conv11', l, 64, 1),
            proj_kk(l, 5, 48, 64),
            proj_233(l, 64, 96),
            pool_proj(l, 64, 'avg')
        ], 3, name='concat')
      with tf.variable_scope('incep-35-288b'):
        l = tf.concat([
            Conv2D('conv11', l, 64, 1),
            proj_kk(l, 5, 48, 64),
            proj_233(l, 64, 96),
            pool_proj(l, 64, 'avg')
        ], 3, name='concat')
      # 35x35x288
      with tf.variable_scope('incep-17-768a'):
        l = tf.concat([
            Conv2D('conv3x3', l, 384, 3, stride=2, padding='VALID'),
            proj_233(l, 64, 96, stride=2),
            MaxPooling('maxpool', l, 3, 2)
        ], 3, name='concat')
      with tf.variable_scope('incep-17-768b'):
        l = tf.concat([
            Conv2D('conv11', l, 192, 1),
            proj_77(l, 128, 192),
            proj_277(l, 128, 192),
            pool_proj(l, 192, 'avg')
        ], 3, name='concat')
      for x in ['c', 'd']:
        with tf.variable_scope('incep-17-768{}'.format(x)):
          l = tf.concat([
              Conv2D('conv11', l, 192, 1),
              proj_77(l, 160, 192),
              proj_277(l, 160, 192),
              pool_proj(l, 192, 'avg')
          ], 3, name='concat')
      with tf.variable_scope('incep-17-768e'):
        l = tf.concat([
            Conv2D('conv11', l, 192, 1),
            proj_77(l, 192, 192),
            proj_277(l, 192, 192),
            pool_proj(l, 192, 'avg')
        ], 3, name='concat')
      # 17x17x768

      with tf.variable_scope('br1'):
        br1 = AvgPooling('avgpool', l, 5, 3, padding='VALID')
        br1 = Conv2D('conv11', br1, 128, 1)
        shape = br1.get_shape().as_list()
        br1 = Conv2D('convout', br1, 768, shape[1:3], padding='VALID')
        br1 = FullyConnected('fc', br1, 5270, nl=tf.identity)

      with tf.variable_scope('incep-17-1280a'):
        l = tf.concat([
            proj_kk(l, 3, 192, 320, stride=2),
            Conv2D('conv73', proj_77(l, 192, 192), 192, 3, stride=2, padding='VALID'),
            MaxPooling('maxpool', l, 3, 2)
        ], 3, name='concat')
      for x in ['a', 'b']:
        with tf.variable_scope('incep-8-2048{}'.format(x)):
          br11 = Conv2D('conv11', l, 320, 1)
          br33 = Conv2D('conv133r', l, 384, 1)
          br33 = tf.concat([
              Conv2D('conv133a', br33, 384, [1, 3]),
              Conv2D('conv133b', br33, 384, [3, 1])
          ], 3, name='conv133')

          br233 = proj_kk(l, 3, 448, 384)
          br233 = tf.concat([
              Conv2D('conv233a', br233, 384, [1, 3]),
              Conv2D('conv233b', br233, 384, [3, 1]),
          ], 3, name='conv233')

          l = tf.concat([
              br11, br33, br233,
              pool_proj(l, 192, 'avg')
          ], 3, name='concat')

      l = GlobalAvgPooling('gap', l)
      # 1x1x2048
      l = Dropout('drop', l, 0.8)
      logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)

    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br1, labels=label)
    loss1 = tf.reduce_mean(loss1, name='loss1')

    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss2 = tf.reduce_mean(loss2, name='loss2')

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

    # weight decay on all W of fc layers
    wd_w = tf.train.exponential_decay(0.00004, get_global_step_var(),
                                      80000, 0.7, True)
    wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

    self.cost = tf.add_n([0.4 * loss1, loss2, wd_cost], name='cost')
    add_moving_summary(loss1, loss2, wd_cost, self.cost)

  def _get_optimizer(self):
      lr = tf.get_variable('learning_rate', initializer=0.045, trainable=False)
      return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_config(model, model_name):
  nr_tower = max(get_nr_gpu(), 1)
  #batch = TOTAL_BATCH_SIZE // nr_tower
  batch = BATCH_SIZE[model_name]
  logger.info("Running on {} towers. Batch size per tower:{}".format(nr_tower,
                                                                     batch))

  dataset_train = get_data('train', batch)
  dataset_val = get_data('val', batch)
  infs = [ClassificationError('wrong-top1', 'val-error-top1'),
          ClassificationError('wrong-top5', 'val-error-top5')]
  if model_name in LEARNING_RATE.keys():
    learning_rate_schedule = LEARNING_RATE[model_name]
  else:
    learning_rate_schedule = [(5, 0.03), (9, 0.01), (12, 0.006),
														  (17, 0.003), (22, 1e-3), (36, 2e-4),
															(41, 8e-5), (48, 1e-5), (53, 2e-6)]
  logger.info("learning rate schedule: {}".format(learning_rate_schedule))
  callbacks=[
    ModelSaver(),
    ScheduledHyperParamSetter('learning_rate',
                              learning_rate_schedule),
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
    model=model,
    dataflow=dataset_train,
    callbacks=callbacks,
    steps_per_epoch=5000,
    max_epoch=110,
    nr_tower=nr_tower
  )


def main(argv):
  if FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  model = Model()
  model_name = ('cdiscount-inceptionv3')

  if FLAGS.pred_train:
    make_pred(model, model_name, 'train', FLAGS.model_path_for_pred,
        PRED_BATCH_SIZE, FLAGS.apply_augmentation, FLAGS.gpu)
  elif FLAGS.pred_test:
    make_pred(model, model_name, 'test', FLAGS.model_path_for_pred,
        PRED_BATCH_SIZE, FLAGS.apply_augmentation, FLAGS.gpu)
  else:
    logger.set_logger_dir(
        os.path.join('train_log', model_name))
    config = get_config(model, model_name)
    if FLAGS.load:
      config.session_init = get_model_loader(FLAGS.load)
    SyncMultiGPUTrainerParameterServer(config).train()
    #SimpleTrainer(config).train()

if __name__ == '__main__':
  app.run()
