#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_inception.py
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
from .inception_utils import *

FLAGS = gflags.FLAGS

gflags.DEFINE_string('mode', 'v3',
                     'should be one of v3, v4, resnetv1, resnetv2.')

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
    'cdiscount-inceptionv3' : 142,
    'cdiscount-inceptionv4' : 142,
    'cdiscount-inceptionresnetv1' : 142,
    'cdiscount-inceptionresnetv2' : 142,
  }
  PRED_BATCH_SIZE = 300
else: #home, 8GB gpu memory.
  BATCH_SIZE={
    'cdiscount-inceptionv3' : 128, #1792
    'cdiscount-inceptionv4' : 128, #1792
    'cdiscount-inceptionresnetv1' : 128,
    'cdiscount-inceptionresnetv2' : 128,
  }
  PRED_BATCH_SIZE = 192
INPUT_SHAPE = 180

LEARNING_RATE={
  'cdiscount-inceptionv3' : [(5, 0.01), (9, 0.08), (12, 0.006),
														 (17, 0.003), (22, 1e-3), (36, 2e-4),
														 (41, 8e-5), (48, 1e-5), (53, 2e-6)],
  'cdiscount-inceptionv4' : [(5, 0.01), (9, 0.08), (12, 0.006),
														 (17, 0.003), (22, 1e-3), (36, 2e-4),
														 (41, 8e-5), (48, 1e-5), (53, 2e-6)],
  'cdiscount-inceptionresnetv1' : [(5, 0.01), (9, 0.08), (12, 0.006),
														 (17, 0.003), (22, 1e-3), (36, 2e-4),
														 (41, 8e-5), (48, 1e-5), (53, 2e-6)],
  'cdiscount-inceptionresnetv2' : [(5, 0.01), (9, 0.08), (12, 0.006),
														 (17, 0.003), (22, 1e-3), (36, 2e-4),
														 (41, 8e-5), (48, 1e-5), (53, 2e-6)],
}


class Model(ModelDesc):
  def __init__(self, mode):
    self.mode = mode

  def _get_inputs(self):
    return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, inputs):
    image, label = inputs
    image = image_preprocess(image, bgr=False)

    INCEPTION_MODES = {
      'v3': inceptionv3,
      'v4': inceptionv4,
      'resnetv1': inceptionResNetv1,
      'resnetv2': inceptionResNetv2,
    }

    inception = INCEPTION_MODES[self.mode]

    if self.mode == 'v3':
      logits, br1 = inception(image)
      loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br1, labels=label)
      loss1 = tf.reduce_mean(loss1, name='loss1')
      loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
      loss2 = tf.reduce_mean(loss2, name='loss2')
    else:
      logits = inception(image)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
      loss = tf.reduce_mean(loss, name='loss')

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

    # weight decay on all W of fc layers
    wd_w = tf.train.exponential_decay(0.00004, get_global_step_var(),
                                      80000, 0.7, True)
    wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='l2_regularize_loss')

    if self.mode == 'v3':
      self.cost = tf.add_n([0.4 * loss1, loss2, wd_cost], name='cost')
      add_moving_summary(loss1, loss2, wd_cost, self.cost)
    else:
      self.cost = tf.add_n([loss, wd_cost], name='cost')
      add_moving_summary(wd_cost, self.cost)

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
        dataset_val, infs, [int(x) for x in
           os.environ['CUDA_VISIBLE_DEVICES'].split(",")]))
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

  model = Model(FLAGS.mode)
  model_name = ('cdiscount-inception{}'.format(FLAGS.mode))

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
