#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_resnet.py
# Author: Yukun Chen <cykustc@gmail.com>
r"""
File to run resnet model on cdiscount data.

Example Usage: (assume you should in ../src)
python -m src.cdiscount_resnet --gpu=0,1,2,3 --resnet_depth=50

python -m src.cdiscount_resnet --gpu=0,1,2,3 --resnet_depth=50 \
    --mode=se-resnet --resnet_width_factor=2

python -m src.cdiscount_resnet --gpu=11,12,13 --resnet_depth=101 \
    --apply_augmentation

python -m src.cdiscount_resnet \
    --gpu=15 \
    --resnet_depth=50 \
    --pred_test=True \
    --model_path_for_pred=./train_log/cdiscount-resnet-d18/model-20000
"""
import sys
import argparse
import numpy as np
import os
import multiprocessing
import socket
import csv

from google.apputils import app
import gflags

import tensorflow as tf
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (TrainConfig, SimpleTrainer,
    SyncMultiGPUTrainerParameterServer)
from tensorpack.predict import OfflinePredictor
from tensorpack.dataflow import imgaug, FakeData, PrefetchDataZMQ, BatchData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from .cdiscount_data import *
from .cdiscount_resnet_utils import *
from .common_utils import *

FLAGS = gflags.FLAGS

gflags.DEFINE_string('mode', None,
                     'should be one of resnet or se-resnet.')

gflags.DEFINE_integer('resnet_depth', 18,
                      'depth of resnet, should be one of [18, 34, 50, 101, 152].')

gflags.DEFINE_integer('resnet_width_factor', 1,
                      'width factor of resnet, should be one of [1,2,3,4].'
                      'See https://arxiv.org/abs/1605.07146')

gflags.DEFINE_bool('load_all_imgs_to_memory', False,
                   'Load all training images to memory before training.')

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
  #TOTAL_BATCH_SIZE = 1792
  TOTAL_BATCH_SIZE = 1344
  PRED_BATCH_SIZE = 256
else:
  TOTAL_BATCH_SIZE = 192
  PRED_BATCH_SIZE = 192
INPUT_SHAPE = 180

RESNET_CONFIG = {
  18: ([2, 2, 2, 2], resnet_basicblock),
  34: ([3, 4, 6, 3], resnet_basicblock),
  50: ([3, 4, 6, 3], resnet_bottleneck),
  101: ([3, 4, 23, 3], resnet_bottleneck),
  152: ([3, 8, 36, 3], resnet_bottleneck)
}

SE_RESNET_CONFIG = {
  18: ([2, 2, 2, 2], resnet_basicblock),
  34: ([3, 4, 6, 3], resnet_basicblock),
  50: ([3, 4, 6, 3], se_resnet_bottleneck),
  101: ([3, 4, 23, 3], se_resnet_bottleneck),
  152: ([3, 8, 36, 3], se_resnet_bottleneck)
}


class Model(ModelDesc):
  def __init__(self, depth, width=1, mode='resnet'):
    self.depth = depth
    self.width = width
    self.mode = mode

  def _get_inputs(self):
    return [InputDesc(tf.float16, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, inputs):
    image, label = inputs
    image = image_preprocess(image, bgr=False)

    if self.mode == 'resnet':
      n_of_blocks, block_func = RESNET_CONFIG[self.depth]
    elif self.mode == 'se-resnet':
      n_of_blocks, block_func = SE_RESNET_CONFIG[self.depth]
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm]):
      logits = resnet_backbone(image, n_of_blocks, block_func, self.width)

    loss = compute_loss_and_error(logits, label)
    wd_loss = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
    add_moving_summary(loss, wd_loss)
    self.cost = tf.add_n([loss, wd_loss], name='cost')

  def _get_optimizer(self):
    lr = get_scalar_var('learning_rate', 0.1, summary=True)
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, batch):
  """
  Args:
    train_or_test: should be one of {'train', 'test', 'val'}
  """
  isTrain = train_or_test == 'train'

  ds = Cdiscount(FLAGS.datadir, FLAGS.img_list_file, train_or_test,
                 shuffle=isTrain, large_mem_sys=FLAGS.load_all_imgs_to_memory)
  if FLAGS.apply_augmentation:
    logger.info("Applying image augmentation.")
    augmentors = fbresnet_augmentor(isTrain)
    ds = AugmentImageComponent(ds, augmentors, copy=False)

  if isTrain:
      ds = PrefetchDataZMQ(ds, min(20, multiprocessing.cpu_count()))
  ds = BatchData(ds, batch, remainder=not isTrain)
  return ds


def get_config(model):
  nr_tower = max(get_nr_gpu(), 1)
  batch = TOTAL_BATCH_SIZE // nr_tower
  logger.info("Running on {} towers. Batch size per tower:{}".format(nr_tower,
                                                                     batch))

  dataset_train = get_data('train', batch)
  dataset_val = get_data('val', batch)
  infs = [ClassificationError('wrong-top1', 'val-error-top1'),
          ClassificationError('wrong-top5', 'val-error-top5')]
  callbacks=[
    ModelSaver(),
    ScheduledHyperParamSetter('learning_rate',
                              [(15, 1e-2), (30, 1e-3), (65, 1e-4), (85, 1e-5), (105, 1e-6)]),
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

  model = Model(FLAGS.resnet_depth, FLAGS.resnet_width_factor, FLAGS.mode)
  width_str = ('-wf' + str(FLAGS.resnet_width_factor) if
      FLAGS.resnet_width_factor == 0 else '')
  model_name = ('cdiscount-{}-d'.format(FLAGS.mode) + str(FLAGS.resnet_depth)
      + width_str + str(FLAGS.log_dir_name_suffix))

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

