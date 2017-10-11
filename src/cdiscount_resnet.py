#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_resnet.py
# Author: Yukun Chen <cykustc@gmail.com>
import sys
import argparse
import numpy as np
import os
import multiprocessing
import socket

from google.apputils import app
import gflags

import tensorflow as tf
from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (TrainConfig, SimpleTrainer,
    SyncMultiGPUTrainerParameterServer)
from tensorpack.dataflow import (imgaug, FakeData, PrefetchDataZMQ, BatchData,
    ThreadedMapData)
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from .cdiscount_data import *
from .cdiscount_resnet_utils import *

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('resnet_depth', 18,
                      'depth of resnet, should be one of [18, 34, 50, 101].')

gflags.DEFINE_bool('load_all_imgs_to_memory', False,
                   'Load all training images to memory before training.')

gflags.DEFINE_string('datadir', './data/train_imgs',
                     'folder to store jpg imgs')

gflags.DEFINE_string('filepath_label_file', './data/train_imgfilelist.txt',
                     'folder to store jpg imgs')

gflags.DEFINE_string('img_list_file', './data/train_imgfilelist.txt',
                     'path of the file store (image, label) list.')

gflags.DEFINE_string('load', None,
                     'load model.')

gflags.DEFINE_string('gpu', None,
                     'specify which gpu(s) to be used.')

gflags.DEFINE_string('log_dir_name_suffix', "",
                     'suffix of the model checkpoint folder name.')


if socket.gethostname() == "localhost.localdomain":
  TOTAL_BATCH_SIZE = 512
else:
  TOTAL_BATCH_SIZE = 128
INPUT_SHAPE = 180

RESNET_CONFIG = {
  18: ([2, 2, 2, 2], resnet_basicblock),
  34: ([3, 4, 6, 3], resnet_basicblock),
  50: ([3, 4, 6, 3], resnet_bottleneck),
  101: ([3, 4, 23, 3], resnet_bottleneck)
}


class Model(ModelDesc):
  def __init__(self, depth):
    self.depth = depth

  def _get_inputs(self):
    return [InputDesc(tf.float16, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, inputs):
    image, label = inputs
    image = image_preprocess(image, bgr=False)

    n_of_blocks, block_func = RESNET_CONFIG[self.depth]
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm]):
      logits = resnet_backbone(image, n_of_blocks, block_func)

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

  ds = Cdiscount(FLAGS.datadir, FLAGS.filepath_label_file, train_or_test,
                 shuffle=isTrain, large_mem_sys=FLAGS.load_all_imgs_to_memory)
  #augmentors = fbresnet_augmentor(isTrain)
  #augmentors.append(imgaug.ToUint8())

  #ds = AugmentImageComponent(ds, augmentors, copy=False)
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

  model = Model(FLAGS.resnet_depth)
  logger.set_logger_dir(
      os.path.join('train_log', 'cdiscount-resnet-d' + str(FLAGS.resnet_depth)
        + str(FLAGS.log_dir_name_suffix)))
  config = get_config(model)
  if FLAGS.load:
    config.session_init = get_model_loader(FLAGS.load)
  SyncMultiGPUTrainerParameterServer(config).train()
  #SimpleTrainer(config).train()


if __name__ == '__main__':
  app.run()

