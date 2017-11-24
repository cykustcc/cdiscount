#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cdiscount_densenet.py
# Author: Jiawei Chen <jwchen.maria@gmail.com>
r"""
File to run densenet model on cdiscount data.
references: https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py
           https://github.com/yeephycho/densenet-tensorflow/blob/master/net/densenet.py

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
from .cdiscount_densenet_utils import *
from .common_utils import *

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('densenet_depth', 121,
                      'depth of densenet, should be one of [121, 169, 201, 264].')

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

gflags.DEFINE_bool('BC_mode', True,
                   'If true, run Densenet-BC with bottleneck and compression transition layers')

gflags.DEFINE_float('compression_rate', 0.5,
                      'compression_rate in transition layer: 0<theta<=1')

gflags.DEFINE_string('model_path_for_pred', "",
                     'model path for prediction on test set.')

gflags.DEFINE_string('log_dir_name_suffix', "",
                     'suffix of the model checkpoint folder name.')

# BATCH_SIZE = 64
if socket.gethostname() == "ESC8000":
  BATCH_SIZE={
    'cdiscount-densenet-d121-gr12-BCTrue-theta0.5' : 384, #1792
  }
  PRED_BATCH_SIZE=300
else:
  BATCH_SIZE={
    'cdiscount-densenet-d121-gr12-BCTrue-theta0.5' : 128, #1792
  }
  PRED_BATCH_SIZE=300

INPUT_SHAPE = 180

DENSENET_CONFIG = {
  121: [6, 12, 24, 16],
  169: [6, 12, 32, 32],
  201: [6, 12, 48, 32],
  264: [6, 12, 64, 48]
}

LEARNING_RATE={
  'cdiscount-densenet-d121-gr12-BCTrue-theta0.5' : [(30, 1e-2), (60, 1e-3), (85, 1e-4),
      (95, 1e-5), (105, 1e-6)]
}

class Model(ModelDesc):
  def __init__(self, depth, growth_rate, BC_mode, compression_rate):
    self.depth = depth
    self.growth_rate = growth_rate
    self.BC_mode = BC_mode
    self.compression_rate = compression_rate
    print(self.depth, self.growth_rate, self.BC_mode, self.compression_rate)

  def _get_inputs(self):
    return [InputDesc(tf.float16, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')]

  def _build_graph(self, input_vars):
    image, label = input_vars
    image = image_preprocess(image, bgr=False)

    n_of_blocks = DENSENET_CONFIG[self.depth]
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm]):
      logits = densenet_backbone(image, n_of_blocks, self.growth_rate, self.BC_mode, self.compression_rate)

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


def get_config(model, model_name):
  nr_tower = max(get_nr_gpu(), 1)
  batch = BATCH_SIZE[model_name]
  logger.info("Running on {} towers. Batch size per tower:{}".format(nr_tower,
                                                                     batch))
  # prepare dataset
  dataset_train = get_data('train', batch)
  dataset_val = get_data('val', batch)
  infs = [ClassificationError('wrong-top1', 'val-error-top1'),
          ClassificationError('wrong-top5', 'val-error-top5')]
  steps_per_epoch = dataset_train.size() // 3
  if model_name in LEARNING_RATE.keys():
    learning_rate_schedule = LEARNING_RATE[model_name]
  else:
    learning_rate_schedule = [(30, 1e-2), (60, 1e-3), (85, 1e-4),
       (95, 1e-5), (105, 1e-6)]
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

  model = Model(FLAGS.densenet_depth, FLAGS.densenet_growth_rate, FLAGS.BC_mode, FLAGS.compression_rate)
  model_name = ('cdiscount-densenet-d' + str(FLAGS.densenet_depth) + '-gr' +
        str(FLAGS.densenet_growth_rate) + '-BC' +
        str(FLAGS.BC_mode) + '-theta' +
        str(FLAGS.compression_rate) +
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
    config = get_config(model, model_name)
    if FLAGS.load:
      config.session_init = get_model_loader(FLAGS.load)
    SyncMultiGPUTrainerParameterServer(config).train()
    #SimpleTrainer(config).train()


if __name__ == '__main__':
  app.run()

