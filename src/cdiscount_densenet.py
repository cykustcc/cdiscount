#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cdiscount_densenet.py
# Author: Jiawei Chen <jwchen.maria@gmail.com>
r"""
File to run densenet model on cdiscount data.
reference: https://github.com/YixuanLi/densenet-tensorflow/blob/master/cifar10-densenet.py

Example Usage: (assume you should in ../src)
python -m src.cdiscount_densenet --gpu=0,1,2,3 --resnet_depth=50

python -m src.cdiscount_densenet \
    --resnet_depth=50 \
    --pred_test=True \
    --model_path_for_pred=./train_log/train_log/cdiscount-resnet-d18/model-20000
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

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('densenet_depth', 18,
                      'depth of densenet, should be one of [18, 34, 50, 101].')

gflags.DEFINE_integer('densenet_growth_rate', 12,
                      'growth rate of densenet, should be one of [12, 24, 32, 40].')

gflags.DEFINE_bool('load_all_imgs_to_memory', False,
                   'Load all training images to memory before training.')

gflags.DEFINE_string('datadir', './data/train_imgs',
                     'folder to store jpg imgs')

gflags.DEFINE_string('datadir_test', './data/test_imgs',
                     'folder to store jpg imgs')

gflags.DEFINE_string('img_list_file', './data/train_imgfilelist.txt',
                     'path of the file store (image, label) list of training'
                     'set.')

gflags.DEFINE_string('img_list_file_test', './data/test_imgfilelist.txt',
                     'path of the file store (image, label) list of test set')

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


gflags.DEFINE_string('model_path_for_pred', "",
                     'model path for prediction on test set.')

gflags.DEFINE_string('log_dir_name_suffix', "",
                     'suffix of the model checkpoint folder name.')

# BATCH_SIZE = 64
if socket.gethostname() == "ESC8000":
  BATCH_SIZE = 512
else:
  BATCH_SIZE = 1
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

    # prob = tf.nn.softmax(logits, name='output')

    # cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    # cost = tf.reduce_mean(cost, name='cross_entropy_loss')

    # wrong = prediction_incorrect(logits, label)
    # # monitor training error
    # add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

    # # weight decay on all W
    # wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
    
    # loss = compute_loss_and_error(logits, label)
    # wd_loss = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
    # add_moving_summary(cost, wd_loss)

    # add_param_summary(('.*/W', ['histogram']))   # monitor W
    # self.cost = tf.add_n([cost, wd_loss], name='cost')

  def _get_optimizer(self):
    lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
    tf.summary.scalar('learning_rate', lr)
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, batch):
  isTrain = train_or_test == 'train'
  # ds = dataset.Cifar10(train_or_test)
  ds = Cdiscount(FLAGS.datadir, FLAGS.img_list_file, train_or_test,
                 shuffle=isTrain, large_mem_sys=FLAGS.load_all_imgs_to_memory)
  if isTrain:
      ds = PrefetchDataZMQ(ds, min(20, multiprocessing.cpu_count()))
  ds = BatchData(ds, batch, remainder=not isTrain)
  return ds
  # pp_mean = ds.get_per_pixel_mean()
  # if isTrain:
  #   augmentors = [
  #     imgaug.CenterPaste((40, 40)),
  #     imgaug.RandomCrop((32, 32)),
  #     imgaug.Flip(horiz=True),
  #     #imgaug.Brightness(20),
  #     #imgaug.Contrast((0.6,1.4)),
  #     imgaug.MapImage(lambda x: x - pp_mean),
  #   ]
  # else:
  #   augmentors = [
  #     imgaug.MapImage(lambda x: x - pp_mean)
  #   ]
  # ds = AugmentImageComponent(ds, augmentors)
  # ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
  # if isTrain:
  #   ds = PrefetchData(ds, 3, 2)
  # return ds

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
  # steps_per_epoch = dataset_train.size()
  # dataset_test = get_data('test')
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
    steps_per_epoch=5000,
    max_epoch=110,
    nr_tower=nr_tower
  )

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
#   parser.add_argument('--load', help='load model')
#   parser.add_argument('--drop_1', default=150,
#       help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
#   parser.add_argument('--drop_2', default=225,
#       help='Epoch to drop learning rate to 0.001')
#   parser.add_argument('--depth', default=40,
#       help='The depth of densenet')
#   parser.add_argument('--max_epoch', default=300,
#       help='max epoch')
#   args = parser.parse_args()

#   if args.gpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#   config = get_config()
#   if args.load:
#     config.session_init = SaverRestore(args.load)
#   if args.gpu:
#     config.nr_tower = len(args.gpu.split(','))
#   SyncMultiGPUTrainer(config).train()

def make_pred(model, train_or_test_or_val):
  PRED_BATCH_SIZE = 128
  if train_or_test_or_val == 'test':
    ds0 = Cdiscount(FLAGS.datadir_test, FLAGS.img_list_file_test, 'test',
                   shuffle=False)
  elif train_or_test_or_val == 'train':
    ds0 = Cdiscount(FLAGS.datadir, FLAGS.img_list_file, 'train',
                   shuffle=False)
  ds = BatchData(ds0, PRED_BATCH_SIZE, remainder=True)
  assert FLAGS.model_path_for_pred!="", "no model_path_for_pred specified!"
  pred_config = PredictConfig(
      model=model,
      session_init=get_model_loader(FLAGS.model_path_for_pred),
      input_names=['input'],
      output_names=['output-prob'])
  pred = OfflinePredictor(pred_config)

  pred_folder = './data/pred/'
  if not os.path.exists(pred_folder):
    os.mkdir(pred_folder)
  pred_fname = os.path.join(pred_folder,
      'pred-' + 'cdiscount-densenet-d' + str(FLAGS.densenet_depth) + str(FLAGS.densenet_densenet_growth_rate) + 
      train_or_test_or_val + str(FLAGS.log_dir_name_suffix) + '.txt')
  with open(pred_fname, 'w') as f:
    writer = csv.writer(f)
    with tqdm.tqdm(total=ds.size(), **get_tqdm_kwargs()) as pbar:
      #MAX_ITER = PRED_BATCH_SIZE
      iter_cnt = 0
      for im, _ in ds.get_data():
        output_prob = pred([im])[0]
        for prob in output_prob:
          # top 10 categories' psuedo id (0 to 5269) in descending order
          top_10_pseudo_id = prob.argsort()[-10:][::-1]
          top_10_prob = prob[top_10_pseudo_id]
          fname = ds0.imglist[iter_cnt][0]
          line_content = [fname] + list(top_10_pseudo_id) + list(top_10_prob)
          writer.writerow(line_content)
          iter_cnt += 1
          pbar.update(1)
        #if iter_cnt >= MAX_ITER:
          #break

def main(argv):
  if FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  model = Model(FLAGS.densenet_depth, FLAGS.densenet_growth_rate)
  if FLAGS.pred_train:
    make_pred(model, 'train')
  elif FLAGS.pred_test:
    make_pred(model, 'test')
  else:
    logger.set_logger_dir(
        os.path.join('train_log', 'cdiscount-densenet-d' + str(FLAGS.densenet_depth)
          + str(FLAGS.log_dir_name_suffix)))
    config = get_config(model)
    if FLAGS.load:
      config.session_init = get_model_loader(FLAGS.load)
    SyncMultiGPUTrainerParameterServer(config).train()
    #SimpleTrainer(config).train()


if __name__ == '__main__':
  app.run()

