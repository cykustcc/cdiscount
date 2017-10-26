#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_resnet_utils.py
# Author: Yukun Chen <cykustc@gmail.com>
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer


import tensorpack as tp
from tensorpack import imgaug
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
        BatchData)
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU,
    LinearWrap)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor


## Resnet Building Blocks
def resnet_shortcut(l, n_out, stride, nl=tf.identity):
  n_in = l.get_shape().as_list()[3]
  if n_in != n_out:   # change dimension when channel is not the same
    return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
  else:
    return l


def apply_preactivation(l, preact):
  if preact == 'bnrelu':
    shortcut = l    # preserve identity mapping
    l = BNReLU('preact', l)
  else:
    shortcut = l
  return l, shortcut


def resnet_basicblock(l, ch_out, stride, preact):
  l, shortcut = apply_preactivation(l, preact)
  l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
  l = Conv2D('conv2', l, ch_out, 3)
  return l + resnet_shortcut(shortcut, ch_out, stride)


def resnet_bottleneck(l, ch_out, stride, preact):
  l, shortcut = apply_preactivation(l, preact)
  l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
  l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
  l = Conv2D('conv3', l, ch_out * 4, 1)
  return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def se_resnet_bottleneck(l, ch_out, stride):
	shortcut = l
	l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
	l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
	l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

	squeeze = GlobalAvgPooling('gap', l)
	squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
	squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
	l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])
	return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride, first=False):
  with tf.variable_scope(name):
    with tf.variable_scope('block0'):
      l = block_func(l, features, stride,
                     'no_preact' if first else 'both_preact')
    for i in range(1, count):
      with tf.variable_scope('block{}'.format(i)):
        l = block_func(l, features, 1, 'default')
    return l


def resnet_backbone(image, num_blocks, block_func, resnet_width_factor=1):
  rwf = resnet_width_factor
  with argscope(Conv2D, nl=tf.identity, use_bias=False,
                W_init=variance_scaling_initializer(mode='FAN_OUT')):
    logits = (LinearWrap(image)
              .Conv2D('conv0', 64*rwf, 7, stride=2, nl=BNReLU)
              .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
              .apply(resnet_group, 'group0', block_func, 64*rwf, num_blocks[0], 1, first=True)
              .apply(resnet_group, 'group1', block_func, 128*rwf, num_blocks[1], 2)
              .apply(resnet_group, 'group2', block_func, 256*rwf, num_blocks[2], 2)
              .apply(resnet_group, 'group3', block_func, 512*rwf, num_blocks[3], 2)
              .BNReLU('bnlast')
              .GlobalAvgPooling('gap')
              .FullyConnected('linear', 5270, nl=tf.identity)())
  return logits


