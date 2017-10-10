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
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU,
    LinearWrap)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor


def image_preprocess(image, bgr=True):
  with tf.name_scope('image_preprocess'):
    if image.dtype.base_dtype != tf.float32:
      image = tf.cast(image, tf.float32)
    image = image * (1.0 / 255)

    # TODO(yukun, replace with mean and std of cdiscount dataset
    #mean = [0.485, 0.456, 0.406]    # rgb
    mean = [0.783, 0.766, 0.757]
    #std = [0.229, 0.224, 0.225]
    std = [ 0.206,  0.209, 0.207]
    if bgr:
      mean = mean[::-1]
      std = std[::-1]
    image_mean = tf.constant(mean, dtype=tf.float32)
    image_std = tf.constant(std, dtype=tf.float32)
    image = (image - image_mean) / image_std
    return image


def compute_loss_and_error(logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
    return loss


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


def resnet_group(l, name, block_func, features, count, stride, first=False):
  with tf.variable_scope(name):
    with tf.variable_scope('block0'):
      l = block_func(l, features, stride,
                     'no_preact' if first else 'both_preact')
    for i in range(1, count):
      with tf.variable_scope('block{}'.format(i)):
        l = block_func(l, features, 1, 'default')
    return l


def resnet_backbone(image, num_blocks, block_func):
  with argscope(Conv2D, nl=tf.identity, use_bias=False,
                W_init=variance_scaling_initializer(mode='FAN_OUT')):
    logits = (LinearWrap(image)
              .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
              .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
              .apply(resnet_group, 'group0', block_func, 64, num_blocks[0], 1, first=True)
              .apply(resnet_group, 'group1', block_func, 128, num_blocks[1], 2)
              .apply(resnet_group, 'group2', block_func, 256, num_blocks[2], 2)
              .apply(resnet_group, 'group3', block_func, 512, num_blocks[3], 2)
              .BNReLU('bnlast')
              .GlobalAvgPooling('gap')
              .FullyConnected('linear', 5270, nl=tf.identity)())
  return logits


