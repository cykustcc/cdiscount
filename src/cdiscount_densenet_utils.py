#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_densenet_utils.py
# Author: Jiawei Chen <jzc245@ist.psu.edu>
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
    Conv2D, AvgPooling, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU,
    LinearWrap)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor


def composite_function(_input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        """
        # BN
        output = BatchNorm('bn_compos', _input)
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = Conv2D('conv_compos', output, out_features, kernel_size)
        return output

def densenet_bottleneck(_input, out_features):
        output = BatchNorm('bn_bottleneck', _input)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = Conv2D('conv_bottleneck', output, inter_features, 1)
        return output

def add_layer(_input, growth_rate, bc_mode):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not bc_mode:
            comp_out = composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif bc_mode:
            bottleneck_out = densenet_bottleneck(_input, out_features=growth_rate)
            comp_out = composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        output = tf.concat(axis=3, values=(_input, comp_out))
        return output

def add_transition(_input, name, bc_mode, theta):
  shape = _input.get_shape().as_list()
  out_features = shape[3]
  with tf.variable_scope(name) as scope:
    if bc_mode:
        out_features = int(out_features * theta)
        print(out_features, theta)
    output = composite_function(_input, out_features=out_features, kernel_size=1)
    # run average pooling
    output = AvgPooling('pool', output, 2)
  return output

def densenet_block(_input, name, growth_rate, bc_mode, count):
  with tf.variable_scope(name):
    for i in range(count):
      with tf.variable_scope('block{}'.format(i)):
        output = add_layer(_input, growth_rate, bc_mode)
    return output

def densenet_backbone(image, num_blocks, growth_rate, bc_mode, theta):
      with argscope(Conv2D, nl=tf.identity, use_bias=False,
                    W_init=variance_scaling_initializer(mode='FAN_OUT')):
        print('backbone:', bc_mode, theta)
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(densenet_block, 'dense_group0', growth_rate, bc_mode, num_blocks[0])
                  .apply(add_transition, 'trans_group0', bc_mode, theta)
                  .apply(densenet_block, 'dense_group1', growth_rate, bc_mode, num_blocks[1])
                  .apply(add_transition, 'trans_group1', bc_mode, theta)
                  .apply(densenet_block, 'dense_group2', growth_rate, bc_mode, num_blocks[2])
                  .apply(add_transition, 'trans_group2', bc_mode, theta)
                  .apply(densenet_block, 'dense_group3', growth_rate, bc_mode, num_blocks[3])
                  .apply(add_transition, 'trans_group3', bc_mode, theta)
                  .BNReLU('bnlast')
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 5270, nl=tf.identity)())
      return logits