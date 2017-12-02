#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_utils.py
# Author: Yukun Chen <cykustc@gmail.com>
import tensorflow as tf

from tensorpack import *
from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d

from .cdiscount_resnet_utils import resnet_shortcut, apply_preactivation

def stem(image):
  l = (LinearWrap(image)
       .Conv2D('conv0', 32, 3, stride=2, padding='VALID')  # 299
       .Conv2D('conv1', 32, 3, padding='VALID')  # 149
       .Conv2D('conv2', 64, 3, padding='SAME')())  # 147

  l = tf.concat([MaxPooling('maxpool', l, 3, 2),
                 Conv2D('conv3', l, 96, 3, stride=2, padding='VALID')],
                3, name='concat') # 147

  left_conv = (LinearWrap(l)
               .Conv2D('conv4', 64, 1, padding='SAME')
               .Conv2D('conv5', 96, 3, padding='VALID')()) # 73

  right_conv = (LinearWrap(l)
                .Conv2D('conv6', 64, 1)
                .Conv2D('conv7', 64, [7, 1])
                .Conv2D('conv8', 64, [1, 7])
                .Conv2D('conv9', 96, 3, padding='VALID')()) # 73

  l = tf.concat([left_conv, right_conv], 3, name='concat2')

  return tf.concat([Conv2D('conv10', l, 192, 3, stride=2, padding='VALID'),
                    MaxPooling('maxpool2', l, 3, stride=2)], 3, name='concat3') # 71

def stemResNetv1(image):
  l = (LinearWrap(image)
       .Conv2D('conv0', 32, 3, stride=2, padding='VALID')  # 299
       .Conv2D('conv1', 32, 3, padding='VALID')  # 149
       .Conv2D('conv2', 64, 3, padding='SAME')
       .MaxPooling('maxpool', 3, 2)
       .Conv2D('conv3', 80, 1, padding='SAME')
       .Conv2D('conv4', 192, 3, padding='VALID')
       .Conv2D('conv5', 256, 3, stride=2, padding='VALID')())  # 147
  return l

def proj_kk(l, k, ch_r, ch, stride=1, scope_name=""):
  with tf.variable_scope(scope_name):
    l = Conv2D('conv{0}{0}r'.format(k), l, ch_r, 1)
    l = Conv2D('conv{0}{0}'.format(k), l, ch, k, stride=stride,
                  padding='VALID' if stride > 1 else 'SAME')
  return l

def proj_33(l, ch0, ch1, ch2):
  return (LinearWrap(l)
          .Conv2D('conv33r', ch0, 1)
          .Conv2D('conv33a', ch1, [1, 3])
          .Conv2D('conv33b', ch2, [3, 1])())

def proj_233(l, ch0, ch1, ch2, stride=1):
  l = Conv2D('conv233r', l, ch0, 1)
  l = Conv2D('conv233a', l, ch1, 3)
  return Conv2D('conv233b', l, ch2, 3, stride=stride,
                padding='VALID' if stride > 1 else 'SAME')

def pool_proj(l, ch, pool_type):
  if pool_type == 'max':
      l = MaxPooling('maxpool', l, 3, 1)
  else:
      l = AvgPooling('maxpool', l, 3, 1, padding='SAME')
  return Conv2D('poolproj', l, ch, 1)

def proj_77(l, ch0, ch1, ch2):
  return (LinearWrap(l)
          .Conv2D('conv77r', ch0, 1)
          .Conv2D('conv77a', ch1, [1, 7])
          .Conv2D('conv77b', ch2, [7, 1])())

def proj_277(l, ch0, ch1, ch2):
  return (LinearWrap(l)
          .Conv2D('conv277r', ch0, 1)
          .Conv2D('conv277aa', ch0, [7, 1])
          .Conv2D('conv277ab', ch1, [1, 7])
          .Conv2D('conv277ba', ch1, [7, 1])
          .Conv2D('conv277bb', ch2, [1, 7])())

def inception_a(l, scope_name='inceptionA'):
  with tf.variable_scope(scope_name):
    l = tf.concat([
        Conv2D('conva0', l, 96, 1),
        proj_kk(l, 5, 64, 96),
        proj_233(l, 64, 96, 96),
        pool_proj(l, 96, 'avg')
    ], 3, name='concat')
  return l

def inception_b(l, scope_name='inceptionB'):
  with tf.variable_scope(scope_name):
    l = tf.concat([
        Conv2D('convb0', l, 384, 1),
        proj_77(l, 192, 224, 256),
        proj_277(l, 192, 224, 256),
        pool_proj(l, 128, 'avg')
    ], 3, name='concat')
  return l

def inception_c(l, scope_name='inceptionC'):
  with tf.variable_scope(scope_name):
    br11 = Conv2D('conv11', l, 256, 1)
    br33 = Conv2D('conv133r', l, 384, 1)
    br33 = tf.concat([
        Conv2D('conv133a', br33, 256, [1, 3]),
        Conv2D('conv133b', br33, 256, [3, 1])
    ], 3, name='conv133')

    br233 = proj_33(l, 384, 448, 512)
    br233 = tf.concat([
        Conv2D('conv233a', br233, 256, [1, 3]),
        Conv2D('conv233b', br233, 256, [3, 1]),
    ], 3, name='conv233')

    l = tf.concat([
        br11, br33, br233,
        pool_proj(l, 256, 'avg')
    ], 3, name='concat')
  return l

def reduction_a(l, kk, ll, mm, nn, scope_name='reductionA'):
  with tf.variable_scope(scope_name):
    branch0 = proj_kk(l, 3, kk, ll)
    branch0 = Conv2D('convra1', branch0, mm, 3, stride=2, padding='VALID')
    return tf.concat([
        MaxPooling('poolra', l, 3, 2),
        Conv2D('convra0', l, nn, 3, stride=2, padding='VALID'),
        branch0
        ], 3, name='concat')

def reduction_b(l, scope_name='reductionB'):
  with tf.variable_scope(scope_name):
    branch0 = Conv2D('convrb0', l, 192, 1)
    branch0 = Conv2D('convra1', l, 192, 3, stride=2, padding='VALID')
    branch1 = proj_77(l, 256, 256, 320)
    branch1 = Conv2D('convra2', l, 320, 3, stride=2, padding='VALID')
    return tf.concat([
        MaxPooling('poolrb', l, 3, 2),
        branch0,
        branch1
        ], 3, name='concat')


def inceptionv3(image):
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
          proj_233(l, 64, 96, 96),
          pool_proj(l, 32, 'avg')
      ], 3, name='concat')
    with tf.variable_scope('incep-35-288a'):
      l = tf.concat([
          Conv2D('conv11', l, 64, 1),
          proj_kk(l, 5, 48, 64),
          proj_233(l, 64, 96, 96),
          pool_proj(l, 64, 'avg')
      ], 3, name='concat')
    with tf.variable_scope('incep-35-288b'):
      l = tf.concat([
          Conv2D('conv11', l, 64, 1),
          proj_kk(l, 5, 48, 64),
          proj_233(l, 64, 96, 96),
          pool_proj(l, 64, 'avg')
      ], 3, name='concat')
    # 35x35x288
    with tf.variable_scope('incep-17-768a'):
      l = tf.concat([
          Conv2D('conv3x3', l, 384, 3, stride=2, padding='VALID'),
          proj_233(l, 64, 96, 96, stride=2),
          MaxPooling('maxpool', l, 3, 2)
      ], 3, name='concat')
    with tf.variable_scope('incep-17-768b'):
      l = tf.concat([
          Conv2D('conv11', l, 192, 1),
          proj_77(l, 128, 128, 192),
          proj_277(l, 128, 128, 192),
          pool_proj(l, 192, 'avg')
      ], 3, name='concat')
    for x in ['c', 'd']:
      with tf.variable_scope('incep-17-768{}'.format(x)):
        l = tf.concat([
            Conv2D('conv11', l, 192, 1),
            proj_77(l, 160, 160, 192),
            proj_277(l, 160, 160, 192),
            pool_proj(l, 192, 'avg')
        ], 3, name='concat')
    with tf.variable_scope('incep-17-768e'):
      l = tf.concat([
          Conv2D('conv11', l, 192, 1),
          proj_77(l, 192, 192, 192),
          proj_277(l, 192, 192, 192),
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
          Conv2D('conv73', proj_77(l, 192, 192, 192), 192, 3, stride=2, padding='VALID'),
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
  return logits, br1

def inceptionv4(image):
  l = stem(image)
  for i in xrange(4):
    l = inception_a(l, "inceptionA{}".format(i))
  l = reduction_a(l, 192, 224, 256, 384, "reductionA")
  for i in xrange(7):
    l = inception_b(l, "inceptionB{}".format(i))
  l = reduction_b(l, "reductionB")
  for i in xrange(3):
    l = inception_c(l, "inceptionC{}".format(i))
  l = GlobalAvgPooling('gap', l)
  l = Dropout('drop', l, 0.8)
  logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)
  return logits

def inceptionResNetAv1(l, scope_name="inceptionResNetAv1"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 32, 1),
        proj_kk(l, 3, 32, 32),
        proj_233(l, 32, 32, 32)], 3, name='concat')
    l = Conv2D('conv1', l,  256, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def inceptionResNetBv1(l, scope_name="inceptionResNetBv1"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 128, 1),
        proj_77(l, 128, 128, 128)], 3, name='concat')
    l = Conv2D('conv1', l, 896, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def inceptionResNetCv1(l, scope_name="inceptionResNetCv1"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 192, 1),
        proj_33(l, 192, 192, 192)], 3, name='concat')
    l = Conv2D('conv1', l, 1792, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def reductionResNetBv1(l, scope_name="reductionResNetBv1"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        MaxPooling('maxpool', l, 3, 2),
        proj_kk(l, 3, 256, 384, stride=2, scope_name='l'),
        proj_kk(l, 3, 256, 256, stride=2, scope_name='r'),
        proj_233(l, 256, 256, 256, stride=2)], 3, name='concat')
  return l

def inceptionResNetAv2(l, scope_name="inceptionResNetAv2"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 32, 1),
        proj_kk(l, 3, 32, 32),
        proj_233(l, 32, 48, 64)], 3, name='concat')
    l = Conv2D('conv1', l, 384, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def inceptionResNetBv2(l, scope_name="inceptionResNetBv2"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 192, 1),
        proj_77(l, 128, 160, 192)], 3, name='concat')
    l = Conv2D('conv1', l, 1152, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def inceptionResNetCv2(l, scope_name="inceptionResNetCv2"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        Conv2D('convbranch0', l, 192, 1),
        proj_33(l, 192, 224, 256)], 3, name='concat')
    l = Conv2D('conv1', l, 2144, 1, use_bias=True, nl=tf.identity)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def reductionResNetBv2(l, scope_name="reductionResNetBv2"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.concat([
        MaxPooling('maxpool', l, 3, 2),
        proj_kk(l, 3, 256, 384, stride=2, scope_name='l'),
        proj_kk(l, 3, 256, 288, stride=2, scope_name='r'),
        proj_233(l, 256, 288, 320, stride=2)], 3, name='concat')
  return l

def inceptionResNetv1(image):
  with argscope(Conv2D, nl=BNReLU, use_bias=False),\
      argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
    l = stemResNetv1(image)
    for i in xrange(5):
      l = inceptionResNetAv1(l, 'inceptionResNetAv1{}'.format(i))
    l = reduction_a(l, 192, 192, 256, 384, 'reductionA')
    for i in xrange(10):
      l = inceptionResNetBv1(l, 'inceptionResNetBv1{}'.format(i))
    l = reductionResNetBv1(l, 'reductionResNetBv1')
    for i in range(5):
      l = inceptionResNetCv1(l, 'inceptionResNetCv1{}'.format(i))
    l = GlobalAvgPooling('gap', l)
    l = Dropout('drop', l, 0.8)
    logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)
  return logits

def inceptionResNetv2(image):
  with argscope(Conv2D, nl=BNReLU, use_bias=False),\
      argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
    l = stem(image)
    for i in xrange(4):
      l = inceptionResNetAv2(l, 'inceptionResNetAv2{}'.format(i))
    l = reduction_a(l, 256, 256, 384, 384, 'reductionA')
    for i in xrange(7):
      l = inceptionResNetBv2(l, 'inceptionResNetBv2{}'.format(i))
    l = reductionResNetBv2(l, 'reductionResNetBv2')
    for i in range(3):
      l = inceptionResNetCv2(l, 'inceptionResNetCv2{}'.format(i))
    l = GlobalAvgPooling('gap', l)
    l = Dropout('drop', l, 0.8)
    logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)
  return logits

@layer_register(log_shape=True)
def SeparableConv2D(x, out_channel, kernel_shape,
                    padding='SAME', stride=(1, 1), depth_multiplier=1,
                    dilation_rate=(1, 1), W_init=None, b_init=None,
                    nl=tf.identity, use_bias=True,
                    data_format='NHWC'):
  """
  2D convolution on 4D inputs.

  Args:
      x (tf.Tensor): a 4D tensor.
          Must have known number of channels, but can have other unknown dimensions.
      out_channel (int): number of output channel.
      kernel_shape: (h, w) tuple or a int.
      stride: (h, w) tuple or a int.
      padding (str): 'valid' or 'same'. Case insensitive.
      W_init: initializer for W. Defaults to `variance_scaling_initializer`.
      b_init: initializer for b. Defaults to zero.
      nl: a nonlinearity function.
      use_bias (bool): whether to use bias.

  Returns:
      tf.Tensor named ``output`` with attribute `variables`.

  Variable Names:

  * ``W``: weights for depthwise convulution
  * ``Wp``: weights for pointwise 1x1 convulution
  * ``b``: bias
  """
  in_shape = x.get_shape().as_list()
  channel_axis = 3 if data_format == 'NHWC' else 1
  in_channel = in_shape[channel_axis]
  assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

  kernel_shape = shape2d(kernel_shape)
  padding = padding.upper()
  filter_shape = kernel_shape + [in_channel, depth_multiplier]
  filter_shape_p = shape2d(1) + [depth_multiplier * in_channel, out_channel]
  stride = shape4d(stride, data_format=data_format)

  if W_init is None:
    W_init = tf.contrib.layers.variance_scaling_initializer()
  if b_init is None:
    b_init = tf.constant_initializer()

  W = tf.get_variable('W', filter_shape, initializer=W_init)
  Wp = tf.get_variable('Wp', filter_shape_p, initializer=W_init)
  if use_bias:
    b = tf.get_variable('b', [out_channel], initializer=b_init)

  conv = tf.nn.separable_conv2d(x, W, Wp, stride, padding, rate=dilation_rate,
                                data_format=data_format)

  ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
  ret.variables = VariableHolder(W=W, Wp=Wp)
  if use_bias:
      ret.variables.b = b
  return ret

def xceptionResNetA(l, ch0, ch1, scope_name="xceptionResNetA", begin_w_relu=True):
  with tf.variable_scope(scope_name):
    shortcut = l
    if begin_w_relu:
      l = tf.nn.relu(l, name='relu0')
    shortcut = Conv2D('conv0', shortcut, ch1, 1, use_bias=True, nl=tf.identity)
    l = SeparableConv2D('sepconv0', l, ch0, 3)
    l = tf.nn.relu(l, name='relu1')
    l = SeparableConv2D('sepconv1', l, ch1, 3)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def xceptionResNetB(l, ch, scope_name="xceptionResNetB"):
  with tf.variable_scope(scope_name):
    shortcut = l
    l = tf.nn.relu(l, name='relu0')
    l = SeparableConv2D('sepconv0', l, ch, 3)
    l = tf.nn.relu(l, name='relu1')
    l = SeparableConv2D('sepconv1', l, ch, 3)
    l = tf.nn.relu(l, name='relu2')
    l = SeparableConv2D('sepconv2', l, ch, 3)
    l = shortcut + l
    l = BNReLU('batch', l)
  return l

def entry_flow(image):
  with tf.variable_scope("entry_flow"):
    l = (LinearWrap(image)
         .Conv2D('conv0', 32, 3, stride=2, padding='VALID')  # 299
         .Conv2D('conv1', 64, 3, padding='VALID')())  # 149
    l = xceptionResNetA(l, 128, 128, "xceptionResNetA0", begin_w_relu=False)
    l = xceptionResNetA(l, 256, 256, "xceptionResNetA1", begin_w_relu=True)
    l = xceptionResNetA(l, 728, 728, "xceptionResNetA2", begin_w_relu=True)
  return l

def middle_flow(l):
  with tf.variable_scope("middle_flow"):
    for i in xrange(8):
      l = xceptionResNetB(l, 728, "xceptionResNetB{}".format(i))
  return l

def exit_flow(l):
  with tf.variable_scope("exit_flow"):
    l = xceptionResNetA(l, 728, 1024, "xceptionResNetA0")
    l = SeparableConv2D("sepconv0", l, 1536, 3)
    l = tf.nn.relu(l, name='relu0')
    l = SeparableConv2D("sepconv1", l, 2048, 3)
    l = tf.nn.relu(l, name='relu1')
    l = GlobalAvgPooling('gap', l)
    l = FullyConnected('fc0', l, 4096, nl=tf.identity)
    l = FullyConnected('fc1', l, 4096, nl=tf.identity)
  return l

def xception(image):
  with argscope(Conv2D, nl=BNReLU, use_bias=False),\
      argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
    l = entry_flow(image)
    l = middle_flow(l)
    l = exit_flow(l)
    l = Dropout('drop', l, 0.8)
    logits = FullyConnected('linear', l, out_dim=5270, nl=tf.identity)
  return l

