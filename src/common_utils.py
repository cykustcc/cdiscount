import cv2
import csv
import os
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod
import tqdm
import datetime as dt

from google.apputils import app
import gflags

from tensorpack import imgaug, dataset, ModelDesc, InputDesc, logger
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ, BatchData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor, OfflinePredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.utils import get_tqdm_kwargs

from .cdiscount_data import *

FLAGS = gflags.FLAGS

gflags.DEFINE_string('datadir', './data/train_imgs',
                     'folder to store jpg imgs')

gflags.DEFINE_string('datadir_test', './data/test_imgs',
                     'folder to store jpg imgs')

gflags.DEFINE_string('img_list_file', './data/train_imgfilelist.txt',
                     'path of the file store (image, label) list of training'
                     'set.')

gflags.DEFINE_string('img_list_file_test', './data/test_imgfilelist.txt',
                     'path of the file store (image, label) list of test set')


class GoogleNetResize(imgaug.ImageAugmentor):
  """
  crop 36%~100% of the original image
  See `Going Deeper with Convolutions` by Google.
  """
  def __init__(self, crop_area_fraction=0.64,
               aspect_ratio_low=0.75, aspect_ratio_high=1.333):
    self._init(locals())

  def _augment(self, img, _):
    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):
        targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
        aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
        ww = int(np.sqrt(targetArea * aspectR) + 0.5)
        hh = int(np.sqrt(targetArea / aspectR) + 0.5)
        if self.rng.uniform() < 0.5:
            ww, hh = hh, ww
        if hh <= h and ww <= w:
            x1 = 0 if w == ww else self.rng.randint(0, w - ww)
            y1 = 0 if h == hh else self.rng.randint(0, h - hh)
            out = img[y1:y1 + hh, x1:x1 + ww]
            out = cv2.resize(out, (180, 180), interpolation=cv2.INTER_CUBIC)
            return out
    out = imgaug.ResizeShortestEdge(180, interp=cv2.INTER_CUBIC).augment(img)
    out = imgaug.CenterCrop(180).augment(out)
    return out


def fbresnet_augmentor(isTrain):
  """
  Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
  """
  if isTrain:
    augmentors = [
      GoogleNetResize(),
      imgaug.RandomOrderAug(
        [imgaug.BrightnessScale((0.6, 1.4), clip=False),
         imgaug.Contrast((0.6, 1.4), clip=False),
         imgaug.Saturation(0.4, rgb=False),
         # rgb-bgr conversion for the constants copied from fb.resnet.torch
         imgaug.Lighting(0.1,
                         eigval=np.asarray(
                             [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                         eigvec=np.array(
                             [[-0.5675, 0.7192, 0.4009],
                              [-0.5808, -0.0045, -0.8140],
                              [-0.5836, -0.6948, 0.4203]],
                             dtype='float32')[::-1, ::-1]
                         )]),
      imgaug.Flip(horiz=True),
    ]
  else:
    augmentors = []
  return augmentors

def test_augmentor():
  return [GoogleNetResize(),
      imgaug.Flip(horiz=True),
      ]

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
    prob = tf.nn.softmax(logits, name='output-prob')

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


def make_pred(model, model_name, train_or_test_or_val, model_path_for_pred,
    pred_batch_size, apply_aug=False):
  """Make per image prediction (top 10 categories and their probabilities) based
    on a trained nueral net model.
  """
  if train_or_test_or_val == 'test':
    ds0 = Cdiscount(FLAGS.datadir_test, FLAGS.img_list_file_test, 'test',
                   shuffle=False)
  elif train_or_test_or_val == 'train':
    ds0 = Cdiscount(FLAGS.datadir, FLAGS.img_list_file, 'train',
                   shuffle=False)
  ds = ds0
  if apply_aug:
    logger.info("Use augmented img for test prediction:")
    augmentors = test_augmentor()
    ds = AugmentImageComponent(ds, augmentors, copy=False)
  ds = BatchData(ds, pred_batch_size, remainder=True)
  assert model_path_for_pred!="", "no model_path_for_pred specified!"
  pred_config = PredictConfig(
      model=model,
      session_init=get_model_loader(model_path_for_pred),
      input_names=['input'],
      output_names=['output-prob'])
  pred = OfflinePredictor(pred_config)

  pred_folder = './data/pred/'
  if not os.path.exists(pred_folder):
    os.mkdir(pred_folder)
  steps = model_path_for_pred.strip().split('-')
  steps = steps[len(steps) - 1]
  pred_fname = os.path.join(pred_folder,
        'pred-' + model_name + '-step' +
        steps + train_or_test_or_val + str(FLAGS.log_dir_name_suffix) + '.txt')
  if apply_aug and os.path.exists(pred_fname):
    pred_fname = pred_fname.replace(".txt", "-{}.txt".format(
        dt.datetime.now().strftime('%Y%m%d%H%M%S')))

  with open(pred_fname, 'w') as f:
    writer = csv.writer(f)
    logger.info("make prediction for {} dataset, stored to {}:".format(
        train_or_test_or_val, pred_fname))
    with tqdm.tqdm(total=ds0.size(), **get_tqdm_kwargs()) as pbar:
      #MAX_ITER = pred_batch_size
      iter_cnt = 0
      for im, _ in ds.get_data():
        output_prob = pred([im])[0]
        batched_line_content = []
        for prob in output_prob:
          # top 10 categories' psuedo id (0 to 5269) in descending order
          top_10_pseudo_id = prob.argsort()[-10:][::-1]
          top_10_prob = prob[top_10_pseudo_id]
          fname = ds0.imglist[iter_cnt][0]
          line_content = [fname] + list(top_10_pseudo_id) + list(top_10_prob)
          batched_line_content.append(line_content)
          iter_cnt += 1
        writer.writerows(batched_line_content)
        pbar.update(pred_batch_size)
        #if iter_cnt >= MAX_ITER:
          #break
