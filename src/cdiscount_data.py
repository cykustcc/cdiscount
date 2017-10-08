#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_data.py
# Author: Yukun Chen <cykustc@gmail.com>
from google.apputils import app
import gflags
import os
from tqdm import tqdm
import numpy as np
import cv2

from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow import *

FLAGS = gflags.FLAGS

gflags.DEFINE_bool('get_per_pixel_mean_img', False,
                   'Calculate per pixel mean image.')

gflags.DEFINE_bool('loop_and_see_example_imgs', False,
                   'Loop and view example images.')

gflags.DEFINE_bool('test_batch_data', False,
                   'Test batch read mode speed.')

gflags.DEFINE_bool('create_lmdb', False,
                   'Create lmdb for training set for sequential read.')


__all__ = ['Cdiscount']

class Cdiscount(RNGDataFlow):
  """
  Produces Cdiscount images of shape [180, 180, 3] and a label (category id)
  """
  def __init__(self, dir, filepath_label_file, name, shuffle=None):
    """
    Args:
      dir (str): directory containing the 2 level structure containing images.
      filepath_label_file (str): filepath of the file has content: "img_path,label" per row.
      name (str): 'train' or 'test'.
    """
    assert os.path.isdir(dir), dir
    assert name in ['train', 'val', 'test']
    self.full_path = dir
    self.filepath_label_file = filepath_label_file
    self.imglist = self._get_img_list(name)
    if shuffle == None:
      shuffle = name == 'train'
    self.shuffle = shuffle

    for fname, _ in self.imglist[:10]:
      fname = os.path.join(self.full_path, fname)
      assert os.path.isfile(fname), fname

  def _get_train_val_split(self, total_num_imgs):
    """Split the orginal training set to train and validation set. (0.8, 0.2)
      Return: a list of [True, False,...]
      True is training set datapoint and False is validation set datapoint.
    """
    np.random.seed(0)
    return list(np.greater(np.random.random(total_num_imgs), 0.2))

  def _get_img_list(self, train_or_val_or_test):
    with open(self.filepath_label_file) as f:
      ret = []
      for line in f.readlines():
        name, cls = line.strip().split(',')
        cls = int(cls)
        ret.append([name, cls])
    assert len(ret)
    train_val_split = self._get_train_val_split(len(ret))
    if train_or_val_or_test == 'train':
      return [ret[i] for i in xrange(len(ret)) if train_val_split[i]]
    elif train_or_val_or_test == 'val':
      return [ret[i] for i in xrange(len(ret)) if not train_val_split[i]]
    else:
      return ret

  def get_data(self):
    for fname, label in self.get_filename_label():
      im = cv2.imread(fname, cv2.IMREAD_COLOR)
      assert im is not None, fname
      yield [im, label]

  def get_filename_label(self):
    idxs = np.arange(self.size())
    if self.shuffle:
      self.rng.shuffle(idxs)
    for k in idxs:
      fname, label = self.imglist[k]
      fname = os.path.join(self.full_path, fname)
      yield [fname, label]

  def size(self):
    return len(self.imglist)

  def get_per_pixel_mean(self, mean_file='../data/img_mean.jpg'):
    """
    return a mean image of all (train and test) images of size
    180x180x3.
    """
    if os.path.exists(mean_file):
      mean_im = cv2.imread(mean_file, cv2.IMREAD_COLOR)
      assert mean_im is not None
    else:
      print "Calculating per pixel mean image."
      bar = tqdm(total=self.size())
      mean_im = np.zeros((180, 180, 3), np.double)
      for im, label in self.get_data():
        mean_im += 1.0 * im / self.size()
        bar.update()
      cv2.imwrite(mean_file, mean_im)
    return mean_im

  def get_per_channel_mean(self):
    """
    return three values as mean of each channel.
    """
    mean = self.get_per_pixel_mean()
    return np.mean(mean, axis=(0, 1))

  def loop_and_see_example_imgs(self, num=20):
    cnt = 0
    for im, label in self.get_data():
      cv2.imshow('Cdiscount image', im)
      cv2.waitKey(0)
      cnt += 1
      if cnt >= num:
        break

def main(argv):
  if FLAGS.loop_and_see_example_imgs:
    ds = Cdiscount('../data/train_imgs_example',
                   '../data/train_imgfilelist_example.txt',
                   'train')
    ds.reset_state()
    print len(ds._get_img_list('train'))
    print len(ds._get_img_list('val'))
    ds.loop_and_see_example_imgs(20)
  if FLAGS.get_per_pixel_mean_img:
    ds = Cdiscount('../data/train_imgs',
                   '../data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds.get_per_pixel_mean()
  if FLAGS.test_batch_data:
    ds = Cdiscount('../data/train_imgs',
                   '../data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds1 = BatchData(ds, 256, use_list=True)
    TestDataSpeed(ds1).start()
  if FLAGS.create_lmdb:
    ds = Cdiscount('../data/train_imgs',
                   '../data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, '../data/Cdiscount-train.lmdb')




if __name__ == '__main__':
  app.run()
