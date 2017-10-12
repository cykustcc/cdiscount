#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_data.py
# Author: Yukun Chen <cykustc@gmail.com>
from google.apputils import app
import gflags
import pickle
import bson                       # this is installed with the pymongo package
import io
import os
from tqdm import tqdm
from skimage.data import imread
import numpy as np
import cv2

from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow import *
from tensorpack.utils import logger

FLAGS = gflags.FLAGS

gflags.DEFINE_bool('get_per_pixel_mean_img', False,
                   'Calculate per pixel mean image.')

gflags.DEFINE_bool('loop_and_see_example_imgs', False,
                   'Loop and view example images.')

gflags.DEFINE_bool('test_batch_data', False,
                   'Test batch read mode speed.')

gflags.DEFINE_bool('create_lmdb', False,
                   'Create lmdb for training set for sequential read.')

gflags.DEFINE_bool('get_per_channel_mean', False,
                   'Get per channel mean.')

gflags.DEFINE_bool('get_per_channel_std', False,
                   'Get per channel std.')

gflags.DEFINE_bool('test_load_all_imgs_into_memory', False,
                   'Test loading all images from mongodb to memory.')

__all__ = ['Cdiscount']

class Cdiscount(RNGDataFlow):
  """
  Produces Cdiscount images of shape [180, 180, 3] and a label (category id)
  """
  def __init__(self, dir, filepath_label_file, name, shuffle=None,
               large_mem_sys=False):
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
    self.mapping, self.inv_mapping = self._get_category_id_mapping()
    if shuffle == None:
      shuffle = name == 'train'
    self.shuffle = shuffle
    self.large_mem_sys = large_mem_sys

    if self.large_mem_sys:
      self.all_imgs_dict = {}
      self._load_imgs_to_memory(name)

    self.imglist = self._get_img_list(name)
    for fname, _ in self.imglist[:10]:
      fname = os.path.join(self.full_path, fname)
      assert os.path.isfile(fname), fname

  def _get_category_id_mapping(self):
    with open('./data/category_id_mapping.pkl', 'rb') as f:
      mapping = pickle.load(f)
    with open('./data/inv_category_id_mapping.pkl', 'rb') as f:
      inv_mapping = pickle.load(f)
    return [mapping, inv_mapping]

  def _get_train_val_split(self, total_num_imgs):
    """Split the orginal training set to train and validation set. (0.98, 0.02)
      Return: a list of [True, False,...]
      True is training set datapoint and False is validation set datapoint.
    """
    np.random.seed(0)
    return list(np.greater(np.random.random(total_num_imgs), 0.02))

  def _load_imgs_to_memory(self, train_or_val_or_test):
    if train_or_val_or_test == 'train' or train_or_val_or_test == 'val':
      imgroot = './data/train_imgs/'
      bson_file = './data/train.bson'
      num_imgs = 12371293
    elif train_or_val_or_test == 'test':
      imgroot = './data/test_imgs/'
      bson_file = './data/test.bson'
      num_imgs = 3095080

    logger.info("Load all imgs into memeory...")
    bar = tqdm(total=num_imgs)
    data = bson.decode_file_iter(open(bson_file, 'rb'))
    for c, d in enumerate(data):
      product_id = d['_id']
      category_id = d['category_id'] # This won't be in Test data
      for e, pic in enumerate(d['imgs']):
        relative_path = os.path.join(str(category_id),
                                     '{}-{}.jpg'.format(product_id, e))
        fname = os.path.join(imgroot,
                             relative_path)
        self.all_imgs_dict[fname] = (
            np.array(imread(io.BytesIO(pic['picture']))),
            category_id)
        bar.update()

  def _get_img_list(self, train_or_val_or_test):
    with open(self.filepath_label_file) as f:
      ret = []
      for line in f.readlines():
        name, cls = line.strip().split(',')
        cls = int(cls)
        cls = self.mapping[cls]
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
      if self.large_mem_sys:
        im = self.all_imgs_dict[fname]
      else:
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

  def get_per_pixel_mean(self, mean_file='./data/img_mean.jpg'):
    """
    return a mean image of all (train and test) images of size
    180x180x3.
    """
    if os.path.exists(mean_file):
      mean_im = cv2.imread(mean_file, cv2.IMREAD_COLOR)
      assert mean_im is not None
    else:
      raise ValueError('Failed to find file: ' + mean_file)
    return mean_im

  def get_per_channel_mean(self):
    """
    return three values as mean of each channel.
    """
    mean = self.get_per_pixel_mean()
    return np.mean(mean, axis=(0, 1))

  def get_per_pixel_std(self, std_file='./data/img_std.jpg'):
    """
    return a std image of all (train and test) images of size
    180x180x3.
    """
    if os.path.exists(std_file):
      std_im = cv2.imread(std_file, cv2.IMREAD_COLOR)
      assert std_im is not None
    else:
      raise ValueError('Failed to find file: ' + mean_file)
    return std_im

  def get_per_channel_std(self):
    """
    return three values as std of each channel.
    """
    std = self.get_per_pixel_std()
    return np.sqrt(np.mean(np.square(std), axis=(0, 1)))

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
    ds = Cdiscount('./data/train_imgs_example',
                   './data/train_imgfilelist_example.txt',
                   'train')
    ds.reset_state()
    print len(ds._get_img_list('train'))
    print len(ds._get_img_list('val'))
    ds.loop_and_see_example_imgs(20)
  if FLAGS.get_per_pixel_mean_img:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds.get_per_pixel_mean()
  if FLAGS.test_batch_data:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds1 = BatchData(ds, 256, use_list=True)
    TestDataSpeed(ds1).start()
  if FLAGS.create_lmdb:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, './data/Cdiscount-train.lmdb')
  if FLAGS.get_per_channel_mean:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    print ds.get_per_channel_mean()
  if FLAGS.get_per_channel_std:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train')
    ds.reset_state()
    print ds.get_per_channel_std()
  if FLAGS.test_load_all_imgs_into_memory:
    ds = Cdiscount('./data/train_imgs',
                   './data/train_imgfilelist.txt',
                   'train',
                   large_mem_sys=True)
    ds.reset_state()




if __name__ == '__main__':
  app.run()

