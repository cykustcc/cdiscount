#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_data.py
# Author: Yukun Chen <cykustc@gmail.com>
import os
import tqdm
import numpy as np
import cv2

from tensorpack.dataflow.base import RNGDataFlow

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
    assert name in ['train', 'test']
    self.full_path = dir
    self.filepath_label_file = filepath_label_file
    self.imglist = self._get_img_list()
    if shuffle == None:
      shuffle = name == 'train'
    self.shuffle = shuffle

    for fname, _ in self.imglist[:10]:
      fname = os.path.join(self.full_path, fname)
      assert os.path.isfile(fname), fname

  def _get_img_list(self):
    with open(self.filepath_label_file) as f:
      ret = []
      for line in f.readlines():
        name, cls = line.strip().split(',')
        cls = int(cls)
        ret.append([name, cls])
    assert len(ret)
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

  def loop_and_see_example_imgs(self, num=20):
    cnt = 0
    for im, label in self.get_data():
      cv2.imshow('Cdiscount image', im)
      cv2.waitKey(0)
      cnt += 1
      if cnt >= num:
        break

if __name__ == '__main__':
  ds = Cdiscount('../data/train_imgs_example',
                 '../data/train_imgfilelist_example.txt',
                 'train')
  ds.reset_state()
  print ds._get_img_list()
  ds.loop_and_see_example_imgs(20)

