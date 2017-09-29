r"""
A set of utils to do data preprocessing and get basic statistics.
"""
from google.apputils import app
import gflags
import numpy as np
import pandas as pd
import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
import csv
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

FLAGS = gflags.FLAGS

gflags.DEFINE_bool('loop_and_see_example', False,
                   'Loop through train_example.bson and see images.')

gflags.DEFINE_bool('get_img_cnt_hist', False,
                   'Get image count per product histogram.')

gflags.DEFINE_bool('get_img_size_stat', False,
                   'Calculate mean and std of image width and height.')

def loop_and_see(bson_file, show_img=False):
  """
  Return: a dictionary of {num of imgs for a product: cnt}
  """
  data = bson.decode_file_iter(open(bson_file, 'rb'))
  num_of_imgs_hist = {}
  for c, d in enumerate(data):
      product_id = d['_id']
      category_id = d['category_id'] # This won't be in Test data
      cnt = 0;
      for e, pic in enumerate(d['imgs']):
        if show_img:
          picture = np.array(imread(io.BytesIO(pic['picture'])))
          # do something with the picture, etc
          plt.imshow(picture);
          plt.pause(5)
        cnt += 1;
      if cnt in num_of_imgs_hist.keys():
        num_of_imgs_hist[cnt] += 1
      else:
        num_of_imgs_hist[cnt] = 1
  return num_of_imgs_hist


def get_img_cnt_per_prodcuct_histogram(bson_file,
    stat_file='../data/img_cnt_for_a_prodcuct_histogram.txt'):
  num_of_imgs_hist = loop_and_see(bson_file)
  print num_of_imgs_hist
  with open(stat_file, 'wb') as f:
    writer = csv.writer(f)
    for k, d in num_of_imgs_hist.items():
      writer.writerow([k, d])


def imgs_size_stat(bson_file,
    stat_file='../data/img_size_stat.txt'):
  data = bson.decode_file_iter(open(bson_file, 'rb'))
  num_of_imgs_hist = {}
  # num_of_imgs = 4369441 + 1128588 * 2 + 542792 * 3 + 1029075 * 4
  num_of_imgs = 12371293
  widths = num_of_imgs * [0]
  heights = num_of_imgs * [0]
  # widths = num_of_imgs * [None]
  # heights = num_of_imgs * [None]
  idx = 0;
  for c, d in enumerate(data):
    for e, pic in enumerate(d['imgs']):
      picture = np.array(imread(io.BytesIO(pic['picture'])))
      [w, h, c] = picture.shape
      widths[idx] = w
      heights[idx] = h
      idx += 1;
  width_mean = np.mean(widths);
  width_std = np.std(widths);
  height_mean = np.mean(heights);
  height_std = np.std(heights);
  with open(stat_file, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow([width_mean, width_std, height_mean, height_std])

def main(argv):
  if FLAGS.loop_and_see_example:
    loop_and_see('../data/train_example.bson', show_img=True)
  if FLAGS.get_img_cnt_hist:
    get_img_cnt_per_prodcuct_histogram('../data/train.bson')
  if FLAGS.get_img_size_stat:
    imgs_size_stat('../data/train.bson')

if __name__ == '__main__':
  app.run()
