import numpy as np
import pandas as pd
import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
import csv
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data


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
          picture = imread(io.BytesIO(pic['picture']))
          # do something with the picture, etc
          plt.imshow(picture);
          plt.pause(5)
        cnt += 1;
      if cnt in num_of_imgs_hist.keys():
        num_of_imgs_hist[cnt] += 1
      else:
        num_of_imgs_hist[cnt] = 1
  return num_of_imgs_hist

def get_img_cnt_for_a_prodcuct_histogram(bson_file,
    stat_file='../data/img_cnt_for_a_prodcuct_histogram.txt'):
  num_of_imgs_hist = loop_and_see(bson_file)
  print num_of_imgs_hist
  with open(stat_file, 'wb') as f:
    writer = csv.writer(f)
    for k, d in num_of_imgs_hist.items():
      writer.writerow([k, d])



if __name__ == '__main__':
  # loop_and_see('../data/train_example.bson', show_img=False)
  get_img_cnt_for_a_prodcuct_histogram('../data/train.bson')
