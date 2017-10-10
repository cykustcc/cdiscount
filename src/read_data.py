r"""
A set of utils to do data preprocessing and get basic statistics.
"""
from google.apputils import app
import gflags
import numpy as np
import cv2
import pickle
import pandas as pd
import io
import os
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
import csv
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
from ipywidgets import IntProgress
from tqdm import tqdm

FLAGS = gflags.FLAGS

gflags.DEFINE_bool('loop_and_see_example', False,
                   'Loop through train_example.bson and see images.')

gflags.DEFINE_bool('get_img_cnt_hist', False,
                   'Get image count per product histogram.')

gflags.DEFINE_bool('get_img_cnt_hist_test', False,
                   'Get image count per product histogram for testing data.')

gflags.DEFINE_bool('get_img_size_stat', False,
                   'Calculate mean and std of image width and height.')

gflags.DEFINE_bool('store_img_as_jpg', False,
                   'Store imgs into a 2 level folder hierachy in the format of'
                   'jpg.')

gflags.DEFINE_bool('get_per_pixel_mean', False,
                   'Calculate per pixel mean image and store it to '
                   './data/img_mean.jpg')

gflags.DEFINE_bool('get_per_pixel_std', False,
                   'Calculate per pixel std image and store it to '
                   './data/img_std.jpg')

gflags.DEFINE_bool('store_category_id_mapping', False,
                   'Store the mapping from category id to proxy id:'
                   '1 to # of category.')


def loop_and_see(bson_file, show_img=False):
  """
  Return: a dictionary of {num of imgs for a product: cnt}
  """
  data = bson.decode_file_iter(open(bson_file, 'rb'))
  num_of_imgs_hist = {}
  for c, d in enumerate(data):
      product_id = d['_id']
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
    stat_file='./data/img_cnt_for_a_prodcuct_histogram.txt'):
  num_of_imgs_hist = loop_and_see(bson_file)
  print num_of_imgs_hist
  with open(stat_file, 'wb') as f:
    writer = csv.writer(f)
    for k, d in num_of_imgs_hist.items():
      writer.writerow([k, d])


def imgs_size_stat(bson_file,
    stat_file='./data/img_size_stat.txt'):
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


def imgs_store_jpg(bson_file,
    imgfilelist='./data/train_imgfilelist.txt',
    imgroot='./data/train_imgs/'):
  # Create categories folders.
  categories = pd.read_csv('./data/category_names.csv', index_col='category_id')

  if not os.path.exists(imgroot):
    os.mkdir(imgroot)
  for category in tqdm(categories.index):
    category_folder_path = os.path.join(imgroot, str(category))
    if not os.path.exists(category_folder_path):
      os.mkdir(category_folder_path)
  # store image as jpg files.
  num_products = 7069896
  bar = tqdm(total=num_products)

  data = bson.decode_file_iter(open(bson_file, 'rb'))
  imgfilelist_content = []
  for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    for e, pic in enumerate(d['imgs']):
      relative_path = os.path.join(str(category_id),
                                   '{}-{}.jpg'.format(product_id, e))
      fname = os.path.join(imgroot,
                           relative_path)
      with open(fname, 'wb') as f:
          f.write(pic['picture'])
      imgfilelist_content.append([relative_path, category_id])
    bar.update()
  with open(imgfilelist, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(imgfilelist_content)


def get_per_pixel_mean(bson_file,
                       mean_file='./data/img_mean.jpg'):
  """
  return a mean image of all (train and test) images of size
  180x180x3.
  """
  print "Calculating per pixel mean image."
  num_imgs = 12371293
  bar = tqdm(total=num_imgs)
  data = bson.decode_file_iter(open(bson_file, 'rb'))
  imgfilelist_content = []
  mean_im = np.zeros((180, 180, 3), np.double)
  for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    for e, pic in enumerate(d['imgs']):
      im = np.array(imread(io.BytesIO(pic['picture'])))
      mean_im += 1.0 * im / num_imgs
      bar.update()
  print mean_im
  cv2.imwrite(mean_file, mean_im)

def get_per_pixel_std(bson_file,
                      mean_file='./data/img_mean.jpg',
                      std_file="./data/img_std.jpg"):
  """
  return a std image of all (train and test) images of size
  180x180x3.
  """
  print "Calculating per pixel std image."
  num_imgs = 12371293
  bar = tqdm(total=num_imgs)
  data = bson.decode_file_iter(open(bson_file, 'rb'))
  imgfilelist_content = []
  mean_im = cv2.imread(mean_file, cv2.IMREAD_COLOR)
  std_im = np.zeros((180, 180, 3), np.double)
  for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    for e, pic in enumerate(d['imgs']):
      im = np.array(imread(io.BytesIO(pic['picture'])))
      std_im += np.square(1.0 * im - 1.0 * mean_im) / num_imgs
      bar.update()
  std_im = np.sqrt(std_im)
  print std_im
  cv2.imwrite(std_file, std_im)


def store_category_id_mapping():
  categories = pd.read_csv('./data/category_names.csv', index_col='category_id')
  category_ids = list(categories.index)
  mapping_dict = {id:new_id for id,new_id in zip(category_ids,
    range(len(category_ids)))};
  inverse_mapping_dict = {new_id:id for id,new_id in zip(category_ids,
    range(len(category_ids)))};
  with open('./data/category_id_mapping.pkl', 'wb') as f:
    pickle.dump(mapping_dict, f)
  with open('./data/inv_category_id_mapping.pkl', 'wb') as f:
    pickle.dump(inverse_mapping_dict, f)


def imgs_store_jpg_test(bson_file,
    imgfilelist='./data/test_imgfilelist.txt',
    imgroot='./data/test_imgs/'):
  # Create categories folders.
  num_imgs = 3095080
  num_of_imgs_per_folder = 5000
  total_num_of_folder = int(np.ceil(
      1.0 * num_imgs / num_of_imgs_per_folder))
  for folder_idx in xrange(total_num_of_folder):
    folder_path = os.path.join(imgroot, str(folder_idx))
    if not os.path.exists(folder_path):
      os.mkdir(folder_path)
  bar = tqdm(total=num_imgs)

  data = bson.decode_file_iter(open(bson_file, 'rb'))
  imgfilelist_content = []
  idx = 0
  for c, d in enumerate(data):
    product_id = d['_id']
    for e, pic in enumerate(d['imgs']):
      folder_name = idx / num_of_imgs_per_folder
      relative_path = os.path.join(str(folder_name),
                                   '{}-{}.jpg'.format(product_id, e))
      fname = os.path.join(imgroot,
                           relative_path)
      with open(fname, 'wb') as f:
          f.write(pic['picture'])
      imgfilelist_content.append([relative_path])
      idx += 1
      bar.update()

  with open(imgfilelist, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(imgfilelist_content)


def main(argv):
  if FLAGS.loop_and_see_example:
    loop_and_see('./data/train_example.bson', show_img=True)
  if FLAGS.get_img_cnt_hist:
    get_img_cnt_per_prodcuct_histogram('./data/train.bson')
  if FLAGS.get_img_cnt_hist_test:
    get_img_cnt_per_prodcuct_histogram('./data/test.bson',
        stat_file='./data/img_cnt_for_a_prodcuct_histogram_test.txt')
  if FLAGS.get_img_size_stat:
    imgs_size_stat('../data/train.bson')
  if FLAGS.store_img_as_jpg:
    imgs_store_jpg('./data/train.bson',
                   './data/train_imgfilelist.txt',
                   './data/train_imgs')
    imgs_store_jpg_test('../data/test.bson',
                   './data/test_imgfilelist.txt',
                   './data/test_imgs')
  if FLAGS.store_category_id_mapping:
    store_category_id_mapping()
  if FLAGS.get_per_pixel_mean:
    get_per_pixel_mean('./data/train.bson')
  if FLAGS.get_per_pixel_std:
    get_per_pixel_std('./data/train.bson')

if __name__ == '__main__':
  app.run()
