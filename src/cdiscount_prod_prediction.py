#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_prod_prediction.py
# Author: Yukun Chen <cykustc@gmail.com>
r"""
Make per product category predicction based on per image prediction from
Neural nets.
Example usage:
  python -m src.cdiscount_prod_prediction \
--nn_pred_file=data/pred/<filename>.txt \
--pos_decay_base=1.1

python -m src.cdiscount_prod_prediction \
--nn_pred_file="./data/pred/pred-cdiscount-resnet-d50-step180000test-*.txt"
"""
from google.apputils import app
import gflags
import csv
import os
import tqdm
import pickle
import glob

from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack import logger

FLAGS=gflags.FLAGS

gflags.DEFINE_string('nn_pred_file', "",
                     'Neural Net outputed top 10 per image prediction file.')

gflags.DEFINE_float('pos_decay_base', 1.0,
                    'position decay base.')

gflags.DEFINE_bool('store_agg_top10', False,
                   'If true, store per product aggregated top 10'
                   'probabilities prediction')


class PerProdInfo():
  """ Per product prediction info given by neural net per image info.
  A product may have multiple associated images.
  """
  def __init__(self, prod_id):
    self.prod_id = prod_id
    self.img_ids = []
    self.pseudo_ids = []
    self.probs = []
    self.num_of_imgs = 0

  def disp(self):
    print "prod_id = {}, with {} imgs.".format(self.prod_id, self.num_of_imgs)
    print self.img_ids
    print self.pseudo_ids
    print self.probs

  def add_a_img_pred(self, img_id, pseudo_ids, probs):
    """Add prediction about an associated image to the Info about this product.
    Inputs:
      img_id : fname of the image.
      pseudo_ids : predicted top 10 speudo ids of the added image.
                   within range of [0, 5699]
      probs : probability of top 10 predicted categories.
    """
    self.img_ids.append(img_id)
    self.pseudo_ids.append(pseudo_ids)
    self.probs.append(probs)
    self.num_of_imgs += 1

  def aggregated_id_prob(self, pos_decay_base):
    """
    Return:
      A dictionary of {pseudo_id: prob} aggregated from all associated imgs'
      top 10 prediction.
    """
    all_pseudo_ids = [item for sublist in self.pseudo_ids for item in sublist]
    all_pseudo_ids = list(set(all_pseudo_ids))
    aggregated_prob = {pseudo_id:0 for pseudo_id in all_pseudo_ids}
    for id_sub_list, prob_sub_list in zip(self.pseudo_ids, self.probs):
      for k, (pseudo_id, prob) in enumerate(zip(id_sub_list, prob_sub_list)):
        aggregated_prob[pseudo_id] += pos_decay_base ** (-k) * prob
    return aggregated_prob

  def pred_category(self, inv_mapping, pos_decay_base=1.1):
    """After all associated images' prediction is added, make prediction about
    this product.
    """
    aggregated_prob = self.aggregated_id_prob(pos_decay_base)
    aggregated_prob = sorted(aggregated_prob.iteritems(), key=lambda (k,v):
        (v,k), reverse=True)
    return inv_mapping[aggregated_prob[0][0]]


class ProdPred():
  """ Class for make per product prediction.
  """
  def __init__(self, top10_per_im_inference_file,
      inv_id_map_file="./data/inv_category_id_mapping.pkl",
      pos_decay_base=1.0):
    self.fname = top10_per_im_inference_file
    self.num_products = 0
    self.all_products = []
    self.load_top10_per_im_inf(self.fname)
    self.num_products = len(self.all_products)
    self.inv_map = pickle.load(open(inv_id_map_file,'rb'))
    self.pos_decay_base = pos_decay_base


  def load_top10_per_im_inf(self, top_10_per_im_inference_file):
    """
    Load top 10 prediction make by some neural net model.
    fill self.all_products with ProdInfo()s.
    """
    prev_prod_id = None
    with open(top_10_per_im_inference_file, 'r') as f:
      num_lines = sum(1 for line in f)
      f.seek(0)
      reader = csv.reader(f)
      logger.info("loading products info from neural net from {}.".format(
          top_10_per_im_inference_file))
      with tqdm.tqdm(total=num_lines, **get_tqdm_kwargs()) as pbar:
        for i, row in enumerate(reader):
          fname = row[0]
          prod_id = int(fname.strip().split('/')[1].split('-')[0])
          pseudo_ids = [int(x) for x in row[1:10]]
          probs = [float(x) for x in row[11:20]]
          pbar.update(1)
          if prod_id == prev_prod_id:
            last_idx = len(self.all_products) - 1
            self.all_products[last_idx].add_a_img_pred(fname, pseudo_ids, probs)
          else:
            new_prod = PerProdInfo(prod_id)
            new_prod.add_a_img_pred(fname, pseudo_ids, probs)
            self.all_products.append(new_prod)
            prev_prod_id = prod_id

  def make_pred(self):
    """make per product prediction.
    """
    pos_decay_str = str(self.pos_decay_base).replace(".", "_")
    prod_pred_filename = self.fname.replace(".txt", "_" + pos_decay_str + "prod.txt")
    logger.info("predicting for each of {} products...".format(
        self.num_products))
    with open(prod_pred_filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['_id', 'category_id'])
      with tqdm.tqdm(total=self.num_products, **get_tqdm_kwargs()) as pbar:
        for prod in self.all_products:
          pred_category = prod.pred_category(self.inv_map,
              self.pos_decay_base)
          writer.writerow([prod.prod_id, pred_category])
          pbar.update(1)


class ProdPredMulti():
  """ Class for make per product prediction based on multiple per product
  predictions from a few per image prediction, where each per image prediction
  is based on an augmented set of test set."""
  def __init__(self, top10_per_im_inference_file_patterns,
      inv_id_map_file="./data/inv_category_id_mapping.pkl",
      pos_decay_base=1.0, store_agg_top10=False):
    self.fname = top10_per_im_inference_file_patterns.replace("*","")
    self.per_im_pred_files = glob.glob(top10_per_im_inference_file_patterns)
    logger.info("loading products info from {} neural net pred...".format(
        len(self.per_im_pred_files)))
    self.inv_map = pickle.load(open(inv_id_map_file,'rb'))
    self.pos_decay_base = pos_decay_base
    self.prod_preds = [ProdPred(x, inv_id_map_file, self.pos_decay_base) for x in
        self.per_im_pred_files]
    self.num_preds = len(self.prod_preds)
    self.store_agg_top10=store_agg_top10

  def make_pred(self):
    """ Make per product prediction based on per-image-multi-croped test set."""
    pos_decay_str = str(self.pos_decay_base).replace(".", "_")
    prod_pred_filename = (self.fname +  "_" + pos_decay_str +
        "prod_{}crop.txt".format(len(self.per_im_pred_files)))
    agg_top_ten_filename = prod_pred_filename.replace("crop", "crop_agg")
    logger.info("predicting for each of {} products based on {}-crops...".format(
        self.prod_preds[0].num_products, self.num_preds))

    def merge_dict(x, y):
      return { k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y) }

    with open(prod_pred_filename, 'w') as f:
      if self.store_agg_top10:
        f_agg = open(agg_top_ten_filename, 'w')
        f_agg_writer = csv.writer(f_agg)
      writer = csv.writer(f)
      writer.writerow(['_id', 'category_id'])
      with tqdm.tqdm(total=self.prod_preds[0].num_products,
                     **get_tqdm_kwargs()) as pbar:
        for prod_list in zip(*[x.all_products for x in self.prod_preds]):
          aggregated_id_prob_dict = {} # aggregated id prob for all crops' prediction
          for prod in prod_list:
            aggregated_id_prob_dict = merge_dict(aggregated_id_prob_dict,
                prod.aggregated_id_prob(self.pos_decay_base))
          aggregated_prob = sorted(aggregated_id_prob_dict.iteritems(),
                                   key=lambda (k,v): (v,k), reverse=True)
          #only top 10 in aggregated_prob are stored as aggregated top 10.
          if self.store_agg_top10:
            tmp = zip(*aggregated_prob[0:10])
            line = [prod.prod_id] + list(tmp[0]) + list(tmp[1])
            f_agg_writer.writerow(line)
          pred_category = self.inv_map[aggregated_prob[0][0]]
          writer.writerow([prod.prod_id, pred_category])
          pbar.update(1)
      if self.store_agg_top10:
        f_agg.close()


def main(argv):
  print FLAGS.nn_pred_file
  if "*" in FLAGS.nn_pred_file:
  # multi-crop testing. 10 crop may be a good number
    prod_pred = ProdPredMulti(FLAGS.nn_pred_file,
        pos_decay_base=FLAGS.pos_decay_base,
        store_agg_top10=FLAGS.store_agg_top10)
  else:
  # single-crop testing.
    prod_pred = ProdPred(FLAGS.nn_pred_file, pos_decay_base=FLAGS.pos_decay_base)
  prod_pred.make_pred()

if __name__ == "__main__":
  app.run()
