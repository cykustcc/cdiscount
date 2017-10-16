#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_prod_prediction.py
# Author: Yukun Chen <cykustc@gmail.com>
from google.apputils import app
import gflags
import csv
import os
import tqdm
import pickle

from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack import logger

FLAGS=gflags.FLAGS

gflags.DEFINE_string('nn_pred_file', "",
                     'Neural Net outputed top 10 per image prediction file.')

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

  def pred_category(self, inv_mapping, pos_decay_base=1.1):
    """After all associated images' prediction is added, make prediction about
    this product.
    """
    all_pseudo_ids = [item for sublist in self.pseudo_ids for item in sublist]
    all_pseudo_ids = list(set(all_pseudo_ids))
    aggregated_prob = {pseudo_id:0 for pseudo_id in all_pseudo_ids}
    for id_sub_list, prob_sub_list in zip(self.pseudo_ids, self.probs):
      for k, (pseudo_id, prob) in enumerate(zip(id_sub_list, prob_sub_list)):
        aggregated_prob[pseudo_id] += pos_decay_base ** (-k) * prob
    aggregated_prob = sorted(aggregated_prob.iteritems(), key=lambda (k,v):
        (v,k), reverse=True)
    return inv_mapping[aggregated_prob[0][0]]


class ProdPred():
  """ Class for make per product prediction.
  """
  def __init__(self, top10_per_im_inference_file,
      inv_id_map_file="./data/inv_category_id_mapping.pkl"):
    self.fname = top10_per_im_inference_file
    self.num_products = 0
    self.all_products = []
    #all_products_file = self.fname.replace(".txt", ".pkl")
    #if os.path.exists(all_products_file):
      #logger.info("loading from {}".format(all_products_file))
      #self.all_products = pickle.load(open(all_products_file, 'rb'))
    #else:
    self.load_top10_per_im_inf(self.fname)
    #pickle.dump(self.all_products, open(all_products_file, 'wb'))
    self.num_products = len(self.all_products)
    self.inv_map = pickle.load(open(inv_id_map_file,'rb'))


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
      logger.info("products info from neural net loaded.")
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
    prod_pred_filename = self.fname.replace(".txt", "_prod.txt")
    logger.info("predict category for each of {} products...".format(
        self.num_products))
    with open(prod_pred_filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['_id', 'category_id'])
      with tqdm.tqdm(total=self.num_products, **get_tqdm_kwargs()) as pbar:
        for prod in self.all_products:
          pred_category = prod.pred_category(self.inv_map)
          writer.writerow([prod.prod_id, pred_category])
          pbar.update(1)


def main(argv):
  prod_pred = ProdPred(FLAGS.nn_pred_file)
  prod_pred.make_pred()

if __name__ == "__main__":
  app.run()
