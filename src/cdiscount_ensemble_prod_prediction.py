#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cdiscount_ensemble_prod_prediction.py
# Author: Yukun Chen <cykustc@gmail.com>
from google.apputils import app
import gflags
import csv
import os
import tqdm
import pickle
import glob

from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack import logger

from .cdiscount_prod_prediction import PerProdInfo

FLAGS=gflags.FLAGS

gflags.DEFINE_string('ensemble_config_file', "",
                     'Ensemble configuration file.'
                     'Each row has the format of <prod-top10-file>, weight')


class Ensemble():
  """ Class for make ensemble prediction based on predicted per product top 10
  probabilities.
  """
  def __init__(self, ensemble_config_file,
      inv_id_map_file="./data/inv_category_id_mapping.pkl"):
    self.fname = ensemble_config_file
    self.inv_map = pickle.load(open(inv_id_map_file,'rb'))
    self.top10_pred_filenames = []
    self.weights = []
    self.all_products = []
    self.load_all_top10()

  def load_all_top10(self):
    with open(self.fname, 'r') as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        print row
        self.top10_pred_filenames.append(row[0])
        self.weights.append(float(row[1]))
      logger.info("loaded {} top-10 per prod pred for ensemble...".format(
        len(self.weights)))
    for fname, weight in zip(self.top10_pred_filenames, self.weights):
      self.load_top10_per_prod(fname, weight)

  def load_top10_per_prod(self, top_10_agg_file, weight):
    """
    Load top 10 prediction make by some neural net model.
    fill self.all_products with ProdInfo()s.
    """
    with open(top_10_agg_file, 'r') as f:
      num_lines = sum(1 for line in f)
      f.seek(0)
      first_load = len(self.all_products) == 0
      reader = csv.reader(f)
      logger.info("loading top-10 per prod pred from {}.".format(
          top_10_agg_file))
      with tqdm.tqdm(total=num_lines, **get_tqdm_kwargs()) as pbar:
        for i, row in enumerate(reader):
          prod_id = int(row[0])
          pseudo_ids = [int(x) for x in row[1:10]]
          probs = [float(x) for x in row[11:20]]
          probs = [x * weight for x in probs]
          pbar.update(1)
          if not first_load:
            self.all_products[i].add_a_img_pred(prod_id, pseudo_ids, probs)
          else:
            new_prod = PerProdInfo(prod_id)
            new_prod.add_a_img_pred(prod_id, pseudo_ids, probs)
            self.all_products.append(new_prod)

  def make_pred(self):
    """make per product prediction.
    """
    prod_pred_filename = (self.fname +  "_ensemble.txt")
    logger.info("predicting...")
    with open(prod_pred_filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['_id', 'category_id'])
      with tqdm.tqdm(total=len(self.all_products), **get_tqdm_kwargs()) as pbar:
        for prod in self.all_products:
          pred_category = prod.pred_category(self.inv_map)
          writer.writerow([prod.prod_id, pred_category])
          pbar.update(1)


def main(argv):
  ensemble = Ensemble(FLAGS.ensemble_config_file)
  ensemble.make_pred()

if __name__ == "__main__":
  app.run()

