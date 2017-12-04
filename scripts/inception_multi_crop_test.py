#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_multi_crop_test.py
# Author: Yukun Chen <cykustc@gmail.com>
r"""
Example Usage:
  python scripts/inception_multi_crop_test.py --mode=resnetv1 --gpu=15 \
--model_path_for_pred=train_log/cdiscount-inceptionresnetv1/model-175000
"""
import os

from google.apputils import app
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string('mode', 'resnet',
                     'should be one of resnet or se-resnet.')

gflags.DEFINE_string('gpu', None,
                     'specify which gpu(s) to be used.')

gflags.DEFINE_bool('apply_augmentation', False,
                     'If true, Apply image augmentation. For training and'
                     'testing, we apply different augmentation')

gflags.DEFINE_string('model_path_for_pred', "",
									 'model path for prediction on test set.')

cmd = """tmux {} """


def run_test():
  gpus = []
  if FLAGS.gpu != None:
    gpus = FLAGS.gpu.split(',')
  per_panel_command = ''
  for i, gpu in enumerate(gpus):
    split_w_str = "split-window " if i > 0 else "new-session "
    test_command = ("python -m src.cdiscount_inception --mode={} "
        "--pred_test=True "
        "--model_path_for_pred={} "
        "--gpu={} ".format(FLAGS.mode, FLAGS.model_path_for_pred, gpu))
    if FLAGS.apply_augmentation:
      test_command += "--apply_augmentation "

    per_panel_command += (split_w_str +
        '"source ~/anaconda2/bin/activate tensorflow ; '
        'sleep {} ; {} ; read" \; select-layout even-vertical \; '.format(i*10,
             test_command))
  print cmd.format(per_panel_command)
  os.system(cmd.format(per_panel_command))


def main(argv):
  run_test()


if __name__ == '__main__':
  app.run()
