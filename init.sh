#!/bin/bash -e
# File: init.sh
# Author: Yukun Chen <cykustc@gmail.com>
. ~/work/libs/virtualenv/tensorflow/bin/activate

#unlink data
ln -s /media/data/data/CDisCount data
sudo mount /dev/sda2 /media/data
