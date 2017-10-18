#!/bin/bash -e
# File: load_tensorflow.sh
# Author: Yukun Chen <cykustc@gmail.com>

if [ "$HOSTNAME" = ESC8000 ]; then
	source ~/anaconda2/bin/activate tensorflow
elif [ "$HOSTNAME" = yukun-mbp ]; then
	source ~/work/libs/virtualenv/tensorflow/bin/activate
elif [ "$HOSTNAME" = wang-imac-01.ist.psu.edu ]; then
	source ~/work/libs/virtualenv/tensorflow/bin/activate
fi
