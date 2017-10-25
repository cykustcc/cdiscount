#!/bin/bash -e
# File: run_multi_command.sh
# Author: Yukun Chen <cykustc@gmail.com>
tmux \
  new-session  "echo "helloworld" ; read " \; \
  split-window "echo "helloworld" ; read " \; \
  split-window "echo "helloworld" ; read " \; \
  split-window "echo "helloworld" ; read " \; \
  select-layout even-vertical
