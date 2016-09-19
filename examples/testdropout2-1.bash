#!/bin/bash


#THEANO_FLAGS=device=gpu0 python cifar10_cnn.py 0.25 0.45 0.5 0.7 > dropout2_0.25_0.45_0.5_0.7.log # ok
THEANO_FLAGS=device=gpu0 python cifar10_cnn.py 0.25 0.25 0.5 0.5 > dropout2_0.25_0.25_0.5_0.5.log # ?
THEANO_FLAGS=device=gpu0 python cifar10_cnn.py 0.3 0.3 0.55 0.55 > dropout2_0.3_0.3_0.55_0.55.log
THEANO_FLAGS=device=gpu0 python cifar10_cnn.py 0.35 0.35 0.6 0.6 > dropout2_0.35_0.35_0.6_0.6.log

