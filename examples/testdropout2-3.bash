#!/bin/bash



THEANO_FLAGS=device=gpu2 python cifar10_cnn.py 0.35 0.35 0.5 0.5 > dropout2_0.35_0.35_0.5_0.5.log
THEANO_FLAGS=device=gpu2 python cifar10_cnn.py 0.4 0.4 0.5 0.5 > dropout2_0.4_0.4_0.5_0.5.log
THEANO_FLAGS=device=gpu2 python cifar10_cnn.py 0.45 0.45 0.5 0.5 > dropout2_0.45_0.45_0.5_0.5.log
THEANO_FLAGS=device=gpu2 python cifar10_cnn.py 0.25 0.25 0.5 0.5 > dropout2_0.25_0.25_0.5_0.5.log
