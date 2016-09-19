#!/bin/bash


#THEANO_FLAGS=device=gpu1 python cifar10_cnn.py 0.4 0.4 0.65 0.65 > dropout2_0.4_0.4_0.65_0.65.log # 0.1
#THEANO_FLAGS=device=gpu1 python cifar10_cnn.py 0.45 0.45 0.7 0.7 > dropout2_0.45_0.45_0.7_0.7.log # 0.1
THEANO_FLAGS=device=gpu1 python cifar10_cnn.py 0.25 0.25 0.5 0.5 > dropout2_0.25_0.25_0.5_0.5.log
THEANO_FLAGS=device=gpu1 python cifar10_cnn.py 0.3 0.3 0.5 0.5 > dropout2_0.3_0.3_0.5_0.5.log
