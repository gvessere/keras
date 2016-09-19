#!/bin/bash


#THEANO_FLAGS=device=gpu3 python cifar10_cnn.py 0.25 0.25 0.55 0.55 > dropout2_0.25_0.25_0.55_0.55.log # 0.1
#THEANO_FLAGS=device=gpu3 python cifar10_cnn.py 0.25 0.25 0.6 0.6 > dropout2_0.25_0.25_0.6_0.6.log # 0.1
THEANO_FLAGS=device=gpu3 python cifar10_cnn.py 0.25 0.25 0.65 0.65 > dropout2_0.25_0.25_0.65_0.65.log
THEANO_FLAGS=device=gpu3 python cifar10_cnn.py 0.25 0.25 0.7 0.7 > dropout2_0.25_0.25_0.7_0.7.log
