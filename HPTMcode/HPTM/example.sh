#!/bin/bash

#This is simple example how to use bi-DCTM for training and testing.

#The train set is a very small part of training set with 5,000 documents
#
#Check ./input/ to show the input file: training set.
#Check ./output to show the output.

make clean
echo
./configure
make
echo
rm -f ./output/*

echo

time ./main est setting.txt ./input/train.txt ./output/

echo