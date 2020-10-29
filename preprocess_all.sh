#!/bin/bash

# Preprocesses the raw corpus and saves a copy of it somewhere
data_train="/u/felipe/HTML/IFT6285-Automne2020/blogs/"
data_test="/u/felipe/HTML/IFT6285-Automne2020/blogs/test/test1/"

# prepare directories
data_train_processed="data/preprocessed/train/"
data_test_processed="data/preprocessed/test/"
logs="data/logs/"
mkdir -p $data_train_processed
mkdir -p $data_test_processed
mkdir -p $logs

# call the preprocessor with pkscreen
pkscreen bash -c "{ time python corpus/preprocessing.py $data_train $data_train_processed ; } 2> $logs/time_preprocess_train.log"
pkscreen bash -c "{ time python corpus/preprocessing.py $data_test $data_test_processed ; } 2> $logs/time_preprocess_test.log