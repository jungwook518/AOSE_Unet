#!/bin/bash

clean_train='/home/nas/DB/[DB]_voice_corpus/train/clean_train_prac.txt'
noise_train='/home/nas/DB/[DB]_voice_corpus/train/noise_train_prac.txt'
clean_test='/home/nas/DB/[DB]_voice_corpus/test/clean_test_prac.txt'
noise_test='/home/nas/DB/[DB]_voice_corpus/test/noise_test_prac.txt'
train_save_path='/home/nas/DB/[DB]_voice_corpus/train/practice/'
test_save_path='/home/nas/DB/[DB]_voice_corpus/test/practice/'
fs='16'

cd data_feature
python demand_data_get_feature.py \
--clean_train $clean_train \
--noise_train $noise_train \
--clean_test $clean_test \
--noise_test $noise_test \
--train_save_path $train_save_path \
--test_save_path $test_save_path \
--fs $fs \
