#!/bin/bash

gpu='0'
exp_day='1027'
snr='0'
train_data_root_folder='/home/nas/DB/[DB]_voice_corpus/train/practice/'
val_data_root_folder='/home/nas/DB/[DB]_voice_corpus/test/practice/'
batch_size='10' #batch 10 --> 11G gpu memory
learning_rate='0.0001'
frame_num='128' #-->128,256..2's power
fs='16' #fixing
modelsave_path='model_ckpt_prac/'

python train_DCUnet_jsdr_demand.py \
--gpu $gpu \
--modelsave_path $modelsave_path \
--exp_day $exp_day \
--snr $snr \
--train_data_root_folder $train_data_root_folder \
--val_data_root_folder $val_data_root_folder \
--batch_size $batch_size --learning_rate $learning_rate \
--frame_num $frame_num \
--fs $fs \
