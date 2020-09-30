#!/bin/bash

gpu='0'
exp_day='0930'
snr='0'
train_data='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/demand_train_sort_noise_0db_num1.txt'
test_data='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/demand_test_sort_noise_0db_num1.txt'
batch_size='10' #batch 10 --> 11G gpu memory
learning_rate='0.0001'
frame_num='128'
fs='16' 
modelsave_path='model_ckpt/'

python train_DCUnet_jsdr_demand.py \
--gpu $gpu \
--modelsave_path $modelsave_path \
--exp_day $exp_day \
--snr $snr \
--train_data $train_data \
--test_data $test_data \
--batch_size $batch_size --learning_rate $learning_rate \
--frame_num $frame_num \
--fs $fs \
