#!/bin/bash

clean_train_txt='/home/nas/DB/[DB]_voice_corpus/test/clean_test_prac.txt'
noise_txt='/home/nas/DB/[DB]_DEMAND/demand_path.txt'
save_path='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/prac/'
snr='0'
fs='16' #fixing

cd data_augment
python data_aug_demand_dataset.py \
--clean_train_txt $clean_train_txt \
--noise_txt $noise_txt \
--save_path $save_path \
--snr $snr \
--fs $fs \
