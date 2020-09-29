#!/bin/bash


fs='48'
test_model='model_ckpt/bestmodel.pth'
test_data_txt='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/sample/sample_data.txt'
test_data_output_path='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/sample_output'

python test_DCUnet_jsdr_demand.py \
--fs $fs --test_model $test_model \
--test_data_txt $test_data_txt \
--test_data_output_path $test_data_output_path
