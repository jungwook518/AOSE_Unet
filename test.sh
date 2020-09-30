#!/bin/bash


fs='44.100' ##16 32 48 44.100, 22.050,..... Other than 16,32,48, sr is automatically changed to nearest fs.
test_model='model_ckpt/bestmodel.pth'
test_data_root_folder='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/sample_audio/'
test_data_output_path='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/sample_output'

python test_DCUnet_jsdr_demand.py \
--fs $fs --test_model $test_model \
--test_data_root_folder $test_data_root_folder \
--test_data_output_path $test_data_output_path
