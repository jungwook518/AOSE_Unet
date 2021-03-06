#!/bin/bash
#     ||||||| /     /
#     (mO_Om)/     /
#        |  /     /
#    ----|----   /
#        |/  mb /
#        /\    /
#       |  |  /
#      /|  | /
#     /_____/             
#     |     |               _________    _________     _________
#     /     /              /________/|  /________/|   /________/|
#    /     /               |________|/  |________|/   | _____  || 
#   /     /                   | ||         | ||       | ||___| ||
#  /     /                    | ||         | ||       | |/___| ||
#                             | ||         | ||       |  ______|/
#                           __| ||___    __| ||___    | ||
#                          /__|_|/__/|  /__|_|/__/|   | ||
#                          |________|/  |________|/   |_|/


fs='48' ##16 32 48 44.100, 22.050,..... Other than 16,32,48, sr is automatically changed to nearest fs.
test_model='model_ckpt/bestmodel.pth'
test_data_root_folder='../sample_48/'
test_data_output_path='/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/sample_output'

python test_DCUnet_jsdr_demand.py \
--fs $fs --test_model $test_model \
--test_data_root_folder $test_data_root_folder \
--test_data_output_path $test_data_output_path
