# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from fairseq.data import FairseqDataset

#from .data_utils import *
#from .jung_collaters import Seq2SeqCollater
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import time

class AV_Lrs2_pickleDataset(FairseqDataset):



    def __init__(self,noise_pickle_paths,fs):

        self.data_paths = np.loadtxt(noise_pickle_paths,str)
        self.fs = fs
        
                
    def __getitem__(self, index):
        data_item = self.data_paths[index] if self.noi_paths is not None else None
        
        win_len = 1024*(fs/16)
        window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)

        data_wav,_ = torchaudio.load(data_item)
        
        data_wav_len = data_wav.shape[1]
        spec_noi = torchaudio.functional.spectrogram(waveform=data_wav, pad=0, window=window, n_fft=win_len, hop_length=win_len/4, win_length=win_len, power=None, normalized=False)
        input_wav_real = spec_noi[0,:,:,0]
        input_wav_imag = spec_noi[0,:,:,1]
        data = {"id": index,"data_wav_len":data_wav_len, "audio_wav" : [data_wav],"audio_data_Real":[input_wav_real], "audio_data_Imagine":[input_wav_imag]}

        
       
        data_wav_len = data["data_wav_len"]
        time_len = data["audio_data_Real"][0].shape[1]
        if time_len < 512:
            empty_in_r = torch.zeros(513,512)
            empty_in_r[:,:time_len]=data["audio_data_Real"][0]
            data["audio_data_Real"][0] = empty_in_r 
            empty_in_i = torch.zeros(513,512)
            empty_in_i[:,:time_len]=data["audio_data_Imagine"][0]
            data["audio_data_Imagine"][0] = empty_in_i
            
            empty_tar_r = torch.zeros(513,512)
            empty_tar_r[:,:time_len]=data["audio_data_Real"][1]
            data["audio_data_Real"][1] = empty_tar_r 
            empty_tar_i = torch.zeros(513,512)
            empty_tar_i[:,:time_len]=data["audio_data_Imagine"][1]
            data["audio_data_Imagine"][1] = empty_tar_i
        
        else:
            data["audio_data_Real"][0] = data["audio_data_Real"][0][:,:512] 
            data["audio_data_Imagine"][0] =  data["audio_data_Imagine"][0][:,:512] 
            
            data["audio_data_Real"][1] = data["audio_data_Real"][1][:,:512] 
            data["audio_data_Imagine"][1] =  data["audio_data_Imagine"][1][:,:512] 


        
      
        return data
        
    

    def __len__(self):
        return len(self.data_paths)
    





if __name__ == '__main__':
    data_path = "/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/demand_test_sort_noise_0db_num1.txt"
    train_dataset = AV_Lrs2_pickleDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,pin_memory = True, num_workers=8)
    for i, (batch_data) in enumerate(train_loader):
        audio = batch_data["tgt_wav_len"]
        print(batch_data)
        
        
        
    
        

    













