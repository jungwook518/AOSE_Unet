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



    def __init__(self,noise_pickle_paths):

        self.data_paths = np.loadtxt(noise_pickle_paths,str)
        
        
                
    def __getitem__(self, index):
        data_item = self.data_paths[index]
    
        with open(data_item, 'rb') as f:
            data = pickle.load(f)
        
       
        tgt_wav_len = data["tgt_wav_len"]
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
        
        
        
    
        

    













