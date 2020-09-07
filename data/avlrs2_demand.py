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



    def __init__(self,noise_pickle_paths,frame_num):

        self.data_paths = np.loadtxt(noise_pickle_paths,str)
        self.frame_num = frame_num
        
                
    def __getitem__(self, index):
        data_item = self.data_paths[index]
    
        with open(data_item, 'rb') as f:
            data = pickle.load(f)
        
       
        tgt_wav_len = data["tgt_wav_len"]
         
        frame_num = self.frame_num
        k = np.random.randint(low=0, high = data["audio_data_Real"][0].shape[1]-frame_num)
        data["audio_data_Real"][0] = data["audio_data_Real"][0][:,k:k+frame_num]
        data["audio_data_Imagine"][0] = data["audio_data_Imagine"][0][:,k:k+frame_num]
        data["audio_data_Real"][1] = data["audio_data_Real"][1][:,k:k+frame_num]
        data["audio_data_Imagine"][1] = data["audio_data_Imagine"][1][:,k:k+frame_num]

        
        data["audio_wav"][0] = data["audio_wav"][0][:,k*256:k*256+(frame_num*256)-1]
        data["audio_wav"][1] = data["audio_wav"][1][:,k*256:k*256+(frame_num*256)-1]
        
        return data
        
    

    def __len__(self):
        return len(self.data_paths)
    

