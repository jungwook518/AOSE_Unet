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
import librosa
import math

def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                li.append(os.path.join(os.path.join(os.path.abspath(d_name),paths), filename))
    len_li = len(li)            
    return li, len_li

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def my_collate(batch):
    
    data_name,data_wav_len, re_fs,data_wav,input_wav_real, input_wav_imag = list(zip(*batch))
    #print(data_wav_len)
    #maxlen = max([len(x) for x in data])
    #zero_arr = np.zeros([B, maxlen])
    #for i in range(B):
    #    zero_arr[i] = np.concatenate((data[i], np.zeros(maxlen - len(data[i]))), axis=0)
    #zero_arr = torch.FloatTensor(zero_arr)
    #label = torch.LongTensor(label)
    
    data_wav_len = torch.FloatTensor(data_wav_len)
    re_fs = torch.FloatTensor(re_fs)
    data_wav = torch.FloatTensor(data_wav)
    input_wav_real = torch.FloatTensor(input_wav_real)
    input_wav_imag = torch.FloatTensor(input_wav_imag)
    
    return (data_name,data_wav_len, re_fs,data_wav,input_wav_real, input_wav_imag)
  
class AV_Lrs2_pickleDataset(Dataset):



    def __init__(self,data_test_list,re_fs,orig_fs):

        self.wav_list = data_test_list
        self.fs = re_fs
        self.orig_fs = orig_fs
                
    def __getitem__(self, index):
        target_fs =[1,2,3] #16,32,48
        data_item = self.wav_list[index] if self.wav_list is not None else None
        data_name = os.path.splitext(os.path.split(data_item)[1])[0]
        win_len = int(1024*(self.fs))
        hop_len = int(win_len/4)
        if self.orig_fs !=1 and self.orig_fs!=2 and self.orig_fs!=3:
            re_fs = find_nearest(target_fs,self.orig_fs)
            
            data_wav,_ = librosa.load(self.wav_list[index],sr=int(re_fs*16000))

        else:
            re_fs = self.orig_fs
            data_wav,_ = librosa.load(self.wav_list[index],sr=int(re_fs*16000))
        
        data_wav_len = data_wav.shape[0]
        spec_noi = librosa.core.stft(data_wav,n_fft=win_len,hop_length=hop_len,win_length=win_len,center=True)
        input_wav_real =np.real(spec_noi)
        input_wav_imag = np.imag(spec_noi)
        
        
        #data = {"data_name": [data_name],"data_wav_len":[data_wav_len], "sr":[re_fs],"audio_wav" : [data_wav],"audio_data_Real":[input_wav_real], "audio_data_Imagine":[input_wav_imag]}

        
        time_len = input_wav_real.shape[1]
        
        
        if time_len < 512:
            empty_in_r = np.zeros([int(win_len/2+1),512])
            empty_in_r[:,:time_len]=input_wav_real
            input_wav_real = empty_in_r 
            empty_in_i = np.zeros([int(win_len/2+1),512])
            empty_in_i[:,:time_len]=input_wav_imag
            input_wav_imag = empty_in_i
            
        elif 512 <= time_len < 1024:
            empty_in_r = np.zeros([int(win_len/2+1),1024])
            empty_in_r[:,:time_len]=input_wav_real
            input_wav_real = empty_in_r 
            empty_in_i = np.zeros([int(win_len/2+1),1024])
            empty_in_i[:,:time_len]=input_wav_imag
            input_wav_imag = empty_in_i
        
        else:
            input_wav_real = input_wav_real[:,:1024] 
            input_wav_imag =  input_wav_imag[:,:1024] 
        
        return (data_name,data_wav_len,re_fs,data_wav,input_wav_real,input_wav_imag)
        
    

    def __len__(self):
        return len(self.wav_list)
    





if __name__ == '__main__':
    data_path = "/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Magnitude_subnetwork/demand_test_sort_noise_0db_num1.txt"
    train_dataset = AV_Lrs2_pickleDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,pin_memory = True, num_workers=8)
    for i, (batch_data) in enumerate(train_loader):
        audio = batch_data["tgt_wav_len"]
        print(batch_data)
        
        
        
    
        

    













