import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import torchaudio
import os
import sys
import math
import random
from fairseq import utils
from DCUnet_jsdr_demand import *

from tensorboardX import SummaryWriter
from dataset.demand_dataset_test import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs',type=int,required=True)
    parser.add_argument('--test_model', type=str,required=True)
    parser.add_argument('--test_data_txt', type=str,required=True)
    parser.add_argument('--test_data_output_path', type=str,required=True)
    args = parser.parse_args()
    return args

def tensor2audio(audio,window,length):
    window = window
    length = length
    audio = audio
    audio = audio.numpy().squeeze()
    return audio
    
def complex_demand_audio(complex_ri,window,length,fs):
    window = window
    length = length
    complex_ri = complex_ri
    fs=fs
    audio = torchaudio.functional.istft(stft_matrix = complex_ri, n_fft=int(1024*fs), hop_length=int(256*fs), win_length=int(1024*fs), window=window, center=True, pad_mode='reflect', normalized=False, onesided=True, length=length)
    audio = audio.numpy().squeeze()
    return audio

if __name__ == '__main__':
    

    
    args = get_args()
    device = torch.device('cpu')
    fs = args.fs/16 #16,32,48
    batch_size = 1
    win_len = 1024*fs
    window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    test_model = args.test_model
    num_epochs = 1

    data_test = args.test_data_txt
    test_data_output_path = args.test_data_output_path
    test_dataset = AV_Lrs2_pickleDataset(data_test,fs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False,num_workers=8)
    
    model_test = UNet().to(device)
    model_test.load_state_dict(torch.load(test_model,map_location=device))
    model_test.eval()
    
    
    with torch.no_grad():
        for i, (batch_data) in enumerate(test_loader):
            print(i)
            
            audio_real = batch_data["audio_data_Real"][0].to(device)
            audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
            audio_maxlen = int(audio_real.shape[-1]*256*fs-1)
            
            enhance_r, enhance_i = model_test(audio_real,audio_imagine)
            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,fs)
            
            data_wav_len = batch_data["data_wav_len"][0]
            
            input_audio = batch_data["audio_wav"][0]
            input_audio = tensor2audio(input_audio,window,length=data_wav_len)

            
            audiosave_path = test_data_output_path
            if not os.path.exists(audiosave_path):
                os.makedirs(audiosave_path)
            
            data_name = batch_data["data_name"][0]
            torchaudio.save(audiosave_path+"/"+data_name+".wav", src=torch.from_numpy(audio_me_pe[:data_wav_len]).unsqueeze(0), sample_rate=int(16000*fs))





