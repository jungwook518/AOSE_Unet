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
import librosa
from fairseq import utils
from DCUnet_jsdr_demand import *

from tensorboardX import SummaryWriter
from dataset.demand_dataset_test import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_day', type=str, required=True)
    parser.add_argument('--snr', type=str, required=True)
    parser.add_argument('--fs',type=int,requuired=True)
    parser.add_argument('--test_model', type=str,required=True)
    args = parser.parse_args()
    return args

def complex_demand_audio(complex_ri,window,length,fs):
    window = window
    length = length
    complex_ri = complex_ri
    fs=fs
    audio = torchaudio.functional.istft(stft_matrix = complex_ri, n_fft=1024*fs, hop_length=256*fs, win_length=1024*fs, window=window, center=True, pad_mode='reflect', normalized=False, onesided=True, length=length)
    
    return audio

if __name__ == '__main__':
    

    
    args = get_args()
    device = torch.device('cpu')
    exp_day = args.exp_day
    SNR = args.snr
    fs = args.fs/16 #16,32,48
    batch_size = 1
    
    window=torch.hann_window(window_length=1024*fs, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    test_model = args.test_model
    num_epochs = 1

    data_test = "data_txt/sample_test.txt"
    test_dataset = AV_Lrs2_pickleDataset(data_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False,num_workers=8)
    
    model_test = UNet().to(device)
    model_test.load_state_dict(torch.load(test_model,map_location=device))
    model_test.eval()
    
    
    with torch.no_grad():
        for i, (batch_data) in enumerate(test_loader):
            print(i)
            
            audio_real = batch_data["audio_data_Real"][0].to(device)
            audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
            target_audio = batch_data["audio_wav"][1].squeeze(1).to(device)
            audio_maxlen = audio_real.shape[-1]*256*fs-1
            
            enhance_r, enhance_i = model_test(audio_real,audio_imagine)
            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = complex_demand_audio(enhance_spec,window,length=audio_maxlen)
            
            

            input_audio = batch_data["audio_wav"][0]
            target_audio = batch_data["audio_wav"][1]
            
            tgt_wav_len = batch_data["tgt_wav_len"][0]
            input_audio = tensor2audio(input_audio,window,length=tgt_wav_len)
            target_audio = tensor2audio(target_audio,window,length=tgt_wav_len)

            
            
            audiosave_path = "audio_output/DCUnet_sample_test_"+str(exp_day)+'_'+str(SNR)+"db
            if not os.path.exists(audiosave_path):
                os.makedirs(audiosave_path)
            
            tgt_wav_len = batch_data["tgt_wav_len"][0]
            torchaudio.save(audiosave_path+"/enhance_"+str(i)+".wav", src=torch.from_numpy(audio_me_pe[:tgt_wav_len]).unsqueeze(0), sample_rate=16000)





