import argparse
import os
import numpy as np
from fairseq.data import FairseqDataset
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

import numpy as np
import torch
import time
import torchaudio


def compare_noise(noise_wav_list):
    noise_len_list = []

    for i in range(len(noise_wav_list)):
        noise_len_list.append(noise_wav_list[i].shape[1])
    noise_len_list.sort()
    cut_f = noise_len_list[0]
    for i in range(len(noise_wav_list)):
        start = (noise_wav_list[i].shape[1]-cut_f)//2
        noise_wav_list[i]=noise_wav_list[i][:,start:start+cut_f]

    mixing_noise = 0
    for i in range(len(noise_wav_list)):
        mixing_noise += noise_wav_list[i]
    return mixing_noise
    
    
    
def compare_src_noise(src, noisy):
    src_len = src.shape[1]
    noisy_len = noisy.shape[1]
    k = np.random.randint(low=0, high = noisy_len - src_len)
    np_src = src.squeeze(0)
    np_noisy = noisy.squeeze(0)
    if src_len > noisy_len :
        p = src_len//noisy_len
        np_noisy = torch.cat((np_noisy.repeat(p), np_noisy[0:src_len%noisy_len]),dim=-1)

    elif src_len < noisy_len :
        np_noisy = np_noisy[k:k+src_len]
    else :
        np_noisy = np_noisy
        np_src = np_src
        
    return np_src.numpy(), np_noisy.numpy()
    

    
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))
    
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

    
def data_augment_onthefly(src,noisy,SNR,opt):
    snr =SNR
    noise_file_list=noisy # noisy is list [path1, path2, ..]
    noise_wav_list =[]
    clean_wav,_ = torchaudio.load(src)
    option = opt

    if option == 1:
        print("clean")
        mixed_amp = clean_wav
    
    elif option == 2:
        print("gauss")
        gaussian_noise = torch.randn(clean_wav.shape)
        
        clean_amp, noise_amp = compare_src_noise(clean_wav,gaussian_noise)
    
        clean_rms = cal_rms(clean_amp)
        noise_rms = cal_rms(noise_amp)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)
        mixed_amp=np.expand_dims(mixed_amp,axis=0)
        mixed_amp = torch.from_numpy(mixed_amp)


        
    elif option == 3:
        for i in range(len(noise_file_list)):
            noise_wav,_ = torchaudio.load(noise_file_list[i])
            noise_wav_list.append(noise_wav)
            
        
        if len(noise_file_list) > 1:
            print("222")
            mixing_wav = compare_noise(noise_wav_list)
        else :
            mixing_wav = noise_wav
        clean_amp, noise_amp = compare_src_noise(clean_wav,mixing_wav)
    
        clean_rms = cal_rms(clean_amp)
        noise_rms = cal_rms(noise_amp)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)
        mixed_amp=np.expand_dims(mixed_amp,axis=0)
        mixed_amp = torch.from_numpy(mixed_amp)

    return clean_wav, mixed_amp

class AV_Lrs2Dataset(FairseqDataset):

    def __init__(self,noi_paths,tgt_paths,SNR,opt,save_path,fs):

        self.noi_paths = np.loadtxt(noi_paths,str)
        self.tgt_paths = np.loadtxt(tgt_paths,str)
        self.snr = SNR
        self.opt = opt #clean, gaussian, noise 1,2,3
        self.save_path = save_path
        self.fs = fs
                
    def __getitem__(self, index):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.noisy_paths = []
        ############ noisy audio prepare ######################
        
        k = np.random.randint(low=0, high=len(self.noi_paths))
        self.noisy_paths.append(self.noi_paths[k])


        ################### audio prepare #######################
        win_len = int(1024*(fs/16))
        window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        tgt_item = self.tgt_paths[index] if self.tgt_paths is not None else None
        tgt_wav, input_mix_wav = data_augment_onthefly(tgt_item,self.noisy_paths,self.snr,self.opt)
        
        tgt_wav_len = tgt_wav.shape[1]
        spec_tgt = torchaudio.functional.spectrogram(waveform=tgt_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
        spec_input = torchaudio.functional.spectrogram(waveform=input_mix_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
        tgt_wav_real = spec_tgt[0,:,:,0]
        tgt_wav_imag = spec_tgt[0,:,:,1]
        input_wav_real = spec_input[0,:,:,0]
        input_wav_imag = spec_input[0,:,:,1]
        
        num = index

        batch_dict = {"id": index,"tgt_wav_len":tgt_wav_len, "audio_wav" : [input_mix_wav, tgt_wav],"audio_data_Real":[input_wav_real,tgt_wav_real], "audio_data_Imagine":[input_wav_imag,tgt_wav_imag]}
        
        with open(self.save_path+'/'+str(num)+'.pkl', 'wb') as f:
            pickle.dump(batch_dict, f)
        
        return index



    def __len__(self):
        return len(self.tgt_paths)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_train_txt', type=str, required=True)
    parser.add_argument('--noise_txt', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--snr', type=str, required=True)
    parser.add_argument('--fs', type=int, required=True)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    #clean_train="/home/nas/DB/[DB]_voice_corpus/train/clean_voice_train_path.txt"
    #noise="/home/nas/DB/[DB]DEMAND/demand_noise_path.txt"
    args = get_args()
    clean_train = args.clean_train_txt
    noise = args.noise_txt
    save_path = args.save_path
    SNR = args.snr
    fs=args.fs
    opt = 3 # gaussian=2 , demand_noise=3
    train_dataset = AV_Lrs2Dataset(noise,clean_train,SNR,opt,save_path,fs)
    for i in range(len(train_dataset)):
        print(train_dataset[i])

    













