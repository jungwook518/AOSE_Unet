import os
import numpy as np
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
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_train', type=str, required=True)
    parser.add_argument('--noise_train', type=str, required=True)
    parser.add_argument('--clean_test', type=str, required=True)
    parser.add_argument('--noise_test', type=str, required=True)
    parser.add_argument('--train_save_path', type=str, required=True)
    parser.add_argument('--test_save_path', type=str, required=True)
    parser.add_argument('--fs', type=int, required=True)
    args = parser.parse_args()
    return args

class AV_Lrs2Dataset_make_feature(Dataset):

    def __init__(self, target_paths, noise_paths,save_path,fs):

        self.tgt_paths = np.loadtxt(target_paths,str)
        self.noi_paths = np.loadtxt(noise_paths,str)
        self.save_path = save_path
        self.fs = fs
    def __getitem__(self, index):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        win_len = int(1024*(fs/16))
        window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        tgt_item = self.tgt_paths[index] if self.tgt_paths is not None else None
        tgt_wav,_ = torchaudio.load(tgt_item)
        
        noi_item = self.noi_paths[index] if self.noi_paths is not None else None
        noi_wav,_ = torchaudio.load(noi_item)
        
        tgt_wav_len = tgt_wav.shape[1]
        
        spec_tgt = torchaudio.functional.spectrogram(waveform=tgt_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
        spec_noi = torchaudio.functional.spectrogram(waveform=noi_wav, pad=0, window=window, n_fft=win_len, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)
        tgt_wav_real = spec_tgt[0,:,:,0]
        tgt_wav_imag = spec_tgt[0,:,:,1]
        input_wav_real = spec_noi[0,:,:,0]
        input_wav_imag = spec_noi[0,:,:,1]
        num = index

        batch_dict = {"id": index,"tgt_wav_len":tgt_wav_len, "audio_wav" : [noi_wav, tgt_wav],"audio_data_Real":[input_wav_real,tgt_wav_real], "audio_data_Imagine":[input_wav_imag,tgt_wav_imag]}
        
        with open(self.save_path+'/'+str(num)+'.pkl', 'wb') as f:
            pickle.dump(batch_dict, f)
        
        return index


    def __len__(self):
        return len(self.tgt_paths)

if __name__ == '__main__':
    
    args = get_args()
    #clean_train="/home/nas/DB/[DB]_voice_corpus/train/clean_voice_train_path.txt"
    #noise_train="/home/nas/DB/[DB]_voice_corpus/train/noise_voice_train_path.txt"
    clean_train = args.clean_train
    noise_train = args.noise_train
    
    clean_test = args.clean_test
    noise_test = args.noise_test
    
    train_save_path = args.train_save_path
    test_save_path = args.test_save_path
    
    fs = args.fs
    
    train_dataset = AV_Lrs2Dataset_make_feature(clean_train,noise_train,train_save_path,fs) #fs 16, 32, 48
    for i in range(len(train_dataset)):
        print(train_dataset[i])
    
    test_dataset = AV_Lrs2Dataset_make_feature(clean_test,noise_test,test_save_path,fs) #fs 16, 32, 48
    for i in range(len(test_dataset)):
        print(test_dataset[i])

    

    













