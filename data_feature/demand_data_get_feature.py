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



class AV_Lrs2Dataset_make_feature(FairseqDataset):

    def __init__(self, target_paths, noise_paths):

        self.tgt_paths = np.loadtxt(target_paths,str)
        self.noi_paths = np.loadtxt(noise_paths,str)
        
    def __getitem__(self, index):
    
        window=torch.hann_window(window_length=1024, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        tgt_item = self.tgt_paths[index] if self.tgt_paths is not None else None
        tgt_wav,_ = torchaudio.load(tgt_item)
        
        noi_item = self.noi_paths[index] if self.noi_paths is not None else None
        noi_wav,_ = torchaudio.load(noi_item)
        
        tgt_wav_len = tgt_wav.shape[1]
        
        spec_tgt = torchaudio.functional.spectrogram(waveform=tgt_wav, pad=0, window=window, n_fft=1024, hop_length=256, win_length=1024, power=None, normalized=False)
        spec_noi = torchaudio.functional.spectrogram(waveform=noi_wav, pad=0, window=window, n_fft=1024, hop_length=256, win_length=1024, power=None, normalized=False)
        tgt_wav_real = spec_tgt[0,:,:,0]
        tgt_wav_imag = spec_tgt[0,:,:,1]
        input_wav_real = spec_noi[0,:,:,0]
        input_wav_imag = spec_noi[0,:,:,1]
        num = index

        batch_dict = {"id": index,"tgt_wav_len":tgt_wav_len, "audio_wav" : [noi_wav, tgt_wav],"audio_data_Real":[input_wav_real,tgt_wav_real], "audio_data_Imagine":[input_wav_imag,tgt_wav_imag]}
        with open('/home/nas/DB/[DB]_voice_corpus/train/noise_000db/'+str(num)+'.pkl', 'wb') as f:
            pickle.dump(batch_dict, f)
        return index


    def __len__(self):
        return len(self.aud_paths)

if __name__ == '__main__':

    clean_train="/home/nas/DB/[DB]_voice_corpus/train/clean_voice_train_path.txt"
    noise_train="/home/nas/DB/[DB]_voice_corpus/train/noise_voice_train_path.txt"
    train_dataset = AV_Lrs2Dataset_make_feature(clean_train,noise_train)
    for i in range(len(train_dataset)):
        print(i)

    

    













