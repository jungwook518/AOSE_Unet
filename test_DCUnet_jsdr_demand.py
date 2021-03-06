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
import librosa
from tensorboardX import SummaryWriter
from dataset.demand_dataset_test_librosa import *
from scipy.io import savemat
import scipy.io.wavfile
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs',type=float,required=True)
    parser.add_argument('--test_model', type=str,required=True)
    parser.add_argument('--test_data_root_folder', type=str,required=True)
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
    #audio = audio.numpy().squeeze()
    return audio

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
    
def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                if filename[:1]=='.':
                    continue
                else:
                    li.append(os.path.join(os.path.abspath(d_name), filename))
    len_li = len(li)            
    return li
    
if __name__ == '__main__':
    

    
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    fs = args.fs/16 #16,32,48
    orig_fs = args.fs/16
    batch_size = 1
    target_fs =[1,2,3] #16,32,48
    
    if fs!=1 and fs!=2 and fs!=3:
        re_fs = find_nearest(target_fs,fs)
        win_len = 1024*re_fs
        window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    else:
        re_fs = fs
        win_len = 1024*fs
        window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    test_model = args.test_model
    num_epochs = 1

    data_test = args.test_data_root_folder
    data_test_list=[]
    data_test_list=search(data_test,data_test_list)
    test_data_output_path = args.test_data_output_path
    test_dataset = AV_Lrs2_pickleDataset(data_test_list,re_fs,orig_fs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, collate_fn=lambda x:my_collate(x),shuffle=False,num_workers=8)
    model_test = UNet().to(device)
    model_test.load_state_dict(torch.load(test_model,map_location=device))
    model_test.eval()
    
    
    with torch.no_grad():
        for i, (data_name,data_wav_len,re_fs,data_wav,input_wav_real,input_wav_imag) in enumerate(test_loader):
            print(i)
            audio_real = input_wav_real.to(device)
            audio_imagine = input_wav_imag.to(device)
            audio_maxlen = int(audio_real.shape[-1]*256*fs-1)
            
            enhance_r, enhance_i = model_test(audio_real,audio_imagine)
            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,re_fs)
            
       
            audiosave_path = test_data_output_path
            if not os.path.exists(audiosave_path):
                os.makedirs(audiosave_path)
            
            data_name = data_name[0]
            re_sr = re_fs
            audio_me_pe=audio_me_pe.to('cpu')

            #n=audio_me_pe.numpy()
            #nn={"rmask":n}
            #savemat(audiosave_path+"/"+data_name+".mat", nn)
            max_data_wav = data_wav.max()
            min_data_wav = data_wav.min()
            if abs(max_data_wav) >= abs(min_data_wav):
                norm_data = abs(min_data_wav)
            else:
                norm_data = abs(max_data_wav)
            print(max_data_wav)
            print(min_data_wav)
            if audio_me_pe.max() >=1 or audio_me_pe.min() <=-1:
                max_aud = audio_me_pe.max()
                min_aud = audio_me_pe.min()
                if abs(max_aud) >= abs(min_aud):
                    audio_me_pe = audio_me_pe * (norm_data/max_aud)
                else:
                    audio_me_pe = audio_me_pe * (norm_data/abs(min_aud))
            
            #    audio_me_pe = audio_me_pe*0.8
            #print(audio_me_pe.dtype)
            torchaudio.save(audiosave_path+"/"+data_name+".wav",src=audio_me_pe[:,:int(data_wav_len)], sample_rate=int(16000*re_sr))
            



