import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import torchaudio
import os
import math
import random
import sys
from fairseq import utils
from DCUnet_jsdr_demand import *
from tensorboardX import SummaryWriter
from dataset.demand_dataset import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--exp_day', type=str, required=True)
    parser.add_argument('--snr', type=str, required=True)
    parser.add_argument('--train_data_root_folder', type=str,required=True)
    parser.add_argument('--val_data_root_folder', type=str,required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float,required=True)
    parser.add_argument('--frame_num', type=int,required=True)
    parser.add_argument('--fs',type=int,required=True)
    parser.add_argument('--modelsave_path', type=str, required=True)

    args = parser.parse_args()
    return args
    
def complex_demand_audio(complex_ri,window,length,fs):
    window = window
    length = length
    complex_ri = complex_ri
    fs=fs
    audio = torchaudio.functional.istft(stft_matrix = complex_ri, n_fft=int(1024*fs), hop_length=int(256*fs), win_length=int(1024*fs), window=window, center=True, pad_mode='reflect', normalized=False, onesided=True, length=length)
    
    return audio
def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.pkl':
                li.append(os.path.join(os.path.join(os.path.abspath(d_name),paths), filename))
    len_li = len(li)            
    return li
if __name__ == '__main__':


    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_day = args.exp_day
    SNR = args.snr
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    frame_num = args.frame_num
    fs = args.fs/16 
    start_epoch = 0
    num_epochs = 50
    
    
    audio_maxlen = int(frame_num*256*fs-1) 
    window=torch.hann_window(window_length=int(1024*fs), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    
    

    best_loss = 10
    data_train = args.train_data_root_folder
    data_train_list=[]
    data_train_list=search(data_train,data_train_list)
    
    data_val = args.val_data_root_folder
    data_val_list=[]
    data_val_list=search(data_val,data_val_list)
    modelsave_path = args.modelsave_path
    
    train_dataset = AV_Lrs2_pickleDataset(data_train_list,frame_num,fs)
    val_dataset = AV_Lrs2_pickleDataset(data_val_list,frame_num,fs)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    model = UNet().to(device)
    
    criterion=wSDRLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,min_lr=0.000005)
    for epoch in range(start_epoch,num_epochs):
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            
            audio_real = batch_data["audio_data_Real"][0].to(device)
            audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
            target_audio = batch_data["audio_wav"][1].squeeze(1).to(device)
            input_audio = batch_data["audio_wav"][0].squeeze(1).to(device)
            
            
            enhance_r, enhance_i = model(audio_real,audio_imagine)
            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,fs)

            
            loss = criterion(input_audio,target_audio,audio_me_pe,eps=1e-8).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()
        train_loss = train_loss/len(train_loader)
        print("train_loss")
        print(train_loss)
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pth')
        model.eval()
        with torch.no_grad():
            val_loss =0.
            for j, (batch_data) in enumerate(val_loader):
                

                audio_real = batch_data["audio_data_Real"][0].to(device)
                audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
                target_audio = batch_data["audio_wav"][1].squeeze(1).to(device)
                input_audio = batch_data["audio_wav"][0].squeeze(1).to(device)
            
            
                enhance_r, enhance_i = model(audio_real,audio_imagine)
                
                enhance_r = enhance_r.unsqueeze(3)
                enhance_i = enhance_i.unsqueeze(3)
                enhance_spec = torch.cat((enhance_r,enhance_i),3)
                audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,fs).to(device)
                
                
                loss = criterion(input_audio,target_audio,audio_me_pe,eps=1e-8).to(device)
                print('valEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()
            val_loss = val_loss/len(val_loader)
            print("val_loss")
            print(val_loss)
            scheduler.step(val_loss)
            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pth')
                best_loss = val_loss
               
