# AOSE_Unet(Audio Only Speech Enhancement using Unet)  

## 0. Prepare dataset 관련(본인 취향의 Dataset을 만들거면 필자처럼 안해도 됨)
### 1) download
1. download and extract the file train / test dataset 28spk https://datashare.is.ed.ac.uk/handle/10283/2791  
2. write address location of files in txt file
```/home/nas/clean/train.txt```
```/home/nas/noise/train.txt```
```/home/nas/clean/test.txt```
```/home/nas/noise/test.txt```

### 2) STFT the audio data and store it as a pickle. 
1. 48khz 데이터들을 16khz로 resampling해준다. 필자는 16khz로 사용했다.
2. ```python data_feature/demand_data_get_feature.py --clean_train /home/nas/clean/train.txt --noise_train /home/nas/noise/train.txt --clean_test /home/nas/clean/test.txt --noise_test /home/nas/noise/test.txt --train_save_path /home/nas/train/save/ --test_save_path /home/nas/test/save --fs 16```
3. or ```sh make_data_feature.sh```



## 1. train 관련  
1. ```python train_DCUnet_jsdr_demand.py --train_data_root_folder /home/nas/train/save/ --val_data_root_folder /home/nas/test/save --gpu 0 --modelsave_path model/save/path --snr 0 --exp_day 0101 --batch_size 20 --frame_num 128 --learning_rate 0.0001 --fs 48```
2. or ```sh train.sh```

## 2. model 관련  
1. ```frame_num``` is time value of STFT.
2. recommend ```frame_num = 128 or 256 like 2's power```
3. if you don't like it, change model padding, stride, kernel etc.

## 3. test 관련  
1. ```python test_DCUnet_jsdr_demand.py --fs 48 --test_model model/path.pth --test_data_root_folder /home/nas/test/audio --test_data_output_path /home/nas/test/output```
2. or ```sh test.sh```
3. ```test_data_root_folder``` is folder that has .wav audio files.
4. The length of the output audio is limited. To solve this, go into```dataset/demand_dataset_test_librosa.py``` and add line 107-113 paragraphs appropriately to the length.

## 4. data 추가 관련  
1. demand noise dataset download https://zenodo.org/record/1227121#.X1Ytv3kzaUk 
2. To do, clean data + demand noise dataset
3. ```cd data_augment python data_aug_demand_dataset.py --clean_train_txt /home/nas/clean/train.txt --noise_txt /home/nas/demand/noise.txt --save_path /home/nas/save/path/ --snr 0 --fs 48```
