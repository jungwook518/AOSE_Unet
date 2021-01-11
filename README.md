# AOSE_Unet(Audio Only Speech Enhancement using Unet)  

## 0. Prepare dataset 관련(본인 취향의 Dataset을 만들거면 필자처럼 안해도 됨)
### 1) download
https://datashare.is.ed.ac.uk/handle/10283/2791  
train / test dataset 28spk download
파일 주소 위치 메모장에 저장.  
ex)
/home/nas/A/B/C/p224.wav  
/home/nas/A/B/C/p225.wav  

### 2) audio data를 STFT 하여 저장한다.  
48khz 데이터들을 16khz로 resampling해준다. 필자는 16khz로 사용했다.
```data_feature/demand_data_get_feature.py``` 들어가서 58,59 line clean_train, noise_train 에다가 위에서 작성한 메모장 파일 주소 넣는다.  
적절하게 noise_train , clean_train, noise_test, clean_test를 각각의 폴더에 pkl file을 저장한다.  
한번더 말하지만 본인 취향이고 이렇게 안 할 사람은 dataset을 본인 나름대로 짜면 된다.  
이건 본인 취향 차이인데 어자피 모델 학습할 때 STFT 변환할 꺼 그냥 변환한거 저장해서 쓰자.  

### 3) 저장한 feature를 각각 메모장에 이름들을 1)번처럼 저장하자.  
```data_txt/train_...txt,test_...txt``` 처럼 pickle로 저장한 파일들을 메모장에 저장한다.  
이것도 필자 방식이니 본인 취향대로 하면 된다.
## 1. train 관련  
```python train_DCUnet_jsdr_demand.py --gpu 0 --snr 0 --exp_day 0907 --batch_size 20 --frame_num 128 --learning_rate 0.0001 --fs 16```
sampling rate fs를 추가하였고 16,32,48 (3가지중 하나)로 하면 된다.
학습이 될텐데 train_DCUnet_jsdr_demand.py file안에서 model_save_path, tensorboard_path, train_val_data_path,  
... 등등 본인 입맛대로 설정한다. 

## 2. model 관련  
학습할 때 필자는 frame_num=128로 설정하였는데 여기서 frame_num이란 STFT 된 audio 신호에 대해 시간축으로 몇개의 sample을 볼 것인지에 대한 말이다.  
model이 Unet 구조여서 encoder가 진행될때마다 적절한 타이밍에 input data가 2배씩 줄 것이고, decoder가 진행될때마다 마찬가지로 적절한 타이밍에 2배씩 늘 것이다.  
따라서 frame_num 을 2의 power로 하는 것을 추천한다. 그게 싫다면 model 만드는 부분에서 padding이나 stride 등 알아서 바꾸면된다.  
enhance된 audio 신호를 보고 싶으면 output의 type을 numpy로 바꾼 후에 librosa써서 audio로 저장하면 된다.

## 3. test 관련  
```python test_DCUnet_jsdr_demand.py --exp_day 0918 --snr 0 --fs 48 --test_model model_ckpt/bestmodel.pth```   
sampling rate fs를 추가하였고 16,32,48 (3가지중 하나)로 하면 된다.
test할 때 test audio들의 주소를 저장해 놓은 파일을 ```test_DCUnet_jsdr_demand.py``` line 57 data_test에 올려놓는다.  
test data에 대한 dataset은 train data와 다르게 설정하였고, train과 달리 audio들을 입력받는다. STFT feature를 뽑아놨으면 line 37에서 line 49를 생략하면 된다.  
그리고 audio의 길이를 line 55에서 time_len < 512(약8.192)로 제한을 하였고 더 긴 길이의 audio를 입력할 거면 수정하면 된다.

## 4. data 추가 관련  
train data를 다운 받으면 아마 0db~20db의 학습데이터가 있을 것이다.  
약 10K개 있을 텐데 본인이 원하는 noise라던지, SNR 비가 있을 텐데 만드는 법을 작성하도록 하겠다.
일단 Speech Enhance 관련하여 많이 사용하는 noise가 DEMAND noise이다. 필자도 DEMAND noise를 사용하였다.  
https://zenodo.org/record/1227121#.X1Ytv3kzaUk 여기 들어가면 DEMAND noise를 다운로드 할 수 있다.
DEMAND noise를 다운받아보면 공원소리, 버스소리, 차소리, 지하철소리 등 다양한 자연환경에서의 noise를 녹음하였다.  
위에서 처음으로 다운로드한 데이터도 DEMAND data를 합성한 것이다.  
아무튼 위 데이터를 다운받으면 본인이 원하는 데이터를 처음으로 다운로드한 Clean data에다가 섞어서 만들면 된다. 되게 간단하다.  
```data_augment/data_aug_demand_dataset.py```에서 clean_train에 audio file 주소 저장한 메모장 주소를 넣고, noise 에 DEMAND noise 주소를 적는다.  
그럼 DEMAND noise 중에 랜덤으로 섞일 것이다. 이건 코드가 간단하니 본인이 원하는대로 수정해서 사용하면 된다.  
또한 43번 line에서 k값의 low를 2400000으로 설정했는데 이것도 본인이 원하는대로 수정해서 사용하면 된다.
