# AOSE_Unet(Audio Only Speech Enhancement using Unet)  

## 0. Prepare dataset 관련(본인 취향의 Dataset을 만들거면 필자처럼 안해도 됨)
### 1) download
https://datashare.is.ed.ac.uk/handle/10283/2791  
위 주소에서 train / test dataset 28spk download 한다.  
파일들 이름들을 메모장에 저장한다.  
ex)p224.wav  
p225.wav  
...
본인 취향인데 나는 feature를 저장하려고 이렇게 했다.  
이렇게 해도 되고 안해도 되고 안 할 사람은 dataset을 본인 나름대로 짜면 된다.  

### 2) audio data를 STFT 하여 저장한다.  
48khz 데이터들을 16khz로 resampling해준다. 필자는 16khz로 사용했다.
```data_feature/demand_data_get_feature.py``` 들어가서 58,59 line clean_train, noise_train 에다가 위에서 작성한 메모장 파일 주소 넣는다.  
적절하게 noise_train , clean_train, noise_test, clean_test를 각각의 폴더에 pkl file을 저장한다.  
한번더 말하지만 본인 취향이고 이렇게 안 할 사람은 dataset을 본인 나름대로 짜면 된다.  
이건 본인 취향 차이인데 어자피 모델 학습할 때 STFT 변환할 꺼 그냥 변환한거 저장해서 쓰자.  

### 3) 저장한 feature를 각각 메모장에 이름들을 1)번처럼 저장하자.  
이것도 본인 취향 차이다.  

## 1. train 관련  
```python train_DCUnet_jsdr_demand.py --gpu 0 --snr 0 --opt 3 --exp_day 0907 --num_noise 1 --batch_size 20 --frame_num 128 --learning_rate 0.0001```  
학습이 될텐데 train_DCUnet_jsdr_demand.py file안에서 model_save_path, tensorboard_path, train_val_data_path,  
... 등등 본인 입맛대로 설정한다. 여기서 snr opt exp_day num_noise는 각 데이터들의 주소를 메모장 파일에 분류해서 저장한 것이다.  
이것도 본인 입맛대로 설정한다.

## 2. model 관련  
학습할 때 필자는 frame_num=128로 설정하였는데 여기서 frame_num이란 STFT 된 audio 신호에 대해 시간축으로 몇개의 sample을 볼 것인지에 대한 말이다.  
model이 Unet 구조여서 encoder가 진행될때마다 적절한 타이밍에 input data가 2배씩 줄 것이고, decoder가 진행될때마다 마찬가지로 적절한 타이밍에 2배씩 늘 것이다.  
따라서 frame_num 을 2의 power로 하는 것을 추천한다. 그게 싫다면 model 만드는 부분에서 padding이나 stride 등 알아서 바꿔라.  
enhance된 audio 신호를 보고 싶으면 output의 type을 numpy로 바꾼 후에 librosa써서 audio로 저장하면 된다.

## 3. data 추가 관련  
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
