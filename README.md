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
이건 본인 취향 차이인데 어자피 모델 학습할 때 STFT 변환할 꺼 그냥 변환한거 저장해서 쓰자.  
data_feature/demand_data_get_feature.py 들어가서 58,59 line clean_train, noise_train 에다가 위에서 작성한 메모장 파일 주소 넣는다.  
적절하게 noise_train , clean_train, noise_test, clean_test를 각각의 폴더에 pkl file을 저장한다.  
한번더 말하지만 본인 취향이고 이렇게 안 할 사람은 dataset을 본인 나름대로 짜면 된다.

```single line``` dsasdad

```python
def a()


```
