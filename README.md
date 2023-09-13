# 데이콘 도배 하자 유형 분류 AI 경진대회

### <div align="center"><b><i></i></b></div>

&nbsp; 

> [대회링크](https://dacon.io/competitions/official/236082/overview/description)
> 
> 프로젝트 기간 2023.04 ~ 2023.05
> 
> 알고리즘 | 비전 | 분류 | MLOps | Weight F1 Score

&nbsp; 

🔎 AI 기술을 활용하여 이미지 데이터를 기반으로 인테리어 하자를 판단하고 빠르게 대처할 수 있는 분류 모델 개발하기

💾 본 레포지토리는 도배하자 경진대회에서 사용한 모델을 학습하는 코드가 저장되어 있습니다.

&nbsp;

# ⚙️ Tech Stack

<div align="center">
<img src="https://img.shields.io/badge/Python-3776AB0?style=for-the-badge&logo=Python&logoColor=white"><img src="https://img.shields.io/badge/Pytorch-009688?style=for-the-badge&logo=Pytorch&logoColor=white"><img src="https://img.shields.io/badge/Tensorflow-E92063?style=for-the-badge&logo=Tensorflow&logoColor=white"><img src="https://img.shields.io/badge/Keras-4169E1?style=for-the-badge&logo=Keras&logoColor=white"><img src="https://img.shields.io/badge/sklearn-FF9900?style=for-the-badge&logo=scikitlearn&logoColor=white"><img src="https://img.shields.io/badge/pyspark-2496ED?style=for-the-badge&logo=apachespark&logoColor=white">
</div>
&nbsp; 

# ❓ About B2win Team

<div align="center">
  
| [@hwaxrang](https://github.com/hwaxrang) | [@Magenta195](https://github.com/Magenta195) | [@moongni](https://github.com/moongni) | [@heehaaaheeeee](https://github.com/heehaaaheeeee) | [@ShinEunChae](https://github.com/ShinEunChae) | [@joseokjun](https://github.com/joseokjun) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="src/khr.png" width=200 /> | <img src="src/kth.jpeg" width=200 /> | <img src="src/mgh.png" width=200 /> | <img src="src/msh.jpg" width=200 /> | <img src="src/sec.jpeg" width=200 /> | <img src="src/jsj.jpg" width=200 /> |
| `권화랑`   | `김태형` | `문건희` | `문숙희` | `신은채` | `조석준`  |

</div>

&nbsp; 

# 🗝️ Key Service

💡 EDA 결과 총 19개의 클래스를 가진 비대칭 데이터(Unbalanced Data)로 확인  

💡 비대칭 데이터의 분류 모델 학습을 위한 다양한 기법 사용

| 적용한 기법 | 기대 효과 | 
|:---:|:---:|
|CutMix|다수 클래스에 대한 과적합을 방지하며 소수 클래스에 대해서 적절한 결정경계를 가지는 모델 학습|
|Focal Loss|easy class에 대한 손실값을 낮추고 hard class에 대한 손실값을 높여 소수 클래스에 대한 가중치 변경 유도|
|Weighted Random Sampling|매 배치 마다 19개 클래스에 대한 비교적 균등한 샘플링을 진행|

💡 다양한 capacity를 가진 CNN 네트워크의 transfer learning 성능을 비교하여 최적의 모델을 찾고 하이퍼파라미터 튜닝을 통해 Best Model 작성
| 모델 계열 중 Best | Weighted F1 Score|
|:---:|:---:|
|ResNet-152|0.5648913541|
|DenseNet161|0.5681891560|
|InceptionNet-v3|0.5740491055|
|EfficientNet-v2m|0.61043914185|

**Best Model**
- EfficientNet-v2m 모델 3개 배깅 앙상블
- Focal loss (label smooth)
- CutMix 적용
- learning rate: 7e-4
- exponetial scheduler: gamma=0.8

&nbsp;

# 🧭 Structure

```bash
🗂️ data
├── 📄 __init__.py
├── 📄 EDA.ipynb
├── 📄 add_augmented.py
├── 📄 augmentation.py
├── 📄 dset.py
└── 📄 dutils.py
🗂️ models
├── 📄 __init__.py
├── 📄 effinet.py
└── 📄 effinetv2.py
🗂️ trainer
├── 📂 loss
│   ├── 📄 __init__.py
│   ├── 📄 celoss.py
│   └── 📄 focalloss.py
├── 📂 score
│   ├── 📄 __init__.py
│   └── 📄 f1score.py
├── 📄 __init__.py
├── 📄 ensemble.py
├── 📄 metric.py
├── 📄 optimizer.py
├── 📄 scheduler.py
└── 📄 trainer.py
🗂️ utils
├── 📄 __init__.py
├── 📄 config.py
└── 📄 seed.py
📄 README.md
📄 run.ipynb
📄 setup.py
```
&nbsp;

# 📝 Tutorial

분류 모델을 학습하는 파일은 `run.ipynb` 입니다.
학습에 필요한 외부 라이브러리는 `setup.py`에 작성되어 있습니다.

local 환경과 google colab 환경에서 필요한 동작이 다르니 아래의 명령어를 통해 PYTHONPATH를 설정하고 라이브러리를 설치해주시길 바랍니다.

### local 환경

```bash
python setup.py develop
```

### Colab 환경

git 명령어를 통해 본 레포지토리를 clone한 후 본 레포지토리 root 디렉토리로 이동하여 아래의 명령어를 실행해주세요

```bash
python setup.py install
```
