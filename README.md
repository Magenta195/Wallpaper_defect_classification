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

💡 사용자는 passerby 웹 및 모바일 환경에 접속하여 **전체 공개수배자 신상정보**를 확인할 수 있습니다.

💡 또한, 공개수배자 정보와 대조하여 바로 제보 및 신고할 수 있도록 **관련 안내 정보**를 제공합니다.

💡 관리자는 passerby db를 통하여 공개수배자 **생성, 조회**, 이미지 및 비디오, 신상정보의 **수정, 삭제**를 수행할 수 있습니다.

&nbsp;

# 🧭 Structure

```bash
🗂️ Backend
├── 📂 app
│   ├── 📄 __init__.py
│   ├── 📂 api
│   │   ├── 📄 __init__.py
│   │   ├── 📄 endpoint.py
│   │   ├── 📂 errors
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 http_errors.py
│   │   └── 📂 routes
│   │       ├── 📄 __init__.py
│   │       ├── 📄 admin.py
│   │       ├── 📄 dl_server.py
│   │       ├── 📄 requester.py
│   │       └── 📄 wanted.py
│   ├── 📂 config
│   │   ├── 📄 __init__.py
│   │   ├── 📄 app.py
│   │   └── 📄 model.py
│   ├── 📂 core
│   │   ├── 📄 __init__.py
│   │   └── 📄 events.py
│   ├── 📂 db
│   │   ├── 📄 __init__.py
│   │   ├── 📄 events.py
│   │   ├── 📂 queries
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 admin.py
│   │   │   └── 📄 wanted.py
│   │   └── 📂 repositories
│   │       ├── 📄 __init__.py
│   │       ├── 📄 admin.py
│   │       ├── 📄 base.py
│   │       ├── 📄 base_class.py
│   │       └── 📄 wanted.py
│   ├── 📂 models
│   │   ├── 📄 __init__.py
│   │   ├── 📂 domain
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 admin.py
│   │   │   ├── 📄 base.py
│   │   │   └── 📄 wanted.py
│   │   └── 📂 schemas
│   │       ├── 📄 __init__.py
│   │       ├── 📄 admin.py
│   │       ├── 📄 base.py
│   │       ├── 📄 errors.py
│   │       └── 📄 wanted.py
│   ├── 📂 resources
│   │   ├── 📄 __init__.py
│   │   └── 📄 strings.py
│   ├── 📂 secure
│   │   ├── 📄 __init__.py
│   │   └── 📄 hash.py
│   ├── 📄 main.py
│   ├── 📄 run.sh
│   └── 📄 run_https.sh
├── 📄 readme.md
└── 📄 requirements.txt
```
&nbsp;

# 📚 API docs

👉 자세한 request 및 response 문법은 백엔드 내 swagger 및 redoc 문서를 참조해 주세요! (추후 문서분리 예정)


|Router|Method|Endpoint|Description|
|---|---|---|---|
| admin | `POST` | `/admin` | 지명수배자 정보 및 이미지 DB에 등록 |
| admin | `PUT` | `/admin/data` | 해당 id의 지명수배자 신상정보 데이터 변경 |
| admin | `PUT` | `/admin/image` | 해당 id의 지명수배자 이미지 변경 및 비디오 재생성 |
| admin | `PUT` | `/admin/video` | 해당 id의 지명수배자 비디오 재생성 |
| dl_server | `POST` | `/dl/video` | 해당 id의 생성된 지명수배자 비디오 경로 DB에 등록 |
| dl_server | `PUT` | `/dl/video` | 해당 id의 재생성된 지명수배자 비디오 경로 DB에 수정 |
| wanted | `GET` | `/wanted` | 보유하고 있는 전체 지명수배자 데이터 및 그 해시값 전송 |
| wanted | `GET` | `/wanted/{id}` | 해당 id의 지명수배자 데이터 전송 |
| wanted | `GET` | `/wanted/check/{data_hash}` | 데이터 해쉬 비교를 통해 프론트엔드 데이터의 유효성 검증 |



&nbsp;

# 📝 Tutorial

본 백엔드 레포지토리는 다음과 같은 환경에서 동작을 확인하였습니다.

+ Python 3.10
+ Alpine linux 도커 인스턴스
+ PostgreSQL 데이터베이스 인스턴스

PostgreSQL 도커 인스턴스가 실행되었다는 전제 하에 설명을 시작하겠습니다.

(PostgreSQL 접속 포트 및 접속 정보 설정은 `app/config/model.py`에서 설정 가능합니다.)

### requirements 설치

requirement를 다음 명령어로 설치해 주세요 (conda 및 venv 가상환경을 추천합니다.)

```bash
pip install -r requirements.txt
```

### http로 실행하기

명령어를 통해 `/app` 디렉토리로 이동한 후, 다음 명령어를 실행해 주세요.

```bash
sh run.sh
```

실행 포트 및 fastapi 기본 설정, swagger docs 사용 설정은 `run.sh`파일을 수정하여 세팅할 수 있습니다.


### https로 실행하기

본 백엔드 프로젝트는 **https 형식** 역시 지원합니다. `app/secure` 디렉토리에 `key.pem`과 `cert.pem` 파일을 생성하거나, openssl 패키지를 먼저 설치하기를 권장합니다.

모든 준비 과정이 끝나면 다음 명령어를 실행해 주세요.

```bash
sh run_https.sh
```
