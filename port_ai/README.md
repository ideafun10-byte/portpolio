# port_ai
노이즈나 변형이 일어난 항만 입출항 정보 열을 주어진 영문 선박 목록과 매핑하는 시스템

## 문제 소개
항만 문제해결을 위한 AI 기술 공모전(https://dacon.io/competitions/official/236771/overview/description)
- 입력:학습 입력 및 정답 데이터 표,선박 제원정보,예측 후보 목록 표
- 출력:예측 ID와 정답 ID가 매핑된 2열짜리 표     
- 평가 방식:0.7 × (한글 선박명) 정확도 + 0.3 × (비한글 선박명) 정확도

## 결과
검증 점수

## 접근 방법
- 전체 흐름 (그림이나 화살표 한 줄)
- 단계별 설명, 특히 직접 고안한 부분

## 시도했지만 잘 안 된 것 / 배운 점

## 폴더 구조
port-ai/
├── .gitignore                  
├── README.md                   
├── open/                       ← 대회 원본 데이터 (읽기 전용, 깃허브 X)
│   └── vessel_catalog.csv 등의 데이터
├── notebooks/                  ← 탐색·실험용 노트북 (BASE 절대 경로를 한 번만 지정)
├── src/                        ← 재사용하는 코드 모듈
│   ├── __init__.py
│   ├── kophon (?)              ← 한글 → 로마자 소리 set
│   ├── engphon (?)             ← 영어 → 로마자 소리 set
│   └── train.py (?)            ← argparse·시드·outputs/vN·config.json 뼈대
├── submit/                     ← 제출용 (zip으로 묶는 폴더)
│   ├── script.py               ← 베이스라인 경로 방식 그대로
│   ├── requirements.txt        ← script.py가 import하는 것만
│   └── model/                  ← 학습된 모델 (깃허브 X)
├── outputs/                    ← 실행 결과 (깃허브 X)
│   ├── v1/
│   └── v2/ ...
├── logs/                       ← 학습 로그 (깃허브 X)
└── env/                        ← 환경 "설명서" (가상환경 자체가 아님)
    └── requirements-py311.txt  ← 채점 서버 버전 고정


## 실행 방법
- 환경 (파이썬 버전, requirements)
- 데이터 받는 법
- 실행 명령어

## 앞으로 개선할 점