# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 다룰 때 참고할 지침을 제공합니다.

코드 작성을 요청 받으면 모호한 부분에 대해서 자체적으로 판단하지 말고, 질문을 하여 답변을 토대로 코드를 작성

## 저장소 개요

이 저장소에는 CLIP 기반 접근법을 활용한 딥러닝 프로젝트들이 포함되어 있습니다:

- **ForensicsAdapter**: 일반화 가능한 얼굴 위조 탐지를 위해 CLIP을 적응 (CVPR 2025)
- **CLIP_Merging**: 딥페이크 탐지를 위한 CLIP 병합 기법 관련 연구 (Cross-dataset 실험 및 Model Merging 기능 포함)
- **mergekit**: 사전 훈련된 언어 모델 병합을 위한 고급 툴킷 (Arcee AI 제작)

ForensicsAdapter와 CLIP_Merging은 동일한 핵심 아키텍처와 의존성을 공유하며, mergekit은 CLIP_Merging에서 구현된 Model Merging 기능의 고도화된 버전을 제공합니다.

## 환경 설정

두 프로젝트 모두 동일한 환경 설정이 필요합니다:

### 요구 사항
- Python 3.9
- PyTorch 1.11+ with CUDA 11.3
- CUDA-compatible GPU

### 설치 방법
각 프로젝트 디렉토리로 이동 후 다음 명령어를 실행합니다:
```bash
cd ForensicsAdapter  # or CLIP_Merging
conda create -n FA python=3.9
conda activate FA
sh install.sh
```

`install.sh` 스크립트는 다음을 포함한 모든 필수 의존성을 설치합니다:
- CUDA 지원이 포함된 PyTorch
- 컴퓨터 비전 라이브러리 (OpenCV, Pillow, albumentations)
- 과학 계산 (numpy, scipy, scikit-learn)
- 딥러닝 유틸리티 (timm, efficientnet-pytorch, segmentation-models-pytorch)

## 개발 명령어

### 학습
```bash
# 기본 설정으로 학습
python train.py

# 사용자 정의 설정으로 학습
python train.py --config_path /path/to/config/train.yaml

# 특정 데이터셋으로 학습
python train.py --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2
```

### 테스트
```bash
# 기본 설정으로 테스트
python test.py

# 사용자 정의 설정으로 테스트 (test.yaml 수정 필요)
python test.py --config_path /path/to/config/test.yaml
```

### 단일 테스트 실행
디버깅이나 빠른 테스트를 위해 설정 파일을 수정할 수 있습니다:
- `config/train.yaml`: 학습 관련 파라미터 수정
- `config/test.yaml`: 테스트 관련 파라미터 수정
- 필요 시 `train_dataset`, `test_dataset` 목록 수정

### 실행 전 확인사항

**CLIP_Merging 프로젝트 실행 시:**
1. 프로젝트 디렉토리로 이동: `cd CLIP_Merging` 또는 `cd ForensicsAdapter`
2. 데이터셋 경로 확인: `/workspace/datasets/bench/` 경로에 데이터셋이 있는지 확인
3. JSON 메타데이터 파일 확인: `/workspace/datasets/bench/dataset_json/` 경로에 필요한 JSON 파일들이 있는지 확인
4. 설정 파일에서 `dataset_root_folder`와 `dataset_json_folder` 경로가 올바른지 확인

**CLIP_Merging 새로운 실험 스크립트들:**
- `train_linear_probing.py`: Linear-probing 실험 (Classification head만 학습)
- `train_full_finetuning.py`: Full fine-tuning 실험 (전체 파라미터 학습)
- `train_two_stage.py`: Two-stage 실험 (Linear-probing → Fine-tuning)
- `test_cross_dataset.py`: Cross-dataset 성능 평가
- `run_experiments.py`: 모든 실험 자동 실행

## 아키텍처 개요

### 프로젝트 구조

**ForensicsAdapter & CLIP_Merging 구조:**
```
├── config/                     # YAML 설정 파일
│   └── experiments/            # CLIP_Merging: 실험별 설정 파일들
├── dataset/                    # 데이터셋 로딩 및 전처리
├── model/                      # 모델 정의 및 아키텍처
│   ├── adapters/               # CLIP용 어댑터 계층
│   ├── clip/                   # CLIP 모델 구현
│   ├── experiment_models.py    # CLIP_Merging: 4가지 실험 모델
│   ├── clip_only_model.py      # CLIP_Merging: CLIP backbone 전용 모델
│   ├── model_merging.py        # CLIP_Merging: Model Soup 유틸리티
│   └── *.py                    # 핵심 모델 구성요소 (ds.py, attn.py, layer.py)
├── trainer/                    # 학습 로직 및 메트릭
├── figures/                    # 아키텍처 다이어그램 및 결과
├── train*.py                   # 다양한 학습 스크립트들
├── test_cross_dataset.py       # CLIP_Merging: Cross-dataset 테스트
├── run_experiments.py          # CLIP_Merging: 실험 자동 실행
├── example_merge.py            # CLIP_Merging: Model Merging 예시
├── MODEL_MERGING.md            # CLIP_Merging: Model Merging 가이드
├── README_EXPERIMENTS.md       # CLIP_Merging: Cross-dataset 실험 가이드
└── install.sh                  # 의존성 설치
```

**mergekit 구조 (고급 모델 병합 툴킷):**
```
├── mergekit/                   # 메인 패키지
│   ├── merge_methods/          # 다양한 병합 알고리즘
│   │   ├── linear.py          # Linear 병합
│   │   ├── slerp.py           # SLERP 병합
│   │   ├── ties.py            # TIES 병합
│   │   ├── dare.py            # DARE 병합
│   │   └── ...                # 기타 고급 병합 방법들
│   ├── scripts/               # 실행 스크립트들
│   ├── config.py              # 설정 관리
│   └── merge.py               # 핵심 병합 로직
├── examples/                   # 병합 설정 예시들
│   ├── linear.yml             # Linear 병합 예시
│   ├── ties.yml               # TIES 병합 예시
│   └── ...
├── docs/                      # 상세 문서
└── README.md                  # 종합 가이드
```

### 핵심 구성 요소

**모델 아키텍처 (`model/`):**
- `ds.py`: CLIP 통합 메인 모델 구현
- `adapters/adapter.py`: CLIP 미세 조정을 위한 어댑터 계층
- `clip/`: 커스텀 수정된 CLIP 전체 구현
- `attn.py`, `layer.py`: 어텐션 메커니즘 및 레이어 유틸리티

**학습 파이프라인 (`trainer/`):**
- `trainer.py`: 분산 학습 지원이 포함된 메인 학습 루프
- `base_trainer.py`: 추상 베이스 트레이너 클래스
- `metrics/`: 평가 메트릭 (AUC, ACC, EER, AP)

**Dataset Management (`dataset/`):**
- 다수의 딥페이크 데이터셋을 위한 추상 데이터셋 클래스
- 지원 데이터셋: FF++, DFDC, DFDCP, DFD, Celeb-DF 등
- 설정 가능한 데이터 증강 및 전처리

### 설정 시스템

이 프로젝트는 YAML 기반 설정을 사용하며 주요 섹션은 다음과 같습니다:

**모델 설정:**
- `clip_model_name`: CLIP 백본 ("ViT-L/14", "ViT-B/16")
- `model_name`: 모델 변형 (일반적으로 "ds")
- `mlp_dim`, `mlp_out_dim`: 어댑터 차원
- `head_num`: 어텐션 헤드 수

**학습 설정:**
- `train_dataset`, `test_dataset`: 데이터셋 선택
- `train_batchSize`, `test_batchSize`: 배치 크기
- `nEpochs`: 학습 에포크
- `optimizer`: 학습률과 가중치 감쇠가 포함된 Adam/SGD

**데이터셋 설정:**
- `dataset_json_folder`: 데이터셋 메타데이터 경로
- `compression`: 비디오 압축 레벨 (c23/c40)
- `frame_num`: 학습/테스트용 비디오당 프레임 수
- `resolution`: 입력 이미지 해상도 (256)

### 다중 데이터셋 지원

코드베이스는 여러 딥페이크 데이터셋에서 동시에 학습 및 평가를 지원합니다:
- FaceForensics++ (FF++) 모든 조작 방법 포함
- DFDC (Deepfake Detection Challenge)
- Celeb-DF (v1 및 v2)
- DFDCP (Deepfake Detection Challenge Preview)
- DeeperForensics-1.0
- UADFV

라벨 매핑은 설정 파일에서 일관된 real=0, fake=1 인코딩으로 정의됩니다.

## 중요 사항

- 두 프로젝트는 동일한 구현으로 보입니다 - 특정 작업에 어떤 것을 사용할지 확인하십시오
- **경로 설정 필수**: 데이터셋이 `/workspace/datasets/bench/` 경로에 위치해야 합니다
- **JSON 메타데이터 필요**: 각 데이터셋에 대응하는 JSON 파일이 `dataset_json/` 폴더에 있어야 합니다
- 설정 파일에 `dataset_root_folder`와 `dataset_json_folder` 경로가 올바르게 설정되어야 합니다
- 학습 스크립트는 `--ddp` 플래그로 분산 학습을 지원합니다
- 모델은 일반적인 딥페이크 탐지가 아닌 얼굴 위조 탐지용으로 설계되었습니다
- 사전 훈련된 가중치는 README 파일의 Google Drive 링크를 통해 사용할 수 있습니다
- **실행 확인**: `python train.py` 명령어가 프로젝트 루트 디렉토리에서 정상 작동합니다
- **Model Merging 기능**: CLIP_Merging은 자체적으로 Model Soup 기능을 구현하여 제공
  - Linear-probing, Full fine-tuning, Two-stage, Model Soup 4가지 방법 비교 실험
  - Cross-dataset 일반화 성능 검증
  - Uniform Soup 및 Greedy Soup 구현
- **Mergekit 활용**: 고급 모델 병합이 필요한 경우 `/workspace/code/mergekit/` 툴킷 사용 가능
  - 12가지 이상의 고급 병합 알고리즘 지원 (TIES, DARE, SLERP, Task Arithmetic 등)
  - YAML 기반 설정 파일로 복잡한 병합 워크플로우 정의
  - GPU/CPU 효율적인 out-of-core 병합 지원


## 데이터셋 경로 구조

본 프로젝트는 학습 및 테스트를 위해 로컬 또는 서버 내 특정 디렉토리 구조를 가정합니다.

### DeepfakeBench 데이터셋 경로
```
/workspace/datasets/
├── bench/
│   ├── Celeb-DF-v1/
│   ├── Celeb-DF-v2/
│   ├── DFDC/
│   ├── DFDCP/
│   ├── FaceForensics++/
│   │   ├── original_sequences/
│   │   │   ├── actors/
│   │   │   └── youtube/
│   │   │       └── c23/frames/   # 압축된 프레임 데이터
│   │   └── manipulated_sequences/
│   ├── UADFV/
│   ├── DeeperForensics-1.0/
│   ├── dataset_json/             # JSON 메타데이터 파일들
│   │   ├── FaceForensics++.json
│   │   ├── Celeb-DF-v1.json
│   │   ├── Celeb-DF-v2.json
│   │   └── ... (기타 데이터셋 JSON 파일들)
│   └── logs/
│       ├── clip_merging/          # CLIP_Merging 훈련 로그
│       ├── clip_merging_test/     # CLIP_Merging 테스트 로그
│       ├── clip_merging_ablation/ # CLIP_Merging ablation 로그
│       └── forensics_adapter/     # ForensicsAdapter 로그
```

### 설정 파일 경로 변경사항

**CLIP_Merging 프로젝트의 경우:**
- `dataset_json_folder`: `/workspace/datasets/bench/dataset_json`
- `dataset_root_folder`: `/workspace/datasets/bench` (데이터셋 루트 경로)
- `log_dir`: `/workspace/datasets/bench/logs/clip_merging` (훈련용)
- `log_dir`: `/workspace/datasets/bench/logs/clip_merging_test` (테스트용)
- `logdir`: `/workspace/datasets/bench/logs/clip_merging_ablation` (ablation 실험용)

기존의 하드코딩된 경로(`/data/cuixinjie/`)는 주석 처리되어 있으므로 필요시 참조할 수 있습니다.

### 실행 관련 수정사항

**CLIP_Merging 프로젝트 실행 문제 해결:**

1. **기본 설정 경로 수정**: `train.py`의 기본 `config_path`를 상대 경로 `config/train.yaml`로 변경
2. **데이터 증강 오류 수정**: `use_data_augmentation: false` 설정이 적용되도록 `abstract_dataset.py` 수정
3. **경로 변환 로직 추가**: JSON 파일의 Windows 스타일 상대 경로를 Unix 절대 경로로 변환
   - Windows 백슬래시(`\`) → Unix 슬래시(`/`) 변환
   - `dataset_root_folder` 설정을 이용한 절대 경로 생성

## Model-soups 참조 프로젝트

`model-soups` 프로젝트는 CLIP_Merging 구현 시 참조할 수 있는 model merging 방법론을 제공합니다.

### 프로젝트 개요

**Model-soups**는 여러 개의 fine-tuned 모델 가중치를 평균화하여 단일 모델보다 더 높은 성능을 달성하는 방법론입니다.

**핵심 아이디어:**
- Foundation Model (CLIP ViT-B/32)을 베이스로 사용
- 동일한 모델을 여러 번 다양한 하이퍼파라미터로 fine-tuning
- 여러 모델의 가중치를 평균화하여 "soup" 생성
- 추론 시간 증가 없이 성능 향상 달성

### 주요 구현 방법

#### 1. Uniform Soup (균등 평균)
```python
# 모든 모델에 동일한 가중치 (1/N) 부여
uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
```

#### 2. Greedy Soup (탐욕적 선택)
```python
# 검증 성능 기준으로 모델 정렬 후 성능 향상되는 모델만 추가
greedy_soup_ingredients = [sorted_models[0]]  # 최고 성능 모델부터 시작
```

### 핵심 컴포넌트

#### ModelWrapper 클래스
```python
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False):
        self.model = model  # CLIP visual encoder
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        # Language transformer 제거로 메모리 절약
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
```

**주요 특징:**
- CLIP의 visual encoder만 사용 (language part 제거)
- Linear classification head 추가
- Feature normalization 적용

### 실행 방법

#### 전체 파이프라인
```bash
cd model-soups
python main.py --download-models --eval-individual-models --uniform-soup --greedy-soup --plot \
    --data-location ~/data --model-location ~/models
```

#### 단계별 실행
```bash
# 1. 사전 훈련된 모델 다운로드 (72개 모델)
python main.py --download-models --model-location ~/models

# 2. 개별 모델 성능 평가
python main.py --eval-individual-models --data-location ~/data --model-location ~/models

# 3. Soup 생성 및 평가
python main.py --uniform-soup --greedy-soup --data-location ~/data --model-location ~/models

# 4. 결과 시각화
python main.py --plot
```

### 평가 데이터셋

- **ImageNet**: 기본 테스트셋
- **ImageNet2p**: Held-out validation set (ImageNet train의 2%)
- **Out-of-Distribution 데이터셋**: ImageNet-V2, ImageNet-Sketch, ImageNet-R, ObjectNet, ImageNet-A

### CLIP_Merging 구현 시 참고 사항

**참고 가능한 핵심 코드:**

1. **모델 가중치 평균화 로직** (`main.py:121-123`):
   ```python
   uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}
   ```

2. **모델 래퍼 구조** (`utils.py:6-30`):
   - CLIP visual encoder + classification head 구조
   - Language transformer 제거 패턴

3. **성능 기반 모델 선택** (`main.py:147-185`):
   - Validation 성능 기준 모델 정렬
   - 탐욕적 모델 추가 전략

4. **배치 처리 및 평가** (`utils.py:43-75`):
   - 데이터 로더 배치 처리
   - 모델 평가 함수

**적용 시 고려사항:**
- CLIP_Merging에서는 얼굴 위조 탐지 태스크에 맞게 classification head 수정 필요
- DeepfakeBench 데이터셋 구조에 맞는 데이터 로더 구현 필요
- Forensics-specific한 평가 메트릭 (AUC, EER 등) 통합 필요

## Mergekit 툴킷

`mergekit`은 Arcee AI에서 개발한 사전 훈련된 언어 모델 병합을 위한 고급 툴킷입니다. CLIP_Merging의 Model Soup 기능보다 훨씬 정교한 병합 알고리즘들을 제공합니다.

### 주요 특징

**지원 병합 방법 (12가지 이상):**
- **Linear**: 단순 가중 평균
- **SLERP/Multi-SLERP**: 구면 선형 보간
- **Task Arithmetic**: 태스크 벡터 기반 병합
- **TIES**: 태스크 간섭 해결 + 희소화
- **DARE**: 무작위 가지치기 + 재스케일링
- **DELLA**: 적응적 크기 기반 가지치기
- **SCE**: 분산 기반 적응적 가중치
- **Arcee Fusion**: 동적 임계값 기반 융합

**핵심 기능:**
- Out-of-core 메모리 효율적 병합 (8GB VRAM으로도 대형 모델 병합 가능)
- YAML 기반 설정으로 복잡한 다단계 병합 워크플로우 정의
- GPU/CPU 하이브리드 실행 지원
- HuggingFace Hub 직접 업로드 지원

### 기본 사용법

#### 1. 설치
```bash
cd /workspace/code/mergekit
pip install -e .
```

#### 2. 기본 병합 실행
```bash
# YAML 설정 파일로 병합 실행
mergekit-yaml examples/linear.yml ./output-model-directory --cuda

# 다중 단계 병합
mergekit-multi complex_merge_config.yml ./output-directory
```

#### 3. 설정 파일 예시
```yaml
merge_method: linear
models:
  - model: model_1_path
    parameters:
      weight: 1.0
  - model: model_2_path
    parameters:
      weight: 0.5
dtype: float16
```

### CLIP_Merging과의 차이점

| 특성 | CLIP_Merging | mergekit |
|------|--------------|----------|
| 병합 방법 | Uniform/Greedy Soup (2가지) | 12가지 이상 고급 알고리즘 |
| 메모리 효율성 | 기본적 | Out-of-core 최적화 |
| 설정 방식 | Python 코드 | YAML 선언적 설정 |
| 모델 지원 | CLIP 특화 | 범용 Transformer 모델 |
| 평가 메트릭 | DeepFake 특화 (AUC, EER) | 일반 분류 메트릭 |

### 활용 시나리오

**CLIP_Merging용 Model Merging을 위해 mergekit 활용:**
1. 기본 Model Soup는 CLIP_Merging 내장 기능 사용
2. 고급 병합 실험 (TIES, DARE 등)이 필요한 경우 mergekit 활용
3. 복잡한 다단계 병합 워크플로우가 필요한 경우 mergekit 활용

**설정 변환 예시:**
```bash
# CLIP_Merging Model Soup → mergekit TIES 병합으로 변환
# 1. CLIP_Merging으로 여러 체크포인트 생성
# 2. mergekit YAML 설정으로 TIES 병합 실행
```

