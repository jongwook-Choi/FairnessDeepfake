# Preserving Fairness and Generalization in Deepfake Detection using CLIP - 프로젝트 가이드
이 파일은 얼굴 이미지 대상의 Deepfake Detection (=face forgery detection) 구현에 대한 프로젝트 가이드
Deepfake Detection 모델의 일반화 성능은 유지하며, 공정한 모델을 구현하는 것을 목표로 한다

딥페이크 탐지, 모델 일반화, 모델 공정성 연구의 전문가의 관점에서 방안 제시

### 용어 정의 
- **일반화 성능**:  FF++ 데이터셋 학습 CelebDF, DFD, DFDC 등 다른 데이터셋에서 평가(AUC)하여 측정
- **공정한 모델**: 인종과 성별과 같은 특정 Subgroup에 측정된 공정성 지표(F_FPR, F_OAE, F_DP, F_MEO)로 측정
- **Subgroup 정의** 
   - 8개 subgroup: (gender × race) = 2 × 4
   - subgroup_id = gender * 4 + race
   - Race: Asian(0), Black(1), White(2), Other(3)
   - Gender: Male(0), Female(1)

## Overall Framework 설명
CLIP의 pre-train weight 존재하는 bias가 downstream task 에도 영향을 끼친다는 보고가 있음
이에, 본 모델은 공정한 모델 구현을 위해 Global, Local 하게 fair한 특징을 추출 가능하도록 학습을 진행할 예정

### Stage 1
Stage1 에서는 fairface, UTKFace, CasualFace 데이터셋을 활용하여, CLIP pre-train weight의 Global bias 제거
학습 : fairface, UTKFace, CasualFace 
평가 : fairness

fairface 데이터셋 경로 : /workspace/datasets/fairness
UTKFace 데이터셋 경로 : /workspace/datasets/UTKFace
CasualFace 데이터셋 경로 : /workspace/datasets/CausalFace
fairness 데이터셋 경로 : /workspace/datasets/fairness

### Stage 2
Stage2 에서는 fairness 데이터셋을 활용하여, Local bias 제거와 함께 Deepfake Detection 학습 
**아직 Local bias 제거의 방안은 미정**


## 개발 규칙
- **질문하기**: 모호한 부분에 대해 최대한 많은 질문을 한 후 작업
- **절대 모킹하지 않기**: 실제 동작하는 코드만 작성
- **Fallback 금지**: 필수 패키지 미설치 시 대체 로직으로 자동 실행하지 않고, 에러 발생시켜 사용자에게 알림
- **의존성 검증**: 코드 실행 전 필수 패키지 설치 여부 확인

### 필수 패키지
- `geomloss`: Sinkhorn distance 계산에 필수

