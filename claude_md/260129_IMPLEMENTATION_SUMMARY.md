# Stage 1 + Stage 2 전면 재설계 구현 정리

## 1. 파일 변경 목록

### 1.1 신규 생성 파일 (11개)

| # | 파일 경로 | 핵심 역할 |
|---|----------|----------|
| 1 | `CLIP_stage1/model/gradient_reversal.py` | GRL 모듈 + DANN lambda scheduling |
| 2 | `CLIP_stage1/losses/adversarial_fairness_loss.py` | GRL adversarial + Cosine Sim + Sinkhorn 통합 loss |
| 3 | `CLIP_stage2/model/adain_fusion.py` | AdaptiveInstanceNorm (fair -> detect style 주입) |
| 4 | `CLIP_stage2/model/demographic_cross_attention.py` | 4-view Multi-Head CA + DemographicFusionModule |
| 5 | `CLIP_stage2/model/stage2_model_v2.py` | 재설계 통합 모델 (AdaIN+DemCA+Gate+LDAM Head) |
| 6 | `CLIP_stage2/losses/subgroup_conditional_loss.py` | Bi-level CVaR (prediction-level fairness) |
| 7 | `CLIP_stage2/losses/ldam_loss.py` | LDAMLoss + LDAMFairnessHead (margin = 1/sqrt(sqrt(n))) |
| 8 | `CLIP_stage2/losses/contrastive_loss.py` | Stage1/Stage2 feature disentanglement |
| 9 | `CLIP_stage2/trainer/stage2_trainer_v2.py` | Fairness warmup + config 기반 loss 조합 |
| 10 | `CLIP_stage2/config/train_stage2_v2.yaml` | 전체 설정 (CLIP norm, fairness_strategy) |
| 11 | `CLIP_stage2/train_stage2_v2.py` | CLI 진입점 (--preset full/cvar/ldam/minimal/none) |

### 1.2 수정 파일 (5개)

| # | 파일 경로 | 변경 내용 |
|---|----------|----------|
| 1 | `CLIP_stage1/model/fairness_adapter.py` | `use_grl`, `initial_lambda_grl` 파라미터 추가, GRL을 classifier 앞에 삽입 |
| 2 | `CLIP_stage1/losses/combined_loss.py` | `create_stage1_loss()` 팩토리 함수 추가 |
| 3 | `CLIP_stage1/trainer/stage1_trainer.py` | GRL scheduling, best model 기준 변경, cosine sim 로깅 |
| 4 | `CLIP_stage1/config/train_stage1.yaml` | GRL config, 20 epochs, `best_model_selection` 추가 |
| 5 | `CLIP_stage1/train_stage1.py` | `flatten_config`에 GRL/best_model_selection 전달, `create_model`에 GRL 파라미터 추가 |

---

## 2. 해결된 문제 매핑

| 문제 ID | 문제 | 해결 방법 | 관련 파일 |
|---------|------|----------|----------|
| P1 | Loss 목표 모순 (classifier + adapter 공동학습) | GRL이 gradient 반전 -> adversarial debiasing | `gradient_reversal.py`, `fairness_adapter.py` |
| P2 | Best model 선정 기준 모순 | `score = cos_sim - 0.3*race_acc - 0.2*gender_acc` | `stage1_trainer.py`, `train_stage1.yaml` |
| P3 | Sinkhorn reference bias | GRL + Sinkhorn 하이브리드로 이중 debiasing | `adversarial_fairness_loss.py` |
| P4 | Cross-Attention 퇴화 (seq_len=1) | 4-view projection -> seq_len=4 | `demographic_cross_attention.py` |
| P5 | Fairness loss 기여도 미미 | CVaR + LDAM + Sinkhorn 3중 전략 | `subgroup_conditional_loss.py`, `ldam_loss.py` |
| P6 | Degenerate Phase (초기 전체 Real 예측) | Fairness warmup (0-5 epoch off, 5-10 선형 증가) | `stage2_trainer_v2.py` |
| P7 | Feature disentanglement 부재 | Cosine dissimilarity loss | `contrastive_loss.py` |
| P8 | Stage 1이 bias 미제거 (race_acc 85%) | GRL adversarial -> race_acc 목표 <35% | `gradient_reversal.py`, `fairness_adapter.py` |
| P9 | Normalization 불일치 | CLIP 표준으로 통일 [0.481, 0.458, 0.408] | `train_stage2_v2.yaml` |
| P10 | Global vs Local 구분 불명확 | Stage1=Ignorance(GRL) / Stage2=Awareness(CVaR+LDAM) | 전체 아키텍처 |

---

## 3. Stage 1 아키텍처

```
Input Image
    |
CLIP Visual Encoder (Frozen, 768-dim)
    |
clip_feat [B, 768]
    |
Additive Adapter (Trainable, 768->512->512->768)
    |
debiased_feat = clip_feat + adapter(clip_feat)
    |
    +-------------------+-------------------+-------------------+
    |                   |                   |                   |
GRL(lambda)       Cosine Sim Loss    Sinkhorn Global     Pairwise Sinkhorn
    |             cos(clip, debiased) (subgroup<->전체)   (subgroup<->subgroup)
    +------+------+
    |             |
Race CLF(4)  Gender CLF(2)
    |             |
CE Loss        CE Loss
(반전 gradient -> adapter가 인구통계 제거)
```

### 3.1 GRL 작동 원리

- Forward: 특징이 그대로 classifier에 전달
- Backward: classifier의 gradient가 `-lambda_grl`만큼 반전되어 adapter에 전달
- 결과: adapter가 classifier를 속이는 방향으로 학습 -> feature에서 인구통계 정보 제거

### 3.2 GRL Lambda Scheduling (DANN 논문 방식)

```
p = epoch / total_epochs
lambda_grl = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
```

- 학습 초기: lambda=0 (adversarial 약함, adapter 안정적 학습)
- 학습 후기: lambda=1 (adversarial 강함, 본격 debiasing)

### 3.3 Loss 구성

```
L_stage1 = lambda_race * CE(race_clf(GRL(debiased)), race_label)        # GRL 반전
         + lambda_gender * CE(gender_clf(GRL(debiased)), gender_label)   # GRL 반전
         + lambda_sim * (1 - cosine_similarity(clip_feat, debiased))     # 유용 정보 보존
         + lambda_fairness * Sinkhorn(subgroup_feats, global_feats)      # 분포 정렬
         + lambda_pairwise * PairwiseSinkhorn(subgroup_pairs)            # 쌍별 정렬
```

기본값: `lambda_race=1.0, lambda_gender=0.5, lambda_sim=2.0, lambda_fairness=0.5, lambda_pairwise=0.3`

### 3.4 Best Model 선정 기준 (v2)

```
score = 1.0 * cosine_sim + (-0.3) * race_acc + (-0.2) * gender_acc
```

- cosine_sim 높을수록 좋음 (유용 정보 보존)
- race_acc 낮을수록 좋음 (random 25%에 가까울수록 debiased)
- gender_acc 낮을수록 좋음 (random 50%에 가까울수록 debiased)

### 3.5 검증 목표

- race_acc < 35% (random=25%)
- gender_acc < 60% (random=50%)
- cosine_similarity > 0.85

---

## 4. Stage 2 아키텍처

```
Input Image
    |
CLIP Visual Encoder (Frozen, 768-dim)
    |
clip_feat [B, 768]
    |
    +-------------------+-------------------+
    |                                       |
Stage 1 Adapter (Frozen)         Stage 2 Adapter (Trainable)
768->512->512->768               768->384->384->768
    |                                       |
fair_feat (no grad)              detect_feat (grad)
    |                                       |
    +-----------------+---------------------+
                      |
       DemographicFusionModule
       |
       +-- Step 1: AdaIN
       |   adain_feat = AdaIN(detect_feat, fair_feat)
       |   (fair_feat의 mean/var로 detect_feat 정규화)
       |
       +-- Step 2: Multi-View Cross-Attention
       |   4개 view projection: [B, 768] -> [B, 4, 768]
       |   Query: detect_feat [B, 1, 768]
       |   Key/Value: fair_views [B, 4, 768]
       |   -> seq_len=4로 attention 실질 작동
       |
       +-- Step 3: Gated Fusion
           gate = sigmoid(W * [adain_out; ca_out])
           fused = gate * adain_out + (1-gate) * ca_out
           (gate_init_bias=0.0 -> sigmoid(0)=0.5 균등 출발)
                      |
               fused_feat [B, 768]
                      |
            Binary Classifier (768->384->192->2)
                      |
               Real/Fake Prediction

    (Optional) LDAM Fairness Head
               fused_feat -> 8-class subgroup prediction
```

### 4.1 AdaIN (Adaptive Instance Normalization)

```python
normalized = InstanceNorm(detect_feat)
gamma, beta = LinearProjection(fair_feat)   # fair_feat에서 style 파라미터 추출
output = gamma * normalized + beta          # fair 통계로 detect 재조정
```

- 파라미터 효율적으로 fair statistics를 detection feature에 주입
- 초기화: gamma~1, beta~0 (identity에 가깝게 시작)

### 4.2 Demographic-Aware Multi-View Cross-Attention

```python
# fair_feat를 4개 view로 확장
view_1 = Linear_1(fair_feat)   # Gender view
view_2 = Linear_2(fair_feat)   # Race view
view_3 = Linear_3(fair_feat)   # General view 1
view_4 = Linear_4(fair_feat)   # General view 2

fair_views = stack([view_1, view_2, view_3, view_4])  # [B, 4, 768]

# Cross-Attention: detect가 fair의 4개 view를 참조
attn_out = MultiHeadAttention(
    query=detect_feat.unsqueeze(1),   # [B, 1, 768]
    key=fair_views,                    # [B, 4, 768]
    value=fair_views                   # [B, 4, 768]
)
# -> attn_out: [B, 1, 768], attn_weights: [B, 1, 4]
```

- 기존 문제: seq_len=1 -> 8-head attention이 trivial weight 1.0 (파라미터 낭비)
- 해결: 4개 view로 확장 -> seq_len=4 -> attention이 의미있게 동작

### 4.3 Fairness Strategy (Config 기반 선택)

#### CVaR (Subgroup-Conditional Detection Loss)

```
1. Per-sample CE loss 계산 (reduction='none')
2. 각 subgroup별로 loss 집계
3. Inner CVaR: subgroup 내 worst-case (top 10%)
4. Outer CVaR: subgroup 간 worst-case (top 50%)
5. Real/Fake 클래스별 별도 적용
```

- Prediction-level fairness: Detection 성능이 subgroup 간 동일해지도록

#### LDAM (Label-Distribution-Aware Margin)

```
margin_k = max_margin / sqrt(sqrt(n_k))
adjusted_logit = logit - margin * one_hot(target)
loss = CE(s * adjusted_logit, target)
```

- 소수 subgroup에 더 큰 margin -> 빈도 불균형 보정
- Auxiliary 8-class subgroup 분류 head

#### Sinkhorn (Per-class Feature Distribution Alignment)

```
L_sinkhorn = (L_real + L_fake) / 2
L_class = mean(Sinkhorn(subgroup_i, subgroup_j)) for all pairs
```

- Feature-level fairness: 각 클래스 내에서 subgroup 간 분포 정렬

### 4.4 Loss 구성

```
L_stage2 = (1-w) * CE + w * CVaR                                        # Detection + CVaR
         + w * lambda_ldam * LDAM(fused_feat, subgroups)                 # LDAM head
         + w * lambda_sinkhorn * Sinkhorn(fused_feat, labels, subgroups) # Per-class Sinkhorn
         + lambda_contrastive * Disentanglement(stage1, stage2)          # Feature 분리

w = fairness_weight (warmup scheduling에 따라 0->1)
```

### 4.5 Fairness Warmup Scheduling

```
epoch < 5:       fairness_weight = 0.0   (detection만 학습)
5 <= epoch < 10: fairness_weight = 선형 증가 (0.0 -> 1.0)
epoch >= 10:     fairness_weight = 1.0   (전체 fairness 적용)
```

- 초기 degenerate phase 방지: detection이 먼저 안정화된 후 fairness 적용

### 4.6 Config Presets

| Preset | CVaR | LDAM | Sinkhorn | 용도 |
|--------|------|------|----------|------|
| `full` | O | O | O | 최강 fairness |
| `cvar` | O | X | O | CVaR 중심 |
| `ldam` | X | O | O | LDAM 중심 |
| `minimal` | X | X | O | Sinkhorn만 |
| `none` | X | X | X | Baseline (fairness 없음) |

### 4.7 Best Model 선정 기준

```
score = 1.0 * AUC + (-0.001) * F_FPR + (-0.001) * F_MEO
```

### 4.8 검증 목표

- FF++ Val AUC >= 0.82
- Cross-dataset 평균 AUC >= 0.71
- Val F_FPR < 120%
- Val F_MEO < 70%
- Gate 동적 범위: 0.3~0.7
- Attention weight: 4개 view에 균등 분산

---

## 5. Stage 1 vs Stage 2 역할 분리

| 측면 | Stage 1 (Global) | Stage 2 (Local) |
|------|-----------------|-----------------|
| 목적 | CLIP의 내재적 인구통계 bias 제거 | Deepfake Detection 내 subgroup 공정성 |
| 철학 | Fairness through Ignorance | Fairness through Awareness |
| Fairness 유형 | Task-agnostic | Task-specific (detection) |
| Fairness 수준 | Feature-level (분포 정렬 + adversarial) | Prediction-level (탐지 성능 균등화) |
| 학습 데이터 | FairFace, UTKFace, CausalFace | FF++ |
| 핵심 Loss | GRL adversarial + Sinkhorn | CVaR + LDAM + Sinkhorn |
| Adapter | Trainable -> Stage 2에서 Frozen | Trainable (detection 특화) |
| 평가 기준 | race_acc < 35%, cosine_sim > 0.85 | AUC >= 0.82, F_FPR < 120% |

---

## 6. 실행 방법

### 6.1 Stage 1 학습

```bash
cd /workspace/code/CLIP_stage1
python train_stage1.py --config config/train_stage1.yaml
```

### 6.2 Stage 2 학습

```bash
cd /workspace/code/CLIP_stage2

# Full preset (CVaR + LDAM + Sinkhorn)
python train_stage2_v2.py --config config/train_stage2_v2.yaml --preset full

# CVaR preset
python train_stage2_v2.py --config config/train_stage2_v2.yaml --preset cvar

# LDAM preset
python train_stage2_v2.py --config config/train_stage2_v2.yaml --preset ldam

# Minimal preset (Sinkhorn만)
python train_stage2_v2.py --config config/train_stage2_v2.yaml --preset minimal

# Baseline (fairness 없음)
python train_stage2_v2.py --config config/train_stage2_v2.yaml --preset none
```

### 6.3 Stage 1 체크포인트 지정

```bash
python train_stage2_v2.py \
    --config config/train_stage2_v2.yaml \
    --preset full \
    --stage1_checkpoint /path/to/stage1/checkpoint_best.pth
```

---

## 7. Ablation Study 계획

| # | 실험 | 명령어 |
|---|------|--------|
| 1 | Baseline (CLIP + Linear + CE) | `--preset none` |
| 2 | CVaR only | `--preset cvar` |
| 3 | LDAM only | `--preset ldam` |
| 4 | Sinkhorn only | `--preset minimal` |
| 5 | Full (CVaR + LDAM + Sinkhorn) | `--preset full` |
| 6 | Stage 1 GRL only (Sinkhorn 제거) | Stage 1 yaml에서 `lambda_fairness: 0, lambda_pairwise: 0` |
| 7 | Stage 2 AdaIN only (CA 제거) | `num_ca_views: 0` (별도 config) |

---

## 8. 주요 하이퍼파라미터

### Stage 1

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `grl.schedule` | `dann` | DANN sigmoid scheduling |
| `grl.gamma` | `10.0` | Scheduling 곡선 가파른 정도 |
| `grl.max_lambda` | `1.0` | 최대 GRL 강도 |
| `loss.lambda_race` | `1.0` | Race adversarial loss 가중치 |
| `loss.lambda_gender` | `0.5` | Gender adversarial loss 가중치 |
| `loss.lambda_similarity` | `2.0` | Cosine sim 보존 loss 가중치 |
| `loss.lambda_fairness` | `0.5` | Global Sinkhorn 가중치 |
| `loss.lambda_pairwise` | `0.3` | Pairwise Sinkhorn 가중치 |
| `training.num_epochs` | `20` | 학습 에포크 수 |
| `training.lr` | `1e-3` | 학습률 |

### Stage 2

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `model.num_ca_views` | `4` | Cross-Attention view 수 |
| `model.gate_init_bias` | `0.0` | Gate 초기값 (0.5 균등) |
| `fairness.cvar.inner_alpha` | `0.9` | Subgroup 내 worst 10% |
| `fairness.cvar.outer_alpha` | `0.5` | Subgroup 간 worst 50% |
| `fairness.ldam.lambda_ldam` | `0.1` | LDAM loss 가중치 |
| `fairness.ldam.max_margin` | `0.5` | LDAM margin 상한 |
| `fairness.sinkhorn.lambda_sinkhorn` | `0.1` | Sinkhorn loss 가중치 |
| `fairness.warmup.warmup_epochs` | `5` | Fairness warmup 시작 |
| `fairness.warmup.warmup_end_epoch` | `10` | Fairness warmup 완료 |
| `contrastive.lambda_contrastive` | `0.05` | Disentanglement 가중치 |
| `training.num_epochs` | `50` | 학습 에포크 수 |
| `training.lr` | `3e-4` | 학습률 |
