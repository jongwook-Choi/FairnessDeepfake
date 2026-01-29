# PG-FDD 벤치마크 프레임워크 통합 정리

> **날짜**: 2026-01-29
> **목적**: PG-FDD (Preserving Generalization - Fair Deepfake Detector)를 벤치마크 프레임워크의 12번째 detector로 통합
> **소스**: `/workspace/code/reproducing/Fairness-Generalization` 공식 구현 기반

---

## 1. 파일 변경 목록

### 1.1 신규 생성 파일 (5개)

| # | 파일 경로 | 핵심 역할 |
|---|----------|----------|
| 1 | `detectors/pg_fdd_detector.py` | PG-FDD detector (3×Xception, 4 Head, AdaIN, Conditional UNet, Bi-level CVaR) |
| 2 | `loss/bi_level_CE.py` | BiCE loss — CrossEntropyLoss(reduction='none'), per-sample loss 반환 |
| 3 | `sam.py` | SAM optimizer — Sharpness Aware Minimization (first_step/second_step 2-phase) |
| 4 | `utils/bypass_bn.py` | BatchNorm running stats 제어 (enable/disable_running_stats) |
| 5 | `configs/pg_fdd.yaml` | PG-FDD 설정 파일 (SAM, paired, batch_size=16, 100 epochs) |

### 1.2 수정 파일 (4개)

| # | 파일 경로 | 변경 내용 |
|---|----------|----------|
| 1 | `detectors/__init__.py` | `PgFddDetector` import 추가 (12번째 detector) |
| 2 | `loss/__init__.py` | `BiCE` import 추가 |
| 3 | `train.py` | SAM optimizer 빌드 + SAM 2-step 학습 루프 + `softmax_fused` 검증 분기 |
| 4 | `evaluate.py` | `softmax_fused` output_type 분기 추가 |

### 1.3 생성 디렉토리

| 경로 | 용도 |
|------|------|
| `results/pg_fdd/checkpoints/` | 학습 체크포인트 저장 |
| `results/pg_fdd/eval/` | 평가 결과 저장 |

---

## 2. PG-FDD 아키텍처

### 2.1 전체 구조

```
Input Images (256×256, batch=16, paired: fake+real)
    │
    ▼
3개 독립 Xception Encoder (각각 pretrained weight 로드)
├── encoder_f   → forgery features  (512-dim)
├── encoder_c   → content features  (512-dim)
└── encoder_fair → fairness features (512-dim)
    │
    ▼
Feature Processing (Conv2d 1×1: 512→256)
├── block_spe(forgery)  → f_spe   [B, 256, 8, 8]
├── block_sha(forgery)  → f_share [B, 256, 8, 8]
└── block_fair(fairness) → f_fair  [B, 256, 8, 8]
    │
    ▼
AdaIN Fusion
    fused_features = AdaIN(f_fair, f_share)  [B, 256, 8, 8]
    │
    ▼
4개 Classification Head
├── head_spe(f_spe)            → 6 classes (specific manipulation type)
├── head_sha(f_share)          → 2 classes (shared binary: real/fake)
├── head_fair(f_fair)          → 8 classes (subgroup classification)
└── head_fused(fused_features) → 2 classes (final detection) ← Inference 사용
    │
    ▼ (학습 시에만)
Reconstruction
    Conditional_UNet + AdaIN → self/cross reconstruction images
```

### 2.2 Feature Disentanglement 설계

| Feature | 역할 | 인코더 | 목표 |
|---------|------|--------|------|
| `f_spe` | Forgery-specific | encoder_f | 조작 유형 분류 (6-class) |
| `f_share` | Shared/Common | encoder_f | 공통 real/fake 분류 (binary) |
| `f_fair` | Fairness-aware | encoder_fair | 인구통계 subgroup 인식 (8-class) |
| `fused` | Fair + Shared | AdaIN(f_fair, f_share) | 최종 공정한 탐지 (binary) |

### 2.3 AdaIN (Adaptive Instance Normalization)

```python
# f_fair의 통계(mean, std)로 f_share를 재정규화
x_normalized = (f_share - mean(f_share)) / std(f_share)
output = x_normalized * std(f_fair) + mean(f_fair)
```

- Fairness feature의 통계 정보를 Detection feature에 주입
- 인구통계 편향 없는 탐지를 유도

### 2.4 Conditional_UNet (Reconstruction)

```
학습 시 paired data (fake + real)를 반으로 분할:
  f1, f2 = f_all.chunk(2)   (forgery features)
  c1, c2 = content.chunk(2) (content features)
  d1, d2 = fair.chunk(2)    (fairness features)

Self-Reconstruction:
  img1_recon = UNet(f1, AdaIN(d1, c1))  → fake 복원
  img2_recon = UNet(f2, AdaIN(d2, c2))  → real 복원

Cross-Reconstruction:
  img1_cross = UNet(f1, AdaIN(d2, c2))  → fake forgery + real content
  img2_cross = UNet(f2, AdaIN(d1, c1))  → real forgery + fake content
```

---

## 3. Loss 구성 (6개 Component)

```
total_loss = loss_sha + 0.1*loss_spe + 0.3*loss_reconstruction
           + 0.05*loss_con + 0.1*loss_fair + loss_fuse
```

| # | Loss | 함수 | 가중치 | 역할 |
|---|------|------|--------|------|
| 1 | `loss_sha` | CrossEntropy(pred_sha, label) | 1.0 | Shared binary 분류 |
| 2 | `loss_spe` | CrossEntropy(pred_spe, label_spe) | 0.1 | Specific 조작 유형 분류 |
| 3 | `loss_reconstruction` | L1(original, reconstructed) × 4 | 0.3 | Self + Cross 재구성 |
| 4 | `loss_con` | ContrastiveRegularization(common, specific, label) | 0.05 | Feature 분리 (margin=3.0) |
| 5 | `loss_fair` | LDAM/Balance(pred_fair, intersec_label) | 0.1 | Subgroup 분류 (불균형 보정) |
| 6 | `loss_fuse` | **Bi-level Smooth CVaR** | 1.0 | 공정한 탐지 (핵심) |

### 3.1 LDAM (Balance) Loss

- 8개 subgroup별 class frequency: `[2475, 25443, 1468, 4163, 8013, 31281, 1111, 2185]`
- Margin: `m_k = 1 / sqrt(sqrt(n_k))` — 소수 subgroup에 더 큰 margin 부여
- 목적: 인구통계 불균형 보정

### 3.2 Bi-level Smooth CVaR (loss_fuse) — 핵심 Fairness Loss

```
Step 1: Per-sample CE loss (reduction='none') 계산
Step 2: 각 subgroup별로 Inner CVaR 계산
    inner_loss_k = CVaR(losses[subgroup_k], α_inner=0.9)
    → subgroup 내 worst 10% 샘플에 집중
Step 3: Outer CVaR 계산
    outer_loss = Smooth_CVaR(inner_losses, α_outer=0.5, τ1=0.001, τ2=0.0001)
    → subgroup 간 worst 50%에 집중
```

- `scipy.optimize.fminbound()`: 최적 lambda 탐색 (CPU에서 실행)
- Smooth CVaR: log-sum-exp 근사로 미분 가능하게 변환
- 목적: 가장 불리한 subgroup의 성능을 집중적으로 개선

---

## 4. SAM Optimizer (Sharpness Aware Minimization)

### 4.1 원리

일반 optimizer가 loss landscape의 sharp minimum에 수렴하는 것을 방지하여 일반화 성능 향상.

```
Standard:  θ ← θ - lr * ∇L(θ)
SAM:       θ ← θ - lr * ∇L(θ + ε)   where ε = ρ * ∇L(θ) / ||∇L(θ)||
```

### 4.2 2-Step 학습 루프

```python
# Step 1: Perturbation (sharp minimum 탐색)
enable_running_stats(model)         # BN 정상 동작
pred_dict = model(data_dict)
losses['overall'].backward()
optimizer.first_step(zero_grad=True)  # θ → θ + ε

# Step 2: Actual Update (flat minimum 방향으로 이동)
disable_running_stats(model)        # BN stats 고정 (momentum=0)
pred_dict = model(data_dict)
losses['overall'].backward()
optimizer.second_step(zero_grad=True)  # θ + ε → θ_new (θ로 복원 후 업데이트)
```

### 4.3 bypass_bn 역할

| 함수 | 동작 | 사용 시점 |
|------|------|----------|
| `enable_running_stats(model)` | BN momentum 복원 | SAM Step 1 전 |
| `disable_running_stats(model)` | BN momentum=0 (stats 업데이트 중지) | SAM Step 2 전 |

- Step 2에서 perturbed 파라미터로 인한 잘못된 BN stats 업데이트 방지

---

## 5. train.py 변경사항

### 5.1 `build_optimizer()` — SAM 분기 추가

```python
if opt_type == 'sam':
    from sam import SAM
    optimizer = SAM(model.parameters(), torch.optim.SGD,
                    lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    params = filter(lambda p: p.requires_grad, model.parameters())
    # 기존 SGD/Adam/AdamW 로직
```

> **주의**: SAM은 `model.parameters()`를 직접 전달 (filter 불필요 — SAM 내부에서 param_groups 관리)

### 5.2 `train_one_epoch()` — SAM 2-step 분기 추가

- `use_sam` 플래그로 분기
- SAM: `enable_running_stats → forward → backward → first_step → disable_running_stats → forward → backward → second_step`
- 일반: `forward → backward → optimizer.step()` (기존 로직 유지)

### 5.3 `validate()` — `softmax_fused` 분기 추가

```python
elif output_type == 'softmax_fused':
    probs = torch.softmax(pred_dict['cls_fused'], dim=1)[:, 1]
```

- PG-FDD는 `cls_fused` 출력으로 최종 탐지 수행

---

## 6. evaluate.py 변경사항

### `evaluate_single_dataset()` — `softmax_fused` 분기 추가

```python
elif output_type == 'softmax_fused':
    probs = torch.softmax(pred_dict['cls_fused'], dim=1)[:, 1]
```

---

## 7. Config 설정 (configs/pg_fdd.yaml)

```yaml
model:
  name: "pg_fdd"
  output_type: "softmax_fused"   # cls_fused + softmax

data:
  data_mode: "paired"            # PG-FDD requires paired data (fake+real per batch)

training:
  epochs: 100
  batch_size: 16                 # 3×Xception → GPU 메모리 큼
  optimizer: "sam"               # SAM optimizer
  lr: 0.0005
  momentum: 0.9
  weight_decay: 0.005
  scheduler:
    type: "step"
    step_size: 60
    gamma: 0.9
  seed: 5
  num_workers: 8
```

---

## 8. Inference 동작

```python
# 학습 시: forward(data_dict, inference=False)
#   → 모든 head 활성화 + reconstruction + 6개 loss 계산
#   → output keys: cls, cls_spe, cls_fair, cls_fused, recontruction_imgs, ...

# 추론 시: forward(data_dict, inference=True)
#   → head_sha, head_spe, head_fused만 실행 (reconstruction 없음)
#   → output keys: cls, feat, cls_fused, feat_fused
#   → 최종 예측: softmax(cls_fused)[:, 1]
```

---

## 9. Benchmark 등록 현황 (12개 Detector)

| # | Detector | Module Name | Output Type | Data Mode | 특징 |
|---|----------|-------------|-------------|-----------|------|
| 1 | XceptionDetector | `xception` | sigmoid | unpaired | Xception backbone |
| 2 | EfficientDetector | `efficientnet` | sigmoid | unpaired | EfficientNet-B4 |
| 3 | F3netDetector | `f3net` | sigmoid | unpaired | DCT Frequency Analysis |
| 4 | SpslDetector | `spsl` | sigmoid | unpaired | Phase Spectrum |
| 5 | SRMDetector | `srm` | sigmoid | unpaired | SRM Filter |
| 6 | CoreDetector | `core` | softmax | unpaired | Consistent Representation |
| 7 | UCFDetector | `ucf` | sigmoid | paired | Uncertainty-guided Contrastive |
| 8 | DawFddDetector | `daw_fdd` | sigmoid | unpaired | Domain Alignment Weighting |
| 9 | DagFddDetector | `dag_fdd` | sigmoid | unpaired | Domain-Agnostic Generalization |
| 10 | ViTDetector | `vit` | sigmoid | unpaired | Vision Transformer |
| 11 | UnivFDDetector | `univfd` | sigmoid | unpaired | Universal Forgery Detection |
| 12 | **PgFddDetector** | **`pg_fdd`** | **softmax_fused** | **paired** | **3×Xception + AdaIN + Bi-level CVaR + SAM** |

---

## 10. 실행 방법

### 10.1 학습

```bash
cd /workspace/code/reproducing/benchmark
python train.py --config configs/pg_fdd.yaml --gpu 0
```

### 10.2 평가

```bash
python evaluate.py \
    --config configs/pg_fdd.yaml \
    --checkpoint results/pg_fdd/checkpoints/best.pth \
    --gpu 0
```

### 10.3 특정 데이터셋 평가

```bash
python evaluate.py \
    --config configs/pg_fdd.yaml \
    --checkpoint results/pg_fdd/checkpoints/best.pth \
    --dataset celebdf \
    --gpu 0
```

---

## 11. 주의사항

| 항목 | 설명 |
|------|------|
| **GPU 메모리** | 3개 Xception encoder + Conditional UNet → GPU 메모리 사용량 큼 (batch_size=16 권장) |
| **학습 시간** | SAM은 매 step 2번의 forward-backward → 학습 시간 약 2배 |
| **CPU 오버헤드** | Bi-level CVaR의 `scipy.optimize.fminbound`는 CPU 실행 → 약간의 오버헤드 |
| **Paired 데이터** | `data_mode: paired` 필수 — 학습 시 fake+real 쌍으로 배치 구성 |
| **Pretrained** | `pretrained/xception-b5690688.pth` 필수 — 없으면 성능 저하 |
| **get_losses 분기** | `label_spe`와 `recontruction_imgs` 유무로 학습/추론 자동 판단 |

---

## 12. 검증 결과 (2026-01-29)

| 검증 항목 | 결과 |
|----------|------|
| 구문 검증 (AST parse) | 모든 파일 통과 |
| Import 검증 | 12개 detector + 10개 loss 모두 registry 등록 확인 |
| Forward pass (학습) | dummy data → 14개 output key 정상, Loss 6개 항목 계산 완료 |
| Backward pass | `overall_loss.backward()` 정상 완료 |
| Forward pass (추론) | dummy data → `cls_fused` shape [B, 2] 정상 |
| SAM optimizer | first_step → second_step → scheduler.step() 정상 |
| 실제 학습 | `python train.py --config configs/pg_fdd.yaml --gpu 0` 정상 시작, Batch 100/3950 도달 (Loss: 3.576) |
