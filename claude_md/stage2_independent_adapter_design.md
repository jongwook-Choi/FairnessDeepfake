# Stage 2 재설계: Independent Dual Adapter with Cross-Attention Fusion

## 1. 설계 철학

### 핵심 원칙
1. **Stage 1 Adapter**: CLIP의 **Global Bias 보정**을 위해 학습된 adapter (Frozen)
2. **Stage 2 Adapter**: CLIP feature를 **Deepfake Detection**에 직접 활용 + **Generalization** + **Local Fairness** 학습
3. **두 Adapter는 독립적으로 CLIP feature를 입력**으로 받음
4. **Cross-Attention**으로 Detection 정보에 Fairness 정보를 주입

### 근거
- CLIP feature가 Deepfake Detection에 효과적이라는 연구 다수 존재
- Stage 1은 CLIP 자체의 bias 제거 목적 (task-agnostic)
- Stage 2는 Detection task에 특화 + 추가 fairness 학습

---

## 2. 아키텍처

### 2.1 전체 Forward Flow

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────┐
│     CLIP Visual Encoder (Frozen, 768-dim)       │
└─────────────────────────────────────────────────┘
    │
    │ CLIP_Feature
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│   Stage 1 Adapter (Frozen)      │    │   Stage 2 Adapter (Trainable)   │
│   768 → 512 → 512 → 768         │    │   768 → 384 → 384 → 768         │
│                                 │    │                                 │
│   목적: Global Fairness 보정    │    │   목적: Detection + Local Fair  │
└─────────────────────────────────┘    └─────────────────────────────────┘
    │                                             │
    ▼                                             ▼
Stage1_Feature = CLIP + Stage1_Add     Stage2_Feature = CLIP + Stage2_Add
(Global Fair)                          (Detection-oriented)
    │                                             │
    │                                             │
    └─────────────────┬───────────────────────────┘
                      ▼
    ┌─────────────────────────────────────────────┐
    │       Cross-Attention Fusion                │
    │                                             │
    │   Query: Stage2_Feature (Detection)         │
    │   Key/Value: Stage1_Feature (Fairness)      │
    │   + Dynamic Gate for Stage1 residual        │
    └─────────────────────────────────────────────┘
                      │
                      ▼
               Fused_Feature (768)
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
Binary           Fairness          Dynamic Loss
Classifier       Loss (Sinkhorn)   Weighting
(Real/Fake)          │                  │
    │                │                  │
    ▼                ▼                  ▼
  L_cls          L_fairness         λ_fair(t)
    │                │                  │
    └────────────────┴──────────────────┘
                     │
                     ▼
    L_total = L_cls + λ_fair(t) * L_fairness
```

### 2.2 Cross-Attention Fusion with Dynamic Gate

```python
class CrossAttentionFusionWithDynamicGate(nn.Module):
    """
    Detection Feature가 Fairness Feature를 참조하여 공정성 정보 주입

    철학:
    - Stage 2 (Detection)가 Query: "Detection 관점에서 필요한 Fairness 정보 선택"
    - Stage 1 (Fairness)이 Key/Value: "Global Fair 정보 제공"
    - Dynamic Gate: Fairness 성능 저하 시 Stage 1 정보 더 많이 활용
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

        # Dynamic Gate: Fairness 정보 활용도 조절
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        # 초기에 Stage1 더 많이 사용 (sigmoid(1.0) ≈ 0.73)
        nn.init.constant_(self.gate_net[-2].bias, 1.0)

    def forward(self, stage2_feat, stage1_feat):
        """
        Args:
            stage2_feat: [B, 768] - Detection feature (from Stage 2)
            stage1_feat: [B, 768] - Fairness feature (from Stage 1)
        """
        s2 = stage2_feat.unsqueeze(1)  # [B, 1, 768]
        s1 = stage1_feat.unsqueeze(1)  # [B, 1, 768]

        # Cross-Attention: Detection queries Fairness
        attn_out, attn_weights = self.cross_attn(s2, s1, s1)
        x = self.norm1(s2 + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.squeeze(1)  # [B, 768]

        # Dynamic Gate
        gate_input = torch.cat([x, stage1_feat], dim=-1)
        gate = self.gate_net(gate_input)  # [B, 1]

        # Fused = gate * Stage1 + (1-gate) * CrossAttn_result
        fused = gate * stage1_feat + (1 - gate) * x

        return fused, gate, attn_weights
```

### 2.3 Dynamic Fairness Loss Weighting

```python
class DynamicFairnessLossWeighting(nn.Module):
    """
    Uncertainty-based automatic loss balancing
    (Kendall et al., 2018: Multi-Task Learning Using Uncertainty)
    """
    def __init__(self):
        super().__init__()
        # log(σ²) 학습 - task uncertainty
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_fair = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_cls, loss_fair):
        precision_cls = torch.exp(-self.log_var_cls)
        precision_fair = torch.exp(-self.log_var_fair)

        total = (precision_cls * loss_cls + self.log_var_cls +
                 precision_fair * loss_fair + self.log_var_fair)

        lambda_fair = precision_fair / (precision_cls + precision_fair + 1e-8)
        return total, lambda_fair
```

---

## 3. Loss 함수 설계

### Stage 2 Fairness Loss

```python
class Stage2FairnessLoss(nn.Module):
    """
    Real/Fake 각 클래스 내에서 subgroup 간 분포 정렬
    """
    def __init__(self, sinkhorn_blur=1e-4, num_subgroups=8):
        super().__init__()
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn", p=2, blur=sinkhorn_blur,
            scaling=0.9, backend="tensorized"
        )
        self.num_subgroups = num_subgroups

    def forward(self, features, labels, subgroups):
        total_loss = 0.0
        count = 0

        for label_val in [0, 1]:  # Real, Fake
            mask = (labels == label_val)
            if mask.sum() >= 4:
                class_loss = self._pairwise_sinkhorn(
                    features[mask], subgroups[mask]
                )
                total_loss = total_loss + class_loss
                count += 1

        return total_loss / max(count, 1)

    def _pairwise_sinkhorn(self, features, subgroups):
        """모든 subgroup 쌍 간 Sinkhorn distance"""
        # Stage 1의 PairwiseFairnessLoss 구현 재활용
        ...
```

---

## 4. 학습 전략

### 학습 가능 파라미터

| Component | Status | 파라미터 수 |
|-----------|--------|-------------|
| CLIP Encoder | Frozen | ~303M |
| Stage 1 Adapter | Frozen | ~1.2M |
| Stage 2 Adapter | **Trainable** | ~600K |
| Cross-Attention Fusion | **Trainable** | ~2.4M |
| Dynamic Gate | **Trainable** | ~1K |
| Loss Weighting | **Trainable** | 2 params |
| Binary Classifier | **Trainable** | ~400K |
| **Total Trainable** | | **~3.4M** |

### Optimizer & Scheduler

```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 5
  T_mult: 2
  eta_min: 1e-6
  warmup_epochs: 1
```

### Joint Training
- Detection + Fairness Loss 처음부터 동시 적용
- Dynamic weighting으로 자동 균형
- Epochs: 20~30

---

## 5. 모니터링 항목

```python
log_dict = {
    # Losses
    'train/loss_total': total_loss,
    'train/loss_cls': loss_cls,
    'train/loss_fair': loss_fair,

    # Dynamic values
    'train/lambda_fair': lambda_fair,      # Fairness loss 가중치
    'train/gate_mean': gate.mean(),        # Stage1 활용도

    # Metrics
    'train/acc': accuracy,
    'val/auc': val_auc,
    'val/acc': val_acc,

    # Fairness
    'val/subgroup_auc_std': std(subgroup_aucs),  # Subgroup 간 편차
    'val/sinkhorn_dist': sinkhorn_distance,
}
```

---

## 6. 파일 생성/수정 계획

### 신규 생성 파일

| 파일 | 설명 |
|------|------|
| `model/stage2_independent_adapter_model.py` | 새 모델 클래스 |
| `model/cross_attention_fusion.py` | CrossAttentionFusionWithDynamicGate |
| `losses/stage2_fairness_loss.py` | Stage2FairnessLoss |
| `losses/dynamic_loss_weighting.py` | DynamicFairnessLossWeighting |
| `trainer/stage2_independent_adapter_trainer.py` | 새 Trainer |
| `train_stage2_independent_adapter.py` | 학습 스크립트 |
| `config/train_stage2_independent_adapter.yaml` | 설정 파일 |

### 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `dataset/fairness_dataset.py` | subgroup 정보 반환 확인 |
| `model/__init__.py` | 새 모델 import |
| `losses/__init__.py` | 새 loss import |

---

## 7. 핵심 구현: Stage2IndependentAdapterModel

```python
class Stage2IndependentAdapterModel(nn.Module):
    """
    Independent Dual Adapter with Cross-Attention Fusion

    철학:
    - Stage 1: CLIP → Global Fairness Feature (Frozen)
    - Stage 2: CLIP → Detection + Local Fairness Feature (Trainable)
    - Cross-Attention: Detection이 Fairness를 참조하여 결합
    """

    def __init__(self, config):
        super().__init__()

        # CLIP Visual Encoder (Frozen)
        self.clip_model, _ = load(config['clip_name'], ...)
        self._freeze_clip()

        # Stage 1 Adapter (Frozen) - Global Fairness
        self.stage1_adapter = AdditiveAdapter(768, 512, 768)
        self._freeze_stage1_adapter()

        # Stage 2 Adapter (Trainable) - Detection + Local Fairness
        self.stage2_adapter = AdditiveAdapter(768, 384, 768)

        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusionWithDynamicGate(dim=768)

        # Binary Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(192, 2)
        )

    def forward(self, data_dict):
        images = data_dict['image']

        # 1. CLIP Features (Frozen)
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(images).float()
            clip_feat = F.normalize(clip_feat, dim=-1)

        # 2. Stage 1: Global Fairness Feature (Frozen)
        with torch.no_grad():
            stage1_add = self.stage1_adapter(clip_feat)
            stage1_feat = F.normalize(clip_feat + stage1_add, dim=-1)

        # 3. Stage 2: Detection + Local Fairness Feature (Trainable)
        stage2_add = self.stage2_adapter(clip_feat)  # CLIP 직접 입력
        stage2_feat = clip_feat + stage2_add

        # 4. Cross-Attention Fusion (Detection queries Fairness)
        fused_feat, gate, attn_weights = self.fusion(stage2_feat, stage1_feat)
        fused_feat_norm = F.normalize(fused_feat, dim=-1)

        # 5. Classification
        logits = self.classifier(fused_feat_norm)
        prob = torch.softmax(logits, dim=1)[:, 1]

        return {
            'clip_features': clip_feat,
            'stage1_features': stage1_feat,
            'stage2_features': stage2_feat,
            'fused_features': fused_feat,
            'fused_features_norm': fused_feat_norm,
            'gate': gate,
            'attn_weights': attn_weights,
            'cls': logits,
            'prob': prob,
        }
```

---

## 8. 검증 방법

### 학습 검증
- Train Loss < 0.5
- Train Acc > 80%
- Gate value 추이 (초기 ~0.7 → 점진적 변화)

### Fairness 검증
- Subgroup별 AUC 편차 < 5%
- Sinkhorn distance 감소

### Generalization 검증
| Train | Test | Target AUC |
|-------|------|------------|
| FF++ | CelebDF | > 0.75 |
| FF++ | DFD | > 0.80 |
| FF++ | DFDC | > 0.70 |

---

## 9. 요약

### 핵심 설계
1. **Independent Adapters**: Stage 1, Stage 2 모두 CLIP feature를 독립적으로 입력
2. **역할 분리**: Stage 1 = Global Fair, Stage 2 = Detection + Local Fair
3. **Cross-Attention Fusion**: Detection이 Fairness를 참조
4. **Dynamic Gate**: Fairness 성능에 따라 Stage 1 활용도 자동 조절
5. **Dynamic Loss Weighting**: Uncertainty 기반 자동 loss 균형
