"""
Stage 2 Independent Adapter Model
Independent Dual Adapter with Cross-Attention Fusion

철학:
- Stage 1 Adapter: CLIP의 Global Bias 보정을 위해 학습된 adapter (Frozen)
- Stage 2 Adapter: CLIP feature를 Deepfake Detection에 직접 활용 + Generalization + Local Fairness 학습
- 두 Adapter는 독립적으로 CLIP feature를 입력으로 받음
- Cross-Attention으로 Detection 정보에 Fairness 정보를 주입

아키텍처:
    Input Image → CLIP Visual Encoder (Frozen) → CLIP Feature (768-dim)
                                                        │
                ┌───────────────────────────────────────┴───────────────────────────────────────┐
                │                                                                               │
                ▼                                                                               ▼
    Stage 1 Adapter (Frozen)                                                    Stage 2 Adapter (Trainable)
    768 → 512 → 512 → 768                                                       768 → 384 → 384 → 768
    목적: Global Fairness 보정                                                   목적: Detection + Local Fair
                │                                                                               │
                ▼                                                                               ▼
    Stage1_Feature = CLIP + Stage1_Add                                          Stage2_Feature = CLIP + Stage2_Add
    (Global Fair)                                                               (Detection-oriented)
                │                                                                               │
                └───────────────────────────────────────┬───────────────────────────────────────┘
                                                        │
                                                        ▼
                                        Cross-Attention Fusion with Dynamic Gate
                                        Query: Stage2_Feature (Detection)
                                        Key/Value: Stage1_Feature (Fairness)
                                                        │
                                                        ▼
                                                Fused_Feature (768)
                                                        │
                                                        ▼
                                            Binary Classifier (Real/Fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve

from model.clip.clip import load
from model.additive_adapter import AdditiveAdapter
from model.cross_attention_fusion import CrossAttentionFusionWithDynamicGate


class Stage2IndependentAdapterModel(nn.Module):
    """
    Independent Dual Adapter with Cross-Attention Fusion

    핵심 설계:
    1. Independent Adapters: Stage 1, Stage 2 모두 CLIP feature를 독립적으로 입력
    2. 역할 분리: Stage 1 = Global Fair (frozen), Stage 2 = Detection + Local Fair (trainable)
    3. Cross-Attention Fusion: Detection이 Fairness를 참조
    4. Dynamic Gate: Fairness 성능에 따라 Stage 1 활용도 자동 조절
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 feature_dim=768,
                 stage1_hidden_dim=512,
                 stage2_hidden_dim=384,
                 classifier_hidden_dims=[384, 192],
                 num_classes=2,
                 dropout=0.1,
                 fusion_num_heads=8,
                 normalize_features=True,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):
        """
        Args:
            clip_name (str): CLIP 모델 이름
            feature_dim (int): CLIP feature 차원 (ViT-L/14: 768)
            stage1_hidden_dim (int): Stage 1 Adapter hidden 차원 (512)
            stage2_hidden_dim (int): Stage 2 Adapter hidden 차원 (384)
            classifier_hidden_dims (list): Binary classifier hidden 차원
            num_classes (int): 출력 클래스 수 (2: Real/Fake)
            dropout (float): Dropout 비율
            fusion_num_heads (int): Cross-attention head 수
            normalize_features (bool): Feature normalization 여부
            device (str): Device
            clip_download_root (str): CLIP 가중치 경로
        """
        super().__init__()

        self.device = device
        self.normalize_features = normalize_features
        self.feature_dim = feature_dim

        # 1. CLIP Visual Encoder (Frozen)
        self.clip_model, self.preprocess = load(
            clip_name,
            device=device,
            download_root=clip_download_root
        )

        # Language transformer 제거 (메모리 절약)
        if hasattr(self.clip_model, 'transformer'):
            delattr(self.clip_model, 'transformer')

        # Feature 차원 자동 감지
        if feature_dim is None:
            self.feature_dim = self.clip_model.visual.output_dim

        # CLIP 동결
        self._freeze_clip()

        # 2. Stage 1 Adapter (Frozen) - Global Fairness
        # 768 → 512 → 512 → 768
        self.stage1_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=stage1_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # 3. Stage 2 Adapter (Trainable) - Detection + Local Fairness
        # 768 → 384 → 384 → 768
        self.stage2_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=stage2_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # 4. Cross-Attention Fusion with Dynamic Gate
        self.fusion = CrossAttentionFusionWithDynamicGate(
            dim=self.feature_dim,
            num_heads=fusion_num_heads,
            dropout=dropout
        )

        # 5. Binary Classifier
        self.binary_classifier = self._build_classifier(
            self.feature_dim,
            classifier_hidden_dims,
            num_classes,
            dropout
        )

        # Metric 추적용 변수
        self.prob = []
        self.label = []

        # Loss function (외부에서 설정 가능)
        self.loss_fn = nn.CrossEntropyLoss()

    def _freeze_clip(self):
        """CLIP visual encoder 완전 동결"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def _freeze_stage1_adapter(self):
        """Stage 1 Adapter 동결"""
        for param in self.stage1_adapter.parameters():
            param.requires_grad = False
        self.stage1_adapter.eval()

    def _build_classifier(self, input_dim, hidden_dims, output_dim, dropout):
        """Binary Classifier 생성"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        classifier = nn.Sequential(*layers)

        # 가중치 초기화
        for module in classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        return classifier

    def load_stage1_checkpoint(self, checkpoint_path, strict=False):
        """
        Stage 1 체크포인트에서 Stage 1 Adapter 가중치 로드

        Args:
            checkpoint_path (str): Stage 1 체크포인트 경로
            strict (bool): strict 로딩 여부
        """
        print(f"Loading Stage 1 checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Stage 1 Adapter 가중치만 추출
        adapter_state_dict = {}
        for key, value in state_dict.items():
            if 'additive_adapter' in key:
                new_key = key.replace('additive_adapter.', '')
                adapter_state_dict[new_key] = value

        if len(adapter_state_dict) == 0:
            print("Warning: No additive_adapter weights found in checkpoint!")
            print(f"Available keys: {list(state_dict.keys())[:10]}...")
            return False

        # Stage 1 Adapter에 가중치 로드
        missing_keys, unexpected_keys = self.stage1_adapter.load_state_dict(
            adapter_state_dict, strict=strict
        )

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded Stage 1 Adapter weights ({len(adapter_state_dict)} parameters)")

        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            print(f"  Stage 1 trained for {checkpoint['epoch']} epochs")
        if 'best_loss' in checkpoint:
            print(f"  Stage 1 best loss: {checkpoint['best_loss']:.4f}")

        # Stage 1 Adapter 동결
        self._freeze_stage1_adapter()
        print("Stage 1 Adapter frozen")

        return True

    def get_clip_features(self, images):
        """
        CLIP visual encoder로 feature 추출 (frozen)

        Args:
            images (torch.Tensor): 입력 이미지 [batch_size, 3, H, W]

        Returns:
            torch.Tensor: CLIP features [batch_size, feature_dim]
        """
        # 이미지 크기 조정 (CLIP 입력 크기: 224x224)
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

        # CLIP visual encoder로 특징 추출 (frozen)
        with torch.no_grad():
            features = self.clip_model.encode_image(images)

        return features.float()

    def forward(self, data_dict, inference=False):
        """
        Forward pass

        Args:
            data_dict (dict): 입력 데이터 딕셔너리
                - 'image': 이미지 텐서 [batch_size, 3, H, W]
            inference (bool): 추론 모드 여부

        Returns:
            dict: 예측 결과
                - 'clip_features': 원본 CLIP features
                - 'stage1_features': Stage 1 Adapter 출력 (CLIP + Stage1_Add, normalized)
                - 'stage2_features': Stage 2 Adapter 출력 (CLIP + Stage2_Add)
                - 'fused_features': Cross-Attention Fusion 출력
                - 'fused_features_norm': 정규화된 fused features
                - 'gate': Dynamic gate 값
                - 'attn_weights': Attention weights
                - 'cls': Classification logits
                - 'prob': Fake 클래스 확률
        """
        images = data_dict['image']

        # 1. CLIP Features (Frozen)
        with torch.no_grad():
            clip_feat = self.get_clip_features(images)

            # CLIP feature normalization
            if self.normalize_features:
                clip_feat = F.normalize(clip_feat, dim=-1)

        # 2. Stage 1: Global Fairness Feature (Frozen)
        with torch.no_grad():
            stage1_add = self.stage1_adapter(clip_feat)
            stage1_feat = clip_feat + stage1_add
            if self.normalize_features:
                stage1_feat = F.normalize(stage1_feat, dim=-1)

        # 3. Stage 2: Detection + Local Fairness Feature (Trainable)
        # CLIP feature를 직접 입력 (Stage 1과 독립적)
        stage2_add = self.stage2_adapter(clip_feat)
        stage2_feat = clip_feat + stage2_add
        # Stage 2는 normalization 하지 않음 (Cross-Attention에서 처리)

        # 4. Cross-Attention Fusion (Detection queries Fairness)
        fused_feat, gate, attn_weights = self.fusion(stage2_feat, stage1_feat)

        # 5. Feature normalization for classification
        if self.normalize_features:
            fused_feat_norm = F.normalize(fused_feat, dim=-1)
        else:
            fused_feat_norm = fused_feat

        # 6. Binary Classification
        cls_logits = self.binary_classifier(fused_feat_norm)

        # 7. Probability 계산
        prob = torch.softmax(cls_logits, dim=1)[:, 1]  # Fake 클래스 확률

        pred_dict = {
            'clip_features': clip_feat,
            'stage1_features': stage1_feat,
            'stage2_features': stage2_feat,
            'fused_features': fused_feat,
            'fused_features_norm': fused_feat_norm,
            'gate': gate,
            'attn_weights': attn_weights,
            'cls': cls_logits,
            'prob': prob,
        }

        # 추론 모드에서 메트릭 수집
        if inference:
            self.prob.append(pred_dict['prob'].detach().cpu())
            self.label.append(data_dict['label'].detach().cpu())

        return pred_dict

    def get_losses(self, data_dict, pred_dict):
        """
        Loss 계산 (Classification only, Fairness loss는 Trainer에서 별도 처리)

        Args:
            data_dict (dict): 입력 데이터 (label 포함)
            pred_dict (dict): 모델 출력

        Returns:
            dict: Loss 딕셔너리
        """
        labels = data_dict['label']
        cls_logits = pred_dict['cls']

        # Classification loss
        cls_loss = self.loss_fn(cls_logits, labels)

        return {
            'overall': cls_loss,
            'cls': cls_loss,
        }

    def get_test_metrics(self):
        """
        테스트 메트릭 계산

        Returns:
            dict: AUC, ACC, EER, AP 등의 메트릭
        """
        if len(self.prob) == 0 or len(self.label) == 0:
            return {'auc': 0.0, 'acc': 0.0, 'eer': 0.5, 'ap': 0.0}

        probs = torch.cat(self.prob).numpy()
        labels = torch.cat(self.label).numpy()

        # Reset
        self.prob = []
        self.label = []

        # AUC
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5

        # Accuracy
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)

        # Average Precision
        try:
            ap = average_precision_score(labels, probs)
        except Exception:
            ap = 0.0

        # EER
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            fnr = 1 - tpr
            diff = abs(fpr - fnr)
            min_idx = diff.argmin()
            eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
        except Exception:
            eer = 0.5

        return {
            'auc': auc,
            'acc': acc,
            'eer': eer,
            'ap': ap,
        }

    def get_trainable_params(self):
        """
        학습 가능한 파라미터만 반환

        Returns:
            list: Stage 2 Adapter + Fusion + Classifier 파라미터
        """
        params = []
        params.extend(self.stage2_adapter.parameters())
        params.extend(self.fusion.parameters())
        params.extend(self.binary_classifier.parameters())
        return params

    def get_trainable_params_with_names(self):
        """
        학습 가능한 파라미터와 이름 반환 (디버깅용)

        Returns:
            list: (name, param) 튜플 리스트
        """
        params = []
        for name, param in self.stage2_adapter.named_parameters():
            params.append((f'stage2_adapter.{name}', param))
        for name, param in self.fusion.named_parameters():
            params.append((f'fusion.{name}', param))
        for name, param in self.binary_classifier.named_parameters():
            params.append((f'binary_classifier.{name}', param))
        return params

    def print_trainable_parameters(self):
        """학습 가능한 파라미터 수 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\nModel Parameters:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

        # 컴포넌트별 파라미터
        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        stage1_adapter_params = sum(p.numel() for p in self.stage1_adapter.parameters())
        stage2_adapter_params = sum(p.numel() for p in self.stage2_adapter.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.binary_classifier.parameters())

        print(f"\nComponent Parameters:")
        print(f"  CLIP:              {clip_params:,} (frozen)")
        print(f"  Stage 1 Adapter:   {stage1_adapter_params:,} (frozen)")
        print(f"  Stage 2 Adapter:   {stage2_adapter_params:,} (trainable)")
        print(f"  Cross-Attn Fusion: {fusion_params:,} (trainable)")
        print(f"  Classifier:        {classifier_params:,} (trainable)")

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'clip': clip_params,
            'stage1_adapter': stage1_adapter_params,
            'stage2_adapter': stage2_adapter_params,
            'fusion': fusion_params,
            'classifier': classifier_params,
        }


if __name__ == "__main__":
    # 테스트 코드
    print("Stage 2 Independent Adapter Model Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Stage2IndependentAdapterModel(
        clip_name="ViT-L/14",
        device=device
    )

    model.print_trainable_parameters()

    # 더미 입력
    dummy_input = {
        'image': torch.randn(2, 3, 256, 256).to(device),
        'label': torch.LongTensor([0, 1]).to(device)
    }

    # Forward pass
    model.to(device)
    output = model(dummy_input)

    print(f"\nOutput keys: {output.keys()}")
    print(f"CLIP features shape: {output['clip_features'].shape}")
    print(f"Stage 1 features shape: {output['stage1_features'].shape}")
    print(f"Stage 2 features shape: {output['stage2_features'].shape}")
    print(f"Fused features shape: {output['fused_features'].shape}")
    print(f"Gate shape: {output['gate'].shape}, mean: {output['gate'].mean().item():.4f}")
    print(f"cls shape: {output['cls'].shape}")
    print(f"prob shape: {output['prob'].shape}")
