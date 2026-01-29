"""
Stage 2 Model v2: AdaIN + Demographic-Aware Cross-Attention + Optional LDAM Head

재설계된 Stage 2 모델:
1. AdaIN: fair_feat의 통계로 detect_feat 재조정 (파라미터 효율적)
2. Demographic-Aware Multi-View CA: seq_len>1로 attention 실질 작동
3. Gated Fusion: AdaIN + CA 장점 결합
4. Optional LDAM Head: subgroup-aware margin 보정

아키텍처:
    Input Image -> CLIP Visual Encoder (Frozen) -> clip_feat [B, 768]
                         |
        +----------------+------------------+
        |                                   |
    Stage 1 Adapter (Frozen)     Stage 2 Adapter (Trainable)
    768->512->512->768           768->384->384->768
        |                                   |
    fair_feat (no grad)          detect_feat (grad)
        |                                   |
        +-----------------------------------+
                         |
              Demographic-Aware Fusion Module
              (AdaIN + Multi-View CA + Gate)
                         |
                   fused_feat [B, 768]
                         |
               Binary Classifier (768->384->192->2)
                         |
                   Real/Fake Prediction

    (Optional) LDAM Fairness Head: fused_feat -> 8-class subgroup prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve

from model.clip.clip import load
from model.additive_adapter import AdditiveAdapter
from model.demographic_cross_attention import DemographicFusionModule


class Stage2ModelV2(nn.Module):
    """
    Stage 2 Model v2

    핵심 설계:
    1. Independent Adapters: Stage 1 (Frozen), Stage 2 (Trainable)
    2. DemographicFusionModule: AdaIN + Multi-View CA + Gate
    3. Optional LDAMFairnessHead: subgroup-aware auxiliary head
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 feature_dim=768,
                 stage1_hidden_dim=512,
                 stage2_hidden_dim=384,
                 classifier_hidden_dims=None,
                 num_classes=2,
                 num_ca_views=4,
                 num_heads=8,
                 dropout=0.1,
                 gate_init_bias=0.0,
                 normalize_features=True,
                 use_ldam_head=False,
                 ldam_cls_num_list=None,
                 ldam_max_margin=0.5,
                 ldam_s=30.0,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):
        """
        Args:
            clip_name (str): CLIP 모델 이름
            feature_dim (int): CLIP feature 차원
            stage1_hidden_dim (int): Stage 1 Adapter hidden 차원
            stage2_hidden_dim (int): Stage 2 Adapter hidden 차원
            classifier_hidden_dims (list): Binary classifier hidden 차원
            num_classes (int): 출력 클래스 수 (2: Real/Fake)
            num_ca_views (int): Cross-Attention view 수
            num_heads (int): Attention head 수
            dropout (float): Dropout 비율
            gate_init_bias (float): Gate 초기 bias
            normalize_features (bool): Feature normalization 여부
            use_ldam_head (bool): LDAM fairness head 사용 여부
            ldam_cls_num_list (list): LDAM subgroup 빈도 리스트
            ldam_max_margin (float): LDAM max margin
            ldam_s (float): LDAM scaling factor
            device (str): Device
            clip_download_root (str): CLIP 가중치 경로
        """
        super().__init__()

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [384, 192]

        self.device = device
        self.normalize_features = normalize_features
        self.feature_dim = feature_dim
        self.use_ldam_head = use_ldam_head

        # 1. CLIP Visual Encoder (Frozen)
        self.clip_model, self.preprocess = load(
            clip_name,
            device=device,
            download_root=clip_download_root
        )

        # Language transformer 제거
        if hasattr(self.clip_model, 'transformer'):
            delattr(self.clip_model, 'transformer')

        if feature_dim is None:
            self.feature_dim = self.clip_model.visual.output_dim

        self._freeze_clip()

        # 2. Stage 1 Adapter (Frozen) - Global Fairness
        self.stage1_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=stage1_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # 3. Stage 2 Adapter (Trainable) - Detection + Local Fairness
        self.stage2_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=stage2_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # 4. Demographic-Aware Fusion Module (AdaIN + CA + Gate)
        self.fusion = DemographicFusionModule(
            dim=self.feature_dim,
            num_views=num_ca_views,
            num_heads=num_heads,
            dropout=dropout,
            gate_init_bias=gate_init_bias
        )

        # 5. Binary Classifier (Real/Fake)
        self.binary_classifier = self._build_classifier(
            self.feature_dim,
            classifier_hidden_dims,
            num_classes,
            dropout
        )

        # 6. Optional LDAM Fairness Head
        if use_ldam_head:
            from losses.ldam_loss import LDAMFairnessHead
            self.ldam_head = LDAMFairnessHead(
                feature_dim=self.feature_dim,
                num_subgroups=8,
                cls_num_list=ldam_cls_num_list,
                max_margin=ldam_max_margin,
                ldam_s=ldam_s,
                dropout=dropout
            )
        else:
            self.ldam_head = None

        # Metric 추적
        self.prob = []
        self.label = []
        self.subgroups = []

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

        missing_keys, unexpected_keys = self.stage1_adapter.load_state_dict(
            adapter_state_dict, strict=strict
        )

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded Stage 1 Adapter weights ({len(adapter_state_dict)} parameters)")

        if 'epoch' in checkpoint:
            print(f"  Stage 1 trained for {checkpoint['epoch']} epochs")
        if 'best_metrics' in checkpoint:
            bm = checkpoint['best_metrics']
            print(f"  Stage 1 best: cosine_sim={bm.get('cosine_sim', 'N/A')}, "
                  f"race_acc={bm.get('race_acc', 'N/A')}")

        # Stage 1 Adapter 동결
        self._freeze_stage1_adapter()
        print("Stage 1 Adapter frozen")

        return True

    def get_clip_features(self, images):
        """CLIP visual encoder로 feature 추출 (frozen)"""
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

        with torch.no_grad():
            features = self.clip_model.encode_image(images)

        return features.float()

    def forward(self, data_dict, inference=False):
        """
        Forward pass

        Args:
            data_dict (dict): 입력 데이터 딕셔너리
                - 'image': 이미지 텐서 [B, 3, H, W]
            inference (bool): 추론 모드 여부

        Returns:
            dict: 예측 결과
        """
        images = data_dict['image']

        # 1. CLIP Features (Frozen)
        with torch.no_grad():
            clip_feat = self.get_clip_features(images)
            if self.normalize_features:
                clip_feat = F.normalize(clip_feat, dim=-1)

        # 2. Stage 1: Global Fairness Feature (Frozen)
        with torch.no_grad():
            stage1_add = self.stage1_adapter(clip_feat)
            fair_feat = clip_feat + stage1_add
            if self.normalize_features:
                fair_feat = F.normalize(fair_feat, dim=-1)

        # 3. Stage 2: Detection + Local Fairness Feature (Trainable)
        stage2_add = self.stage2_adapter(clip_feat)
        detect_feat = clip_feat + stage2_add

        # 4. Demographic-Aware Fusion (AdaIN + CA + Gate)
        fused_feat, gate, attn_weights = self.fusion(detect_feat, fair_feat)

        # 5. Feature normalization for classification
        if self.normalize_features:
            fused_feat_norm = F.normalize(fused_feat, dim=-1)
        else:
            fused_feat_norm = fused_feat

        # 6. Binary Classification
        cls_logits = self.binary_classifier(fused_feat_norm)

        # 7. Probability
        prob = torch.softmax(cls_logits, dim=1)[:, 1]

        pred_dict = {
            'clip_features': clip_feat,
            'stage1_features': fair_feat,
            'stage2_features': detect_feat,
            'fused_features': fused_feat,
            'fused_features_norm': fused_feat_norm,
            'gate': gate,
            'attn_weights': attn_weights,
            'cls': cls_logits,
            'prob': prob,
        }

        # 8. Optional LDAM Head
        if self.use_ldam_head and self.ldam_head is not None:
            ldam_logits = self.ldam_head(fused_feat_norm.detach())
            pred_dict['ldam_logits'] = ldam_logits

        # 추론 모드에서 메트릭 수집
        if inference:
            self.prob.append(pred_dict['prob'].detach().cpu())
            self.label.append(data_dict['label'].detach().cpu())
            if 'subgroup' in data_dict:
                self.subgroups.append(data_dict['subgroup'].detach().cpu())

        return pred_dict

    def get_losses(self, data_dict, pred_dict):
        """
        기본 Classification Loss (Fairness loss는 Trainer에서 별도 처리)
        """
        labels = data_dict['label']
        cls_logits = pred_dict['cls']
        cls_loss = self.loss_fn(cls_logits, labels)

        return {
            'overall': cls_loss,
            'cls': cls_loss,
        }

    def get_test_metrics(self):
        """테스트 메트릭 계산"""
        if len(self.prob) == 0 or len(self.label) == 0:
            return {'auc': 0.0, 'acc': 0.0, 'eer': 0.5, 'ap': 0.0}

        probs = torch.cat(self.prob).numpy()
        labels = torch.cat(self.label).numpy()

        self.prob = []
        self.label = []
        self.subgroups = []

        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5

        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)

        try:
            ap = average_precision_score(labels, probs)
        except Exception:
            ap = 0.0

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

    def get_fairness_metrics(self):
        """공정성 메트릭 계산"""
        if len(self.prob) == 0 or len(self.label) == 0 or len(self.subgroups) == 0:
            return None

        from utils.fairness_metrics import compute_fairness_metrics

        probs = torch.cat(self.prob).numpy()
        labels = torch.cat(self.label).numpy()
        subgroups = torch.cat(self.subgroups).numpy()

        fairness_results = compute_fairness_metrics(probs, labels, subgroups)

        if 'error' in fairness_results:
            return None

        return {
            'F_FPR': fairness_results.get('F_FPR', 0.0),
            'F_OAE': fairness_results.get('F_OAE', 0.0),
            'F_DP': fairness_results.get('F_DP', 0.0),
            'F_MEO': fairness_results.get('F_MEO', 0.0),
            'num_subgroups': fairness_results.get('num_subgroups', 0),
            'avg_fpr': fairness_results.get('avg_fpr', 0.0),
            'avg_acc': fairness_results.get('avg_acc', 0.0),
        }

    def get_trainable_params(self):
        """학습 가능한 파라미터만 반환"""
        params = []
        params.extend(self.stage2_adapter.parameters())
        params.extend(self.fusion.parameters())
        params.extend(self.binary_classifier.parameters())
        if self.ldam_head is not None:
            params.extend(self.ldam_head.parameters())
        return params

    def get_trainable_params_with_names(self):
        """학습 가능한 파라미터와 이름 반환"""
        params = []
        for name, param in self.stage2_adapter.named_parameters():
            params.append((f'stage2_adapter.{name}', param))
        for name, param in self.fusion.named_parameters():
            params.append((f'fusion.{name}', param))
        for name, param in self.binary_classifier.named_parameters():
            params.append((f'binary_classifier.{name}', param))
        if self.ldam_head is not None:
            for name, param in self.ldam_head.named_parameters():
                params.append((f'ldam_head.{name}', param))
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

        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        stage1_params = sum(p.numel() for p in self.stage1_adapter.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_adapter.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.binary_classifier.parameters())

        print(f"\nComponent Parameters:")
        print(f"  CLIP:              {clip_params:,} (frozen)")
        print(f"  Stage 1 Adapter:   {stage1_params:,} (frozen)")
        print(f"  Stage 2 Adapter:   {stage2_params:,} (trainable)")
        print(f"  Demographic Fusion:{fusion_params:,} (trainable)")
        print(f"  Classifier:        {classifier_params:,} (trainable)")

        if self.ldam_head is not None:
            ldam_params = sum(p.numel() for p in self.ldam_head.parameters())
            print(f"  LDAM Head:         {ldam_params:,} (trainable)")

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
        }
