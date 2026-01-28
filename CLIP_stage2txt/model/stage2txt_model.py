"""
Stage 2 TXT Linear Probing Model
CLIP + Additive Adapter (Stage 1_txt 가중치 로드, frozen) + Binary Classifier (trainable)

구조:
    Input Image → CLIP Visual Encoder (Frozen) → CLIP Feature (768-dim)
                                                        ↓
                                        Additive Adapter (Frozen, Stage 1_txt 가중치)
                                                        ↓
                                        Final Feature = CLIP + Additive
                                                        ↓
                                        Binary Classifier (Trainable)
                                                        ↓
                                        Real/Fake Classification

Stage1_txt와의 차이점:
- text_anchors를 체크포인트에서 로드하여 buffer로 저장
- Stage1_txt에서 학습된 Adapter 가중치 로드
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from model.clip.clip import load
from model.additive_adapter import AdditiveAdapter


class Stage2TxtModel(nn.Module):
    """
    Stage 2 TXT Linear Probing 모델

    - CLIP + Additive Adapter (Stage 1_txt 가중치 로드, frozen)
    - Text Anchors (Stage 1_txt에서 로드)
    - Binary Classifier (trainable)
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 feature_dim=768,
                 adapter_hidden_dim=512,
                 classifier_hidden_dims=[384, 192],
                 num_classes=2,
                 num_subgroups=8,
                 dropout=0.1,
                 normalize_features=True,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):
        """
        Args:
            clip_name (str): CLIP 모델 이름
            feature_dim (int): CLIP feature 차원 (ViT-L/14: 768)
            adapter_hidden_dim (int): Adapter hidden 차원
            classifier_hidden_dims (list): Binary classifier hidden 차원 리스트
            num_classes (int): 출력 클래스 수 (2: Real/Fake)
            num_subgroups (int): Subgroup 수 (8: gender × race)
            dropout (float): Dropout 비율
            normalize_features (bool): Feature normalization 여부
            device (str): Device
            clip_download_root (str): CLIP 가중치 다운로드 경로
        """
        super().__init__()

        self.device = device
        self.normalize_features = normalize_features
        self.feature_dim = feature_dim
        self.num_subgroups = num_subgroups

        # CLIP 모델 로드
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

        # CLIP 모델 동결
        self._freeze_clip()

        # Additive Adapter (Stage 1_txt에서 가중치 로드 예정)
        self.additive_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # Text Anchors Buffer (Stage 1_txt에서 로드 예정)
        # 초기값은 None, load_stage1txt_checkpoint에서 설정됨
        self.register_buffer('text_anchors', None)

        # Binary Classifier (3층 MLP: 768 → 384 → 192 → 2)
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
        """CLIP visual encoder 파라미터 완전 동결"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

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

    def load_stage1txt_checkpoint(self, checkpoint_path, strict=False):
        """
        Stage 1_txt 체크포인트에서 Additive Adapter + Text Anchors 가중치 로드

        Args:
            checkpoint_path (str): Stage 1_txt 체크포인트 경로
            strict (bool): strict 로딩 여부

        Returns:
            bool: 로드 성공 여부
        """
        print(f"Loading Stage 1_txt checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 1. Additive Adapter 가중치 로드
        adapter_state_dict = {}
        for key, value in state_dict.items():
            if 'additive_adapter' in key:
                # 'additive_adapter.' 프리픽스 제거
                new_key = key.replace('additive_adapter.', '')
                adapter_state_dict[new_key] = value

        if len(adapter_state_dict) == 0:
            print("Warning: No additive_adapter weights found in checkpoint!")
            print(f"Available keys: {list(state_dict.keys())[:10]}...")
            adapter_loaded = False
        else:
            # Additive Adapter에 가중치 로드
            missing_keys, unexpected_keys = self.additive_adapter.load_state_dict(
                adapter_state_dict, strict=strict
            )

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

            print(f"Successfully loaded Additive Adapter weights ({len(adapter_state_dict)} parameters)")
            adapter_loaded = True

        # 2. Text Anchors 로드 (Stage 1_txt 전용)
        text_anchors_loaded = False
        if 'text_anchors' in state_dict:
            self.register_buffer('text_anchors', state_dict['text_anchors'].to(self.device))
            print(f"Successfully loaded Text Anchors: shape {self.text_anchors.shape}")
            text_anchors_loaded = True
        else:
            # state_dict의 키 중에서 text_anchors 관련 키 찾기
            for key in state_dict.keys():
                if 'text_anchors' in key.lower():
                    self.register_buffer('text_anchors', state_dict[key].to(self.device))
                    print(f"Successfully loaded Text Anchors from key '{key}': shape {self.text_anchors.shape}")
                    text_anchors_loaded = True
                    break

        if not text_anchors_loaded:
            print("Warning: No text_anchors found in checkpoint!")
            print(f"Available keys containing 'anchor': {[k for k in state_dict.keys() if 'anchor' in k.lower()]}")

        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            print(f"  Stage 1_txt trained for {checkpoint['epoch']} epochs")
        if 'best_loss' in checkpoint:
            print(f"  Stage 1_txt best loss: {checkpoint['best_loss']:.4f}")

        return adapter_loaded

    def freeze_backbone(self):
        """CLIP + Additive Adapter 동결 (Linear Probing 모드)"""
        # CLIP은 이미 동결됨
        for param in self.additive_adapter.parameters():
            param.requires_grad = False
        self.additive_adapter.eval()
        print("Backbone (CLIP + Additive Adapter) frozen for Linear Probing")

    def unfreeze_adapter(self):
        """Additive Adapter 학습 가능하게 설정 (Fine-tuning 모드)"""
        for param in self.additive_adapter.parameters():
            param.requires_grad = True
        self.additive_adapter.train()
        print("Additive Adapter unfrozen for fine-tuning")

    def unfreeze_clip(self):
        """CLIP visual encoder unfreeze (Full fine-tuning 모드)"""
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        print("CLIP visual encoder unfrozen for full fine-tuning")

    def get_full_trainable_params(self):
        """CLIP + Adapter + Classifier 전체 파라미터 반환"""
        params = []
        params.extend(self.clip_model.visual.parameters())
        params.extend(self.additive_adapter.parameters())
        params.extend(self.binary_classifier.parameters())
        return params

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
                - 'additive_features': Additive adapter 출력
                - 'final_features': CLIP + Additive features
                - 'cls': Classification logits
                - 'prob': Fake 클래스 확률
        """
        images = data_dict['image']

        # 1. CLIP features 추출 (frozen)
        clip_features = self.get_clip_features(images)

        # 2. Feature normalization (optional)
        if self.normalize_features:
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

        # 3. Additive features 생성
        additive_features = self.additive_adapter(clip_features)

        # 4. Final features = CLIP + Additive
        final_features = clip_features + additive_features

        # 5. Feature normalization for final features
        if self.normalize_features:
            final_features_norm = final_features / final_features.norm(dim=-1, keepdim=True)
        else:
            final_features_norm = final_features

        # 6. Binary classification
        cls_logits = self.binary_classifier(final_features_norm)

        # 7. Probability 계산
        prob = torch.softmax(cls_logits, dim=1)[:, 1]  # Fake 클래스 확률

        pred_dict = {
            'clip_features': clip_features,
            'additive_features': additive_features,
            'final_features': final_features,
            'final_features_norm': final_features_norm,
            'cls': cls_logits,
            'prob': prob,
        }

        # Text Anchors가 있으면 similarity 추가
        if self.text_anchors is not None:
            # [batch_size, feature_dim] @ [num_subgroups, feature_dim].T -> [batch_size, num_subgroups]
            text_similarity = torch.matmul(final_features_norm, self.text_anchors.T)
            pred_dict['text_similarity'] = text_similarity

        # 추론 모드에서 메트릭 수집
        if inference:
            self.prob.append(pred_dict['prob'].detach().cpu())
            self.label.append(data_dict['label'].detach().cpu())

        return pred_dict

    def get_losses(self, data_dict, pred_dict):
        """
        Loss 계산

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
            from sklearn.metrics import roc_curve
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
        """학습 가능한 파라미터만 반환 (Binary Classifier)"""
        return self.binary_classifier.parameters()

    def get_all_trainable_params(self):
        """모든 학습 가능한 파라미터 반환 (Adapter + Classifier)"""
        params = []
        params.extend(self.additive_adapter.parameters())
        params.extend(self.binary_classifier.parameters())
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
        adapter_params = sum(p.numel() for p in self.additive_adapter.parameters())
        classifier_params = sum(p.numel() for p in self.binary_classifier.parameters())

        print(f"\nComponent Parameters:")
        print(f"  CLIP:       {clip_params:,} (frozen)")
        print(f"  Adapter:    {adapter_params:,}")
        print(f"  Classifier: {classifier_params:,}")

        # Text Anchors 정보
        if self.text_anchors is not None:
            print(f"  Text Anchors: {self.text_anchors.shape} (buffer, not trainable)")
        else:
            print(f"  Text Anchors: Not loaded")

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
        }


class Stage2TxtFineTuningModel(Stage2TxtModel):
    """
    Stage 2 TXT Fine-tuning 모델 (Adapter도 학습)

    Linear Probing과 동일하지만 Additive Adapter도 학습 가능
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Adapter는 학습 가능하게 유지
        for param in self.additive_adapter.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """CLIP만 동결 (Adapter는 학습 가능)"""
        # CLIP은 이미 동결됨
        print("CLIP frozen, Adapter trainable for Fine-tuning")


if __name__ == "__main__":
    # 테스트 코드
    print("Stage 2 TXT Model Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Stage2TxtModel(
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
    print(f"cls shape: {output['cls'].shape}")
    print(f"prob shape: {output['prob'].shape}")
