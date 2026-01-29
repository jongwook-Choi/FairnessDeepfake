"""
Fairness Adapter 모델 (v2 - GRL 통합)
CLIP frozen + Additive Adapter + GRL + Race/Gender Classifiers

GRL(Gradient Reversal Layer)을 통해 classifier의 gradient가 반전되어
adapter가 인구통계 정보를 제거하는 방향으로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip.clip import load
from model.additive_adapter import AdditiveAdapter
from model.gradient_reversal import GradientReversalLayer


class FairnessAdapter(nn.Module):
    """
    Fairness를 위한 CLIP Adapter 모델 (GRL 기반 Adversarial Debiasing)

    구조:
        Input Image -> CLIP Visual Encoder (Frozen) -> CLIP Feature (768-dim)
                                                            |
                                                    Additive Adapter (Learnable)
                                                            |
                                                    debiased_feat = CLIP + Additive
                                                            |
                    +-------------------+-------------------+-------------------+
                    |                   |                   |                   |
                    v                   v                   v                   v
              GRL(lambda)        Similarity Loss     Sinkhorn Fairness    Pairwise Sinkhorn
                    |           cos(clip, debiased)
              +-----+-----+
              |           |
        Race CLF(4)   Gender CLF(2)
              |           |
         CE Loss       CE Loss
    (reversed gradient -> adapter)
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 adapter_hidden_dim=512,
                 feature_dim=768,
                 num_races=4,
                 num_genders=2,
                 dropout=0.1,
                 normalize_features=True,
                 use_grl=True,
                 initial_lambda_grl=0.0,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):
        """
        Args:
            clip_name (str): CLIP 모델 이름 ("ViT-L/14", "ViT-B/16" 등)
            adapter_hidden_dim (int): Adapter hidden 차원
            feature_dim (int): CLIP feature 차원 (ViT-L/14: 768, ViT-B/16: 512)
            num_races (int): 인종 클래스 수 (Asian, White, Black, Other = 4)
            num_genders (int): 성별 클래스 수 (Male, Female = 2)
            dropout (float): dropout 비율
            normalize_features (bool): feature normalization 여부
            use_grl (bool): GRL 사용 여부 (True: adversarial, False: 기존 방식)
            initial_lambda_grl (float): GRL 초기 lambda 값
            device (str): device
            clip_download_root (str): CLIP 가중치 다운로드 경로
        """
        super().__init__()

        self.device = device
        self.normalize_features = normalize_features
        self.feature_dim = feature_dim
        self.use_grl = use_grl

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

        # Additive Adapter (학습 가능)
        self.additive_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # GRL (Gradient Reversal Layer)
        if self.use_grl:
            self.grl = GradientReversalLayer(lambda_grl=initial_lambda_grl)
        else:
            self.grl = None

        # Race Classifier (4-class: Asian, White, Black, Other)
        self.race_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_races)
        )

        # Gender Classifier (2-class: Male, Female)
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_genders)
        )

        # 가중치 초기화
        self._init_classifiers()

    def _freeze_clip(self):
        """CLIP visual encoder 파라미터 완전 동결"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def _init_classifiers(self):
        """Classifier 가중치 초기화"""
        for classifier in [self.race_classifier, self.gender_classifier]:
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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
                - 'final_features': CLIP + Additive features (debiased)
                - 'final_features_norm': L2 정규화된 debiased features
                - 'race_logits': Race classification logits (GRL 통과 후)
                - 'gender_logits': Gender classification logits (GRL 통과 후)
        """
        images = data_dict['image']

        # 1. CLIP features 추출 (frozen)
        clip_features = self.get_clip_features(images)

        # 2. Feature normalization (optional)
        if self.normalize_features:
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

        # 3. Additive features 생성
        additive_features = self.additive_adapter(clip_features)

        # 4. Debiased features = CLIP + Additive
        final_features = clip_features + additive_features

        # 5. Feature normalization for final features
        if self.normalize_features:
            final_features_norm = final_features / final_features.norm(dim=-1, keepdim=True)
        else:
            final_features_norm = final_features

        # 6. Race/Gender classification (GRL 통과)
        if self.use_grl and self.grl is not None:
            # GRL: forward는 동일, backward에서 gradient 반전
            reversed_features = self.grl(final_features_norm)
            race_logits = self.race_classifier(reversed_features)
            gender_logits = self.gender_classifier(reversed_features)
        else:
            # GRL 없이 직접 분류 (기존 방식)
            race_logits = self.race_classifier(final_features_norm)
            gender_logits = self.gender_classifier(final_features_norm)

        pred_dict = {
            'clip_features': clip_features,
            'additive_features': additive_features,
            'final_features': final_features,
            'final_features_norm': final_features_norm,
            'race_logits': race_logits,
            'gender_logits': gender_logits,
        }

        return pred_dict

    def get_trainable_params(self):
        """학습 가능한 파라미터만 반환 (Adapter + Classifiers)"""
        params = []
        params.extend(self.additive_adapter.parameters())
        params.extend(self.race_classifier.parameters())
        params.extend(self.gender_classifier.parameters())
        return params

    def get_adapter_params(self):
        """Adapter 파라미터만 반환"""
        return list(self.additive_adapter.parameters())

    def get_classifier_params(self):
        """Classifier 파라미터만 반환"""
        params = []
        params.extend(self.race_classifier.parameters())
        params.extend(self.gender_classifier.parameters())
        return params


class FairnessAdapterWithBinaryClassifier(FairnessAdapter):
    """
    Binary classification (Real/Fake)을 추가한 Fairness Adapter
    Stage2에서 사용 예정
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 adapter_hidden_dim=512,
                 feature_dim=768,
                 num_races=4,
                 num_genders=2,
                 num_classes=2,
                 dropout=0.1,
                 normalize_features=True,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):

        super().__init__(
            clip_name=clip_name,
            adapter_hidden_dim=adapter_hidden_dim,
            feature_dim=feature_dim,
            num_races=num_races,
            num_genders=num_genders,
            dropout=dropout,
            normalize_features=normalize_features,
            use_grl=False,  # Binary classifier 모드에서는 GRL 사용하지 않음
            device=device,
            clip_download_root=clip_download_root
        )

        # Binary Classifier (Real/Fake)
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 가중치 초기화
        for module in self.binary_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data_dict, inference=False):
        """Forward pass with binary classification"""
        pred_dict = super().forward(data_dict, inference)

        # Binary classification
        final_features_norm = pred_dict['final_features_norm']
        binary_logits = self.binary_classifier(final_features_norm)

        pred_dict['cls'] = binary_logits
        pred_dict['prob'] = torch.softmax(binary_logits, dim=1)[:, 1]

        return pred_dict
