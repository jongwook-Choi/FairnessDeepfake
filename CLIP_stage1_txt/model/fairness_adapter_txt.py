"""
Fairness Adapter with Text Encoder
CLIP frozen + Text Encoder (frozen) + Additive Adapter + Race/Gender Classifiers

Text Features를 Fairness Anchor로 활용:
- 각 demographic subgroup(8개)에 대한 Text Prompt를 생성
- Text Features를 "fair anchor"로 사용
- Visual Features가 해당 subgroup의 Text Anchor에 가까워지도록 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip.clip import load, tokenize
from model.additive_adapter import AdditiveAdapter
from utils.prompt_templates import get_subgroup_prompts, SUBGROUP_NAMES


class FairnessAdapterWithText(nn.Module):
    """
    Text Encoder를 활용한 Fairness Adapter 모델

    구조:
                        CLIP Text Encoder (Frozen)
                                ↓
            Subgroup Prompts → Text Features (768) → Text Anchors [8, 768] (Cached)
                                                            ↓
        Input Image → CLIP Visual Encoder (Frozen) → CLIP Feature (768)
                                                        ↓
                                            Additive Adapter (Learnable)
                                                        ↓
                                            Final Feature (768)
                                                        ↓
            ┌───────────────┬───────────────┬───────────────┬───────────────┐
            ↓               ↓               ↓               ↓               ↓
      Race Classifier  Gender Classifier  Fairness Loss  Text-Visual     Text Consistency
          (4)              (2)           (Sinkhorn)     Alignment Loss       Loss
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 adapter_hidden_dim=512,
                 feature_dim=768,
                 num_races=4,
                 num_genders=2,
                 num_subgroups=8,
                 num_prompts_per_subgroup=6,
                 dropout=0.1,
                 normalize_features=True,
                 device='cuda',
                 clip_download_root='/data/cuixinjie/weights'):
        """
        Args:
            clip_name (str): CLIP 모델 이름 ("ViT-L/14", "ViT-B/16" 등)
            adapter_hidden_dim (int): Adapter hidden 차원
            feature_dim (int): CLIP feature 차원 (ViT-L/14: 768, ViT-B/16: 512)
            num_races (int): 인종 클래스 수 (Asian, Black, White, Other = 4)
            num_genders (int): 성별 클래스 수 (Male, Female = 2)
            num_subgroups (int): Subgroup 수 (8)
            num_prompts_per_subgroup (int): 각 subgroup당 프롬프트 수
            dropout (float): dropout 비율
            normalize_features (bool): feature normalization 여부
            device (str): device
            clip_download_root (str): CLIP 가중치 다운로드 경로
        """
        super().__init__()

        self.device = device
        self.normalize_features = normalize_features
        self.feature_dim = feature_dim
        self.num_subgroups = num_subgroups
        self.num_prompts_per_subgroup = num_prompts_per_subgroup

        # CLIP 모델 로드 (Visual + Text Encoder 모두 유지)
        self.clip_model, self.preprocess = load(
            clip_name,
            device=device,
            download_root=clip_download_root
        )

        # NOTE: Text Encoder가 필요하므로 transformer 삭제하지 않음!
        # 기존 fairness_adapter.py의 line 65-66 코드 제거됨

        # Feature 차원 자동 감지
        if feature_dim is None:
            self.feature_dim = self.clip_model.visual.output_dim

        # CLIP 모델 완전 동결 (Visual + Text Encoder 모두)
        self._freeze_clip()

        # Text Anchors 초기화 (나중에 캐싱됨)
        # register_buffer로 저장하여 학습 시 gradient 계산 제외
        self.register_buffer('text_anchors', None)
        self._text_anchors_initialized = False

        # Additive Adapter (학습 가능)
        self.additive_adapter = AdditiveAdapter(
            input_dim=self.feature_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=self.feature_dim,
            dropout=dropout
        )

        # Race Classifier (4-class: Asian, Black, White, Other)
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

        print(f"[FairnessAdapterWithText] Initialized")
        print(f"  CLIP model: {clip_name}")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Num subgroups: {num_subgroups}")
        print(f"  Prompts per subgroup: {num_prompts_per_subgroup}")

    def _freeze_clip(self):
        """CLIP 전체 (visual + text encoder) 파라미터 완전 동결"""
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

    @torch.no_grad()
    def _compute_text_anchors(self):
        """
        각 subgroup의 Text Anchor 계산 및 캐싱

        8개 subgroup에 대해:
        1. 각 subgroup의 프롬프트들을 Text Encoder로 인코딩
        2. 프롬프트별 feature를 평균하여 anchor 생성
        3. L2 정규화 적용

        Returns:
            torch.Tensor: [num_subgroups, feature_dim] 형태의 Text Anchors
        """
        # 프롬프트 가져오기
        subgroup_prompts = get_subgroup_prompts(self.num_prompts_per_subgroup)

        anchors = []
        for subgroup_id in range(self.num_subgroups):
            prompts = subgroup_prompts[subgroup_id]

            # 토큰화
            tokens = tokenize(prompts).to(self.device)

            # Text Encoder로 feature 추출
            text_features = self.clip_model.encode_text(tokens)  # [num_prompts, feature_dim]

            # L2 정규화
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 프롬프트별 feature 평균
            anchor = text_features.mean(dim=0)  # [feature_dim]

            # 평균 후 다시 L2 정규화
            anchor = anchor / anchor.norm()

            anchors.append(anchor)

            print(f"  Subgroup {subgroup_id} ({SUBGROUP_NAMES[subgroup_id]}): "
                  f"{len(prompts)} prompts -> anchor shape {anchor.shape}")

        # [num_subgroups, feature_dim]
        text_anchors = torch.stack(anchors, dim=0)

        return text_anchors.float()

    def initialize_text_anchors(self):
        """
        Text Anchors 초기화 (최초 1회만 실행)
        모델이 올바른 device로 이동한 후 호출해야 함
        """
        if self._text_anchors_initialized:
            print("[FairnessAdapterWithText] Text anchors already initialized")
            return

        print("[FairnessAdapterWithText] Computing text anchors...")
        text_anchors = self._compute_text_anchors()

        # register_buffer로 저장 (gradient 계산 제외, 체크포인트에 포함)
        self.register_buffer('text_anchors', text_anchors)
        self._text_anchors_initialized = True

        print(f"[FairnessAdapterWithText] Text anchors cached: {self.text_anchors.shape}")

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
                - 'final_features_norm': L2 정규화된 final features
                - 'race_logits': Race classification logits
                - 'gender_logits': Gender classification logits
                - 'text_anchors': Text anchors [num_subgroups, feature_dim]
        """
        # Text anchors 초기화 확인
        if self.text_anchors is None:
            self.initialize_text_anchors()

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

        # 6. Race/Gender classification
        race_logits = self.race_classifier(final_features_norm)
        gender_logits = self.gender_classifier(final_features_norm)

        pred_dict = {
            'clip_features': clip_features,
            'additive_features': additive_features,
            'final_features': final_features,
            'final_features_norm': final_features_norm,
            'race_logits': race_logits,
            'gender_logits': gender_logits,
            'text_anchors': self.text_anchors,  # [num_subgroups, feature_dim]
        }

        return pred_dict

    def get_trainable_params(self):
        """학습 가능한 파라미터만 반환"""
        params = []
        params.extend(self.additive_adapter.parameters())
        params.extend(self.race_classifier.parameters())
        params.extend(self.gender_classifier.parameters())
        return params

    def get_adapter_params(self):
        """Adapter 파라미터만 반환"""
        return self.additive_adapter.parameters()

    def get_classifier_params(self):
        """Classifier 파라미터만 반환"""
        params = []
        params.extend(self.race_classifier.parameters())
        params.extend(self.gender_classifier.parameters())
        return params

    def get_text_anchor_similarity(self, visual_features, subgroup_ids):
        """
        Visual features와 해당 subgroup의 Text anchor 간 cosine similarity 계산

        Args:
            visual_features: [batch_size, feature_dim]
            subgroup_ids: [batch_size] - 각 샘플의 subgroup ID

        Returns:
            torch.Tensor: [batch_size] - cosine similarity 값
        """
        if self.text_anchors is None:
            raise RuntimeError("Text anchors not initialized. Call initialize_text_anchors() first.")

        # 각 샘플에 해당하는 text anchor 선택
        selected_anchors = self.text_anchors[subgroup_ids]  # [batch_size, feature_dim]

        # L2 정규화
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarities = (visual_features_norm * selected_anchors).sum(dim=-1)  # [batch_size]

        return similarities

    def print_text_anchor_info(self):
        """Text anchor 정보 출력"""
        if self.text_anchors is None:
            print("[FairnessAdapterWithText] Text anchors not initialized")
            return

        print("\n[Text Anchor Information]")
        print(f"  Shape: {self.text_anchors.shape}")
        print(f"  Device: {self.text_anchors.device}")
        print(f"  Dtype: {self.text_anchors.dtype}")

        # 각 anchor 간 cosine similarity 계산
        print("\n  Inter-anchor Cosine Similarity:")
        for i in range(self.num_subgroups):
            for j in range(i + 1, self.num_subgroups):
                sim = (self.text_anchors[i] * self.text_anchors[j]).sum().item()
                print(f"    {SUBGROUP_NAMES[i]} <-> {SUBGROUP_NAMES[j]}: {sim:.4f}")


class FairnessAdapterWithTextBinaryClassifier(FairnessAdapterWithText):
    """
    Binary classification (Real/Fake)을 추가한 Fairness Adapter with Text
    Stage2에서 사용 예정
    """

    def __init__(self,
                 clip_name="ViT-L/14",
                 adapter_hidden_dim=512,
                 feature_dim=768,
                 num_races=4,
                 num_genders=2,
                 num_subgroups=8,
                 num_prompts_per_subgroup=6,
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
            num_subgroups=num_subgroups,
            num_prompts_per_subgroup=num_prompts_per_subgroup,
            dropout=dropout,
            normalize_features=normalize_features,
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
