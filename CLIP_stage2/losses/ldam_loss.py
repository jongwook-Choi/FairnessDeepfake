"""
LDAM (Label-Distribution-Aware Margin) Loss

PG-FDD 논문의 bal_loss.py 참조
소수 subgroup에 더 큰 margin을 부여하여 빈도 불균형 보정

margin = C / n^(1/4)  (n: subgroup 빈도, C: max_margin)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """
    LDAM Loss for Subgroup-Aware Margin

    소수 subgroup에 더 큰 margin -> 빈도 불균형 보정
    margin = max_margin / sqrt(sqrt(n_k))  (n_k: subgroup k의 샘플 수)

    Logit adjustment:
        adjusted_logit[k] = logit[k] - margin[k]  (for target class k)
    """

    def __init__(self, cls_num_list, max_margin=0.5, ldam_s=30.0,
                 label_smoothing=0.0):
        """
        Args:
            cls_num_list (list): 각 subgroup의 샘플 수 리스트
                예: [2475, 25443, 1468, 4163, 8013, 31281, 1111, 2185]
            max_margin (float): 최대 margin 값 (default: 0.5)
            ldam_s (float): Loss scaling factor (default: 30.0)
            label_smoothing (float): Label smoothing (default: 0.0)
        """
        super().__init__()

        self.ldam_s = ldam_s
        self.label_smoothing = label_smoothing

        # Margin 계산: m_k = C / n_k^(1/4)
        cls_num_array = np.array(cls_num_list, dtype=np.float32)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_array))
        m_list = m_list * (max_margin / np.max(m_list))

        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.num_classes = len(cls_num_list)

        print(f"[LDAMLoss] Margins per subgroup:")
        for i, (n, m) in enumerate(zip(cls_num_list, m_list)):
            print(f"  Subgroup {i}: n={n:,}, margin={m:.4f}")

    def forward(self, logits, targets):
        """
        LDAM Loss 계산

        Args:
            logits (torch.Tensor): 모델 출력 logits [B, num_classes]
            targets (torch.Tensor): Ground truth labels [B]

        Returns:
            torch.Tensor: LDAM loss (scalar)
        """
        # One-hot encoding
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)

        # 타겟 클래스에 대한 margin 적용
        batch_m = self.m_list[targets]  # [B]
        x_m = logits - batch_m.unsqueeze(1) * index.float()

        # Margin이 적용된 logits로 교체 (타겟 클래스만)
        output = torch.where(index, x_m, logits)

        # Scaled CE loss
        if self.label_smoothing > 0:
            loss = F.cross_entropy(
                self.ldam_s * output, targets,
                label_smoothing=self.label_smoothing
            )
        else:
            loss = F.cross_entropy(self.ldam_s * output, targets)

        return loss


class LDAMFairnessHead(nn.Module):
    """
    LDAM Auxiliary Fairness Head

    8-class subgroup 분류 head + LDAM loss
    Config에서 enable/disable 가능

    Fairness through Awareness:
    - 인구통계를 인식하여 불균형을 보정
    - Stage 1(Ignorance)과 상호보완적
    """

    def __init__(self, feature_dim=768, num_subgroups=8,
                 cls_num_list=None, max_margin=0.5, ldam_s=30.0,
                 dropout=0.2):
        """
        Args:
            feature_dim (int): 입력 feature 차원
            num_subgroups (int): Subgroup 수
            cls_num_list (list): Subgroup별 샘플 수
            max_margin (float): LDAM 최대 margin
            ldam_s (float): LDAM scaling factor
            dropout (float): Dropout 비율
        """
        super().__init__()

        self.num_subgroups = num_subgroups

        # Subgroup 분류 head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_subgroups)
        )

        # LDAM loss
        if cls_num_list is not None:
            self.ldam_loss = LDAMLoss(
                cls_num_list=cls_num_list,
                max_margin=max_margin,
                ldam_s=ldam_s
            )
        else:
            # cls_num_list가 없으면 균등 분포 가정
            default_list = [1000] * num_subgroups
            self.ldam_loss = LDAMLoss(
                cls_num_list=default_list,
                max_margin=max_margin,
                ldam_s=ldam_s
            )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Head 가중치 초기화"""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features):
        """
        Forward pass

        Args:
            features (torch.Tensor): 입력 feature [B, feature_dim]

        Returns:
            torch.Tensor: Subgroup logits [B, num_subgroups]
        """
        return self.head(features)

    def compute_loss(self, features, subgroup_labels):
        """
        LDAM loss 계산

        Args:
            features (torch.Tensor): 입력 feature [B, feature_dim]
            subgroup_labels (torch.Tensor): Subgroup labels [B]

        Returns:
            torch.Tensor: LDAM loss (scalar)
        """
        logits = self.forward(features)
        return self.ldam_loss(logits, subgroup_labels)
