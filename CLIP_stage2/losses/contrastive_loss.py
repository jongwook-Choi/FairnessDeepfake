"""
Feature Disentanglement Loss (Contrastive)

Stage 1 adapter feature와 Stage 2 adapter feature 간의 분리를 촉진
두 adapter가 중복 표현을 학습하지 않도록 함

방법: 두 feature 간의 cosine similarity를 낮추는 방향으로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDisentanglementLoss(nn.Module):
    """
    Stage 1 / Stage 2 Feature Disentanglement Loss

    두 adapter feature 간의 독립성을 유지하기 위한 loss

    방법:
    1. Cosine dissimilarity: cos(stage1_feat, stage2_feat)를 최소화
    2. margin 이하면 loss=0 (완전 직교일 필요 없음)
    """

    def __init__(self, margin=3.0, loss_type='cosine'):
        """
        Args:
            margin (float): Margin threshold (cosine: 0~2 범위의 거리)
                - cosine distance = 1 - cos_sim (0~2)
                - margin=3.0: 사실상 항상 loss 작동
            loss_type (str): 'cosine' or 'correlation'
        """
        super().__init__()

        self.margin = margin
        self.loss_type = loss_type

    def forward(self, stage1_feat, stage2_feat):
        """
        Disentanglement loss 계산

        Args:
            stage1_feat (torch.Tensor): Stage 1 adapter feature [B, dim]
            stage2_feat (torch.Tensor): Stage 2 adapter feature [B, dim]

        Returns:
            torch.Tensor: Disentanglement loss (scalar)
            dict: 상세 정보
        """
        if self.loss_type == 'cosine':
            return self._cosine_loss(stage1_feat, stage2_feat)
        elif self.loss_type == 'correlation':
            return self._correlation_loss(stage1_feat, stage2_feat)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _cosine_loss(self, feat1, feat2):
        """
        Cosine similarity 기반 disentanglement

        두 feature의 cosine similarity 절대값을 최소화
        """
        # L2 정규화
        feat1_norm = F.normalize(feat1, dim=-1)
        feat2_norm = F.normalize(feat2, dim=-1)

        # Cosine similarity
        cos_sim = (feat1_norm * feat2_norm).sum(dim=-1)  # [B]

        # 절대값의 평균 (양/음 상관 모두 감소)
        loss = cos_sim.abs().mean()

        info = {
            'mean_cos_sim': cos_sim.mean().item(),
            'mean_abs_cos_sim': cos_sim.abs().mean().item(),
        }

        return loss, info

    def _correlation_loss(self, feat1, feat2):
        """
        Cross-correlation 기반 disentanglement (Barlow Twins 영감)

        feature 차원 간 cross-correlation을 identity에 가깝게
        """
        # 배치 정규화
        feat1_centered = feat1 - feat1.mean(dim=0)
        feat2_centered = feat2 - feat2.mean(dim=0)

        # Cross-correlation matrix [dim, dim]
        batch_size = feat1.shape[0]
        c = (feat1_centered.T @ feat2_centered) / batch_size

        # Normalize
        std1 = feat1_centered.std(dim=0) + 1e-8
        std2 = feat2_centered.std(dim=0) + 1e-8
        c = c / (std1.unsqueeze(1) * std2.unsqueeze(0))

        # Off-diagonal elements -> 0 (독립성)
        # 대각 요소는 무시 (같은 차원 간 상관은 허용)
        mask = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
        off_diag = c[mask]

        loss = off_diag.pow(2).mean()

        info = {
            'mean_cross_corr': c.abs().mean().item(),
            'off_diag_mean': off_diag.abs().mean().item(),
        }

        return loss, info
