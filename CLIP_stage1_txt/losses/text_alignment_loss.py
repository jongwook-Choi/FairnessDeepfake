"""
Text-Visual Alignment Loss for CLIP Stage1 with Text

두 가지 Loss 정의:
1. TextVisualAlignmentLoss: Visual Features가 해당 subgroup의 Text Anchor에 가까워지도록
2. TextConsistencyLoss: 모든 Text Anchors 간 거리 최소화 (fairness 유도)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextVisualAlignmentLoss(nn.Module):
    """
    Visual Features가 해당 subgroup의 Text Anchor에 가까워지도록 학습

    각 샘플의 visual feature와 해당 subgroup의 text anchor 간 cosine distance 최소화

    Loss = mean(1 - cosine_similarity(visual_features, text_anchors[subgroup]))

    Args:
        temperature (float): Softmax temperature for scaling (default: 1.0)
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_features, subgroups, text_anchors):
        """
        Text-Visual Alignment Loss 계산

        Args:
            visual_features: [batch_size, feature_dim] - L2 정규화된 visual features
            subgroups: [batch_size] - 각 샘플의 subgroup ID (0-7)
            text_anchors: [num_subgroups, feature_dim] - L2 정규화된 text anchors

        Returns:
            torch.Tensor: scalar loss value
            dict: 추가 정보 (subgroup별 loss 등)
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device

        # 각 샘플에 해당하는 text anchor 선택
        # subgroups: [batch_size] -> selected_anchors: [batch_size, feature_dim]
        selected_anchors = text_anchors[subgroups]

        # Cosine similarity 계산
        # visual_features와 selected_anchors 모두 이미 L2 정규화되어 있다고 가정
        # 추가로 정규화 (안전을 위해)
        visual_features_norm = F.normalize(visual_features, p=2, dim=-1)
        selected_anchors_norm = F.normalize(selected_anchors, p=2, dim=-1)

        # Cosine similarity: [batch_size]
        cosine_sim = (visual_features_norm * selected_anchors_norm).sum(dim=-1)

        # Loss = 1 - cosine_similarity (cosine distance)
        loss = (1 - cosine_sim).mean()

        # Subgroup별 loss 계산 (분석용)
        subgroup_losses = {}
        unique_subgroups = subgroups.unique()
        for sg in unique_subgroups:
            mask = (subgroups == sg)
            if mask.sum() > 0:
                sg_loss = (1 - cosine_sim[mask]).mean()
                subgroup_losses[sg.item()] = sg_loss.item()

        info_dict = {
            'mean_cosine_sim': cosine_sim.mean().item(),
            'min_cosine_sim': cosine_sim.min().item(),
            'max_cosine_sim': cosine_sim.max().item(),
            'subgroup_losses': subgroup_losses,
        }

        return loss, info_dict


class TextConsistencyLoss(nn.Module):
    """
    모든 Text Anchors 간 거리 최소화 (fairness 유도)

    모든 text anchor 쌍 간의 cosine distance를 최소화하여
    각 subgroup의 representation이 비슷해지도록 유도

    Loss = mean(1 - cosine_similarity(anchor_i, anchor_j)) for all pairs i < j
    또는
    Loss = variance of anchor-to-global-mean distances

    Args:
        loss_type (str): 'pairwise' or 'variance'
            - 'pairwise': 모든 anchor 쌍 간 cosine distance 평균
            - 'variance': anchor들의 분산 최소화
    """

    def __init__(self, loss_type='pairwise'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, text_anchors):
        """
        Text Consistency Loss 계산

        Args:
            text_anchors: [num_subgroups, feature_dim] - L2 정규화된 text anchors

        Returns:
            torch.Tensor: scalar loss value
            dict: 추가 정보
        """
        num_subgroups = text_anchors.shape[0]

        # L2 정규화 (안전을 위해)
        text_anchors_norm = F.normalize(text_anchors, p=2, dim=-1)

        if self.loss_type == 'pairwise':
            # 모든 anchor 쌍 간 cosine similarity 계산
            # Similarity matrix: [num_subgroups, num_subgroups]
            sim_matrix = torch.mm(text_anchors_norm, text_anchors_norm.t())

            # 상삼각 행렬의 요소만 추출 (자기 자신 제외)
            # i < j인 모든 쌍
            mask = torch.triu(torch.ones(num_subgroups, num_subgroups, device=text_anchors.device), diagonal=1)
            pairwise_sims = sim_matrix[mask.bool()]

            # Loss = 1 - mean(cosine_similarity)
            # Anchor들이 더 비슷해지도록
            mean_sim = pairwise_sims.mean()
            loss = 1 - mean_sim

            info_dict = {
                'mean_pairwise_sim': mean_sim.item(),
                'min_pairwise_sim': pairwise_sims.min().item(),
                'max_pairwise_sim': pairwise_sims.max().item(),
            }

        elif self.loss_type == 'variance':
            # Global mean anchor 계산
            global_mean = text_anchors_norm.mean(dim=0)  # [feature_dim]
            global_mean = F.normalize(global_mean, p=2, dim=0)

            # 각 anchor와 global mean 간의 cosine similarity
            sims_to_mean = (text_anchors_norm * global_mean).sum(dim=-1)  # [num_subgroups]

            # Variance 최소화 (모든 anchor가 global mean에 가깝도록)
            loss = sims_to_mean.var()

            info_dict = {
                'mean_sim_to_global': sims_to_mean.mean().item(),
                'variance': loss.item(),
            }

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss, info_dict


class CombinedTextLoss(nn.Module):
    """
    Text-Visual Alignment Loss와 Text Consistency Loss를 결합

    Total = lambda_align * L_text_align + lambda_consist * L_text_consist
    """

    def __init__(self,
                 lambda_text_align=0.1,
                 lambda_text_consist=0.01,
                 consistency_loss_type='pairwise'):
        super().__init__()

        self.lambda_text_align = lambda_text_align
        self.lambda_text_consist = lambda_text_consist

        self.text_align_loss = TextVisualAlignmentLoss()
        self.text_consist_loss = TextConsistencyLoss(loss_type=consistency_loss_type)

    def forward(self, visual_features, subgroups, text_anchors):
        """
        Combined Text Loss 계산

        Args:
            visual_features: [batch_size, feature_dim]
            subgroups: [batch_size]
            text_anchors: [num_subgroups, feature_dim]

        Returns:
            dict: Loss dictionary containing individual and combined losses
        """
        loss_dict = {}

        # Text-Visual Alignment Loss
        loss_align, align_info = self.text_align_loss(visual_features, subgroups, text_anchors)
        loss_dict['text_align'] = loss_align
        loss_dict['text_align_weighted'] = self.lambda_text_align * loss_align
        loss_dict['text_align_info'] = align_info

        # Text Consistency Loss
        loss_consist, consist_info = self.text_consist_loss(text_anchors)
        loss_dict['text_consist'] = loss_consist
        loss_dict['text_consist_weighted'] = self.lambda_text_consist * loss_consist
        loss_dict['text_consist_info'] = consist_info

        # Total text loss
        loss_dict['text_total'] = (
            loss_dict['text_align_weighted'] +
            loss_dict['text_consist_weighted']
        )

        return loss_dict
