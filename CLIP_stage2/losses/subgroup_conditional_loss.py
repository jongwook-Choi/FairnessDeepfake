"""
Subgroup-Conditional Detection Loss (Bi-level CVaR)

Stage 2 Local Fairness: Prediction-level 공정성

방법:
1. Per-sample CE loss 계산 (reduction='none')
2. 각 subgroup별로 loss를 집계
3. Worst-case subgroup에 더 큰 가중치 부여 (CVaR)
4. Real/Fake 클래스 별로 별도 적용 (Per-class fairness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgroupConditionalDetectionLoss(nn.Module):
    """
    Bi-level CVaR 기반 Subgroup-Conditional Detection Loss

    Stage 1(Global): Feature 분포가 subgroup 간 동일해지도록 (Sinkhorn + GRL)
    Stage 2(Local): Detection 성능이 subgroup 간 동일해지도록 (CVaR + Per-class)

    Inner CVaR: subgroup 내 worst-case 샘플에 집중
    Outer CVaR: subgroup 간 worst-case에 집중
    """

    def __init__(self, num_subgroups=8, inner_alpha=0.9, outer_alpha=0.5,
                 label_smoothing=0.1):
        """
        Args:
            num_subgroups (int): Subgroup 수 (default: 8 = 2 gender x 4 race)
            inner_alpha (float): Subgroup 내 worst-case 비율 (0.9 -> top 10%)
            outer_alpha (float): Subgroup 간 worst-case 비율 (0.5 -> top 50%)
            label_smoothing (float): Label smoothing
        """
        super().__init__()

        self.num_subgroups = num_subgroups
        self.inner_alpha = inner_alpha
        self.outer_alpha = outer_alpha
        self.ce_loss = nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=label_smoothing
        )

    def forward(self, logits, labels, subgroups):
        """
        Bi-level CVaR loss 계산

        Args:
            logits (torch.Tensor): 모델 출력 [B, num_classes]
            labels (torch.Tensor): Ground truth [B]
            subgroups (torch.Tensor): Subgroup ID [B]

        Returns:
            torch.Tensor: CVaR loss (scalar)
            dict: 상세 정보
        """
        # 1. Per-sample loss
        per_sample_loss = self.ce_loss(logits, labels)

        # 2. Per-class, Per-subgroup CVaR
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        valid_classes = 0
        loss_info = {'class_losses': {}, 'subgroup_losses': {}}

        for class_val in [0, 1]:  # Real(0), Fake(1)
            class_name = 'real' if class_val == 0 else 'fake'
            class_mask = (labels == class_val)

            if class_mask.sum() < 2:
                continue

            class_sg_losses = []
            class_sg_ids = []

            for sg_id in range(self.num_subgroups):
                sg_mask = class_mask & (subgroups == sg_id)
                if sg_mask.sum() >= 2:
                    sg_losses = per_sample_loss[sg_mask]
                    # Inner CVaR: subgroup 내 worst-case
                    inner_cvar = self._smooth_cvar(sg_losses, self.inner_alpha)
                    class_sg_losses.append(inner_cvar)
                    class_sg_ids.append(sg_id)
                    loss_info['subgroup_losses'][f'{class_name}_sg{sg_id}'] = inner_cvar.item()

            if len(class_sg_losses) > 0:
                stacked = torch.stack(class_sg_losses)
                # Outer CVaR: subgroup 간 worst-case
                outer_cvar = self._smooth_cvar(stacked, self.outer_alpha)
                total_loss = total_loss + outer_cvar
                valid_classes += 1
                loss_info['class_losses'][class_name] = outer_cvar.item()

        # 두 클래스 평균
        if valid_classes > 0:
            total_loss = total_loss / valid_classes

        return total_loss, loss_info

    def _smooth_cvar(self, losses, alpha):
        """
        Smooth CVaR (미분 가능한 soft-CVaR)

        alpha=0.9: 상위 10%의 loss에 집중
        alpha=0.5: 상위 50%의 loss에 집중

        Args:
            losses (torch.Tensor): 1D loss 텐서
            alpha (float): CVaR level (0 < alpha < 1)

        Returns:
            torch.Tensor: Smooth CVaR 값 (scalar)
        """
        sorted_losses, _ = torch.sort(losses, descending=True)
        k = max(1, int((1 - alpha) * len(sorted_losses)))
        return sorted_losses[:k].mean()
