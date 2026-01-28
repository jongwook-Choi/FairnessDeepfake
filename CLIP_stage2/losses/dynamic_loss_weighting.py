"""
Dynamic Loss Weighting
Uncertainty-based automatic loss balancing

Reference: Kendall et al., 2018
"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"

핵심 아이디어:
- 각 task의 uncertainty (log variance)를 학습 가능한 파라미터로 설정
- Uncertainty가 높은 task는 가중치가 낮아지고, uncertainty가 낮은 task는 가중치가 높아짐
- 학습이 진행됨에 따라 자동으로 loss 균형 조절
"""

import torch
import torch.nn as nn
import math


class DynamicFairnessLossWeighting(nn.Module):
    """
    Uncertainty-based automatic loss balancing for Detection + Fairness

    L_total = (1/(2*σ_cls²)) * L_cls + (1/(2*σ_fair²)) * L_fair + log(σ_cls) + log(σ_fair)

    간소화:
    L_total = exp(-log_var_cls) * L_cls + exp(-log_var_fair) * L_fair + log_var_cls + log_var_fair

    여기서 log_var = log(σ²) 는 학습 가능한 파라미터
    """

    def __init__(self, init_log_var_cls=0.0, init_log_var_fair=0.0):
        """
        Args:
            init_log_var_cls (float): Classification loss의 초기 log variance
            init_log_var_fair (float): Fairness loss의 초기 log variance
        """
        super().__init__()

        # log(σ²) 학습 가능 파라미터
        self.log_var_cls = nn.Parameter(torch.tensor(init_log_var_cls))
        self.log_var_fair = nn.Parameter(torch.tensor(init_log_var_fair))

    def forward(self, loss_cls, loss_fair):
        """
        Dynamic weighted loss 계산

        Args:
            loss_cls (torch.Tensor): Classification loss (scalar)
            loss_fair (torch.Tensor): Fairness loss (scalar)

        Returns:
            total_loss (torch.Tensor): 가중 합산된 total loss
            lambda_fair (torch.Tensor): Fairness loss의 상대적 가중치 (모니터링용)
            info (dict): 상세 정보
        """
        # Precision (inverse variance) = exp(-log_var) = 1/σ²
        precision_cls = torch.exp(-self.log_var_cls)
        precision_fair = torch.exp(-self.log_var_fair)

        # Weighted loss + regularization term
        # L = (1/σ²) * L + log(σ) = exp(-log_var) * L + 0.5 * log_var
        weighted_cls = precision_cls * loss_cls + 0.5 * self.log_var_cls
        weighted_fair = precision_fair * loss_fair + 0.5 * self.log_var_fair

        total_loss = weighted_cls + weighted_fair

        # 상대적 가중치 (모니터링용)
        # λ_fair = precision_fair / (precision_cls + precision_fair)
        lambda_fair = precision_fair / (precision_cls + precision_fair + 1e-8)

        info = {
            'precision_cls': precision_cls.item(),
            'precision_fair': precision_fair.item(),
            'log_var_cls': self.log_var_cls.item(),
            'log_var_fair': self.log_var_fair.item(),
            'weighted_cls': weighted_cls.item(),
            'weighted_fair': weighted_fair.item(),
        }

        return total_loss, lambda_fair, info


class AdaptiveFairnessLossWeighting(nn.Module):
    """
    Adaptive Loss Weighting with warmup and clamping

    학습 초기에는 classification에 집중하고,
    점진적으로 fairness loss 가중치를 증가시킴
    """

    def __init__(self, init_lambda_fair=0.1, max_lambda_fair=1.0,
                 warmup_epochs=5, total_epochs=30):
        """
        Args:
            init_lambda_fair (float): 초기 fairness 가중치
            max_lambda_fair (float): 최대 fairness 가중치
            warmup_epochs (int): Warmup 기간 (epochs)
            total_epochs (int): 전체 학습 기간
        """
        super().__init__()

        self.init_lambda_fair = init_lambda_fair
        self.max_lambda_fair = max_lambda_fair
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # 현재 epoch 추적 (외부에서 업데이트)
        self.current_epoch = 0

    def get_lambda_fair(self, epoch=None):
        """
        현재 epoch에 따른 fairness 가중치 계산

        Args:
            epoch (int): 현재 epoch (None이면 self.current_epoch 사용)

        Returns:
            float: Fairness loss 가중치
        """
        if epoch is None:
            epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
            lambda_fair = self.init_lambda_fair + progress * (self.max_lambda_fair - self.init_lambda_fair)
        else:
            # Constant after warmup
            lambda_fair = self.max_lambda_fair

        return lambda_fair

    def forward(self, loss_cls, loss_fair, epoch=None):
        """
        Adaptive weighted loss 계산

        Args:
            loss_cls (torch.Tensor): Classification loss
            loss_fair (torch.Tensor): Fairness loss
            epoch (int): 현재 epoch

        Returns:
            total_loss (torch.Tensor): 가중 합산된 loss
            lambda_fair (float): 현재 fairness 가중치
            info (dict): 상세 정보
        """
        lambda_fair = self.get_lambda_fair(epoch)

        total_loss = loss_cls + lambda_fair * loss_fair

        info = {
            'lambda_fair': lambda_fair,
            'weighted_cls': loss_cls.item(),
            'weighted_fair': (lambda_fair * loss_fair).item(),
        }

        return total_loss, lambda_fair, info

    def set_epoch(self, epoch):
        """현재 epoch 설정"""
        self.current_epoch = epoch


class GradNormLossWeighting(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing

    Reference: Chen et al., 2018
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"

    각 task의 gradient magnitude를 동적으로 균형화
    """

    def __init__(self, num_tasks=2, alpha=1.5):
        """
        Args:
            num_tasks (int): Task 수 (cls + fairness = 2)
            alpha (float): Gradient 균형 강도 (높을수록 더 강한 균형)
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.alpha = alpha

        # 학습 가능한 task weights
        self.weights = nn.Parameter(torch.ones(num_tasks))

        # 초기 loss 저장 (relative training rate 계산용)
        self.initial_losses = None

    def forward(self, losses, shared_params=None):
        """
        GradNorm weighted loss 계산

        Args:
            losses (list): [loss_cls, loss_fair] - 각 task의 loss
            shared_params (list): 공유 파라미터 리스트 (gradient 계산용)

        Returns:
            total_loss (torch.Tensor): 가중 합산된 loss
            weights (torch.Tensor): 현재 task weights
            info (dict): 상세 정보
        """
        # Softmax로 weights 정규화 (항상 양수, 합 = num_tasks)
        normalized_weights = self.num_tasks * torch.softmax(self.weights, dim=0)

        # Weighted sum of losses
        weighted_losses = [w * l for w, l in zip(normalized_weights, losses)]
        total_loss = sum(weighted_losses)

        info = {
            'weight_cls': normalized_weights[0].item(),
            'weight_fair': normalized_weights[1].item(),
            'raw_weight_cls': self.weights[0].item(),
            'raw_weight_fair': self.weights[1].item(),
        }

        return total_loss, normalized_weights, info

    def get_weights(self):
        """현재 정규화된 weights 반환"""
        return self.num_tasks * torch.softmax(self.weights, dim=0)


class FixedLossWeighting(nn.Module):
    """
    Fixed Loss Weighting (비교 실험용)

    고정된 가중치로 loss 합산
    """

    def __init__(self, lambda_cls=1.0, lambda_fair=0.1):
        """
        Args:
            lambda_cls (float): Classification loss 가중치
            lambda_fair (float): Fairness loss 가중치
        """
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_fair = lambda_fair

    def forward(self, loss_cls, loss_fair):
        """
        Fixed weighted loss 계산

        Args:
            loss_cls (torch.Tensor): Classification loss
            loss_fair (torch.Tensor): Fairness loss

        Returns:
            total_loss (torch.Tensor): 가중 합산된 loss
            lambda_fair (float): Fairness 가중치
            info (dict): 상세 정보
        """
        total_loss = self.lambda_cls * loss_cls + self.lambda_fair * loss_fair

        # 상대적 가중치
        relative_lambda_fair = self.lambda_fair / (self.lambda_cls + self.lambda_fair)

        info = {
            'lambda_cls': self.lambda_cls,
            'lambda_fair': self.lambda_fair,
            'weighted_cls': (self.lambda_cls * loss_cls).item(),
            'weighted_fair': (self.lambda_fair * loss_fair).item(),
        }

        return total_loss, relative_lambda_fair, info


if __name__ == "__main__":
    # 테스트 코드
    print("Dynamic Loss Weighting Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 더미 loss
    loss_cls = torch.tensor(0.5, device=device, requires_grad=True)
    loss_fair = torch.tensor(0.1, device=device, requires_grad=True)

    # 1. DynamicFairnessLossWeighting
    print("\n1. DynamicFairnessLossWeighting (Uncertainty-based)")
    dynamic_weight = DynamicFairnessLossWeighting().to(device)
    total, lambda_fair, info = dynamic_weight(loss_cls, loss_fair)
    print(f"   Total loss: {total.item():.4f}")
    print(f"   Lambda fair: {lambda_fair.item():.4f}")
    print(f"   Precision cls: {info['precision_cls']:.4f}")
    print(f"   Precision fair: {info['precision_fair']:.4f}")

    # 학습 파라미터 확인
    print(f"   Trainable params: {sum(p.numel() for p in dynamic_weight.parameters())}")

    # 2. AdaptiveFairnessLossWeighting
    print("\n2. AdaptiveFairnessLossWeighting (Warmup-based)")
    adaptive_weight = AdaptiveFairnessLossWeighting(
        init_lambda_fair=0.1, max_lambda_fair=1.0,
        warmup_epochs=5, total_epochs=30
    )

    for epoch in [0, 2, 5, 10, 20]:
        total, lambda_fair, info = adaptive_weight(loss_cls.detach(), loss_fair.detach(), epoch)
        print(f"   Epoch {epoch:2d}: lambda_fair = {lambda_fair:.4f}")

    # 3. FixedLossWeighting
    print("\n3. FixedLossWeighting")
    fixed_weight = FixedLossWeighting(lambda_cls=1.0, lambda_fair=0.1)
    total, lambda_fair, info = fixed_weight(loss_cls.detach(), loss_fair.detach())
    print(f"   Total loss: {total.item():.4f}")
    print(f"   Relative lambda fair: {lambda_fair:.4f}")

    # 4. GradNormLossWeighting
    print("\n4. GradNormLossWeighting")
    gradnorm_weight = GradNormLossWeighting(num_tasks=2, alpha=1.5).to(device)
    total, weights, info = gradnorm_weight([loss_cls, loss_fair])
    print(f"   Total loss: {total.item():.4f}")
    print(f"   Weight cls: {info['weight_cls']:.4f}")
    print(f"   Weight fair: {info['weight_fair']:.4f}")
