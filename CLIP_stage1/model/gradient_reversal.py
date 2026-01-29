"""
Gradient Reversal Layer (GRL)
DANN(Ganin et al., 2016) 기반의 adversarial debiasing을 위한 Gradient Reversal Layer

Forward: x를 그대로 통과
Backward: gradient에 -lambda_grl을 곱하여 방향 반전

결과: adapter가 classifier를 속이는 방향으로 학습 -> feature에서 인구통계 정보 제거
"""

import math
import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Function (autograd 기반)"""

    @staticmethod
    def forward(ctx, x, lambda_grl):
        """
        Forward: 입력을 그대로 통과

        Args:
            ctx: autograd context
            x (torch.Tensor): 입력 텐서
            lambda_grl (float): gradient reversal 강도

        Returns:
            torch.Tensor: 입력과 동일한 텐서
        """
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: gradient 방향 반전

        Args:
            ctx: autograd context
            grad_output (torch.Tensor): 상위 레이어에서 전달된 gradient

        Returns:
            tuple: (반전된 gradient, None)
        """
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    """
    Gradient Reversal Layer 모듈

    사용 방법:
        grl = GradientReversalLayer(lambda_grl=1.0)
        reversed_feat = grl(features)  # forward: 동일, backward: gradient 반전
    """

    def __init__(self, lambda_grl=1.0):
        """
        Args:
            lambda_grl (float): gradient reversal 강도 (0 ~ 1)
        """
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            torch.Tensor: 입력과 동일 (backward 시 gradient 반전)
        """
        return GradientReversalFunction.apply(x, self.lambda_grl)

    def set_lambda(self, lambda_grl):
        """lambda_grl 값 업데이트"""
        self.lambda_grl = lambda_grl

    def extra_repr(self):
        return f'lambda_grl={self.lambda_grl}'


class GRLScheduler:
    """
    GRL lambda scheduling (DANN 논문 방식)

    epoch 진행에 따라 lambda_grl을 점진적으로 증가
    p = epoch / total_epochs
    lambda_grl = 2.0 / (1.0 + exp(-gamma * p)) - 1.0  (0 -> 1로 증가)
    """

    def __init__(self, grl_layer, total_epochs, gamma=10.0,
                 initial_lambda=0.0, max_lambda=1.0, schedule='dann'):
        """
        Args:
            grl_layer (GradientReversalLayer): GRL 레이어
            total_epochs (int): 전체 학습 에포크 수
            gamma (float): scheduling 곡선의 가파른 정도 (높을수록 급격)
            initial_lambda (float): 초기 lambda 값
            max_lambda (float): 최대 lambda 값
            schedule (str): scheduling 방식
                - 'dann': DANN 논문 sigmoid schedule
                - 'linear': 선형 증가
                - 'step': epoch threshold에서 step 증가
        """
        self.grl_layer = grl_layer
        self.total_epochs = total_epochs
        self.gamma = gamma
        self.initial_lambda = initial_lambda
        self.max_lambda = max_lambda
        self.schedule = schedule

    def step(self, epoch):
        """
        현재 epoch에 따라 lambda_grl 업데이트

        Args:
            epoch (int): 현재 epoch (1-based)

        Returns:
            float: 업데이트된 lambda_grl 값
        """
        p = epoch / self.total_epochs  # 0 -> 1 진행률

        if self.schedule == 'dann':
            # DANN sigmoid schedule: 0 -> max_lambda
            lambda_grl = 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0
            lambda_grl = self.initial_lambda + (self.max_lambda - self.initial_lambda) * lambda_grl

        elif self.schedule == 'linear':
            # 선형 증가: initial -> max
            lambda_grl = self.initial_lambda + (self.max_lambda - self.initial_lambda) * p

        elif self.schedule == 'step':
            # Step schedule: 절반 이후 max_lambda
            if p < 0.5:
                lambda_grl = self.initial_lambda
            else:
                lambda_grl = self.max_lambda

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

        # lambda 범위 제한
        lambda_grl = max(self.initial_lambda, min(self.max_lambda, lambda_grl))

        self.grl_layer.set_lambda(lambda_grl)
        return lambda_grl
