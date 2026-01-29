"""
Adaptive Instance Normalization (AdaIN) Fusion Module

fair_feat의 통계(mean/var)로 detect_feat를 재조정하여
암묵적으로 fair statistics를 detection feature에 주입

PG-FDD 논문 참조
"""

import torch
import torch.nn as nn


class AdaptiveInstanceNorm(nn.Module):
    """
    AdaIN: fair_feat에서 style 파라미터(gamma, beta)를 생성하여
    detect_feat를 재정규화

    detect_feat -> InstanceNorm -> gamma * normalized + beta
    (gamma, beta는 fair_feat에서 추출)
    """

    def __init__(self, dim=768):
        """
        Args:
            dim (int): Feature 차원
        """
        super().__init__()

        # InstanceNorm1d: detect_feat 정규화
        # [B, dim] -> [B, 1, dim] -> InstanceNorm -> [B, 1, dim] -> [B, dim]
        self.norm = nn.InstanceNorm1d(dim, affine=False)

        # fair_feat에서 style 파라미터(gamma, beta) 생성
        self.style_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2)  # gamma(dim) + beta(dim)
        )

        # 초기화: gamma=1, beta=0에 가깝게
        self._init_weights()

    def _init_weights(self):
        """Style transform 초기화 (gamma~1, beta~0)"""
        # 마지막 Linear의 weight를 작게, bias를 [1...1, 0...0]으로
        last_linear = self.style_transform[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        # bias: 앞쪽 dim은 1.0 (gamma), 뒤쪽 dim은 0.0 (beta)
        dim = last_linear.out_features // 2
        bias = torch.cat([torch.ones(dim), torch.zeros(dim)])
        last_linear.bias.data = bias

    def forward(self, detect_feat, fair_feat):
        """
        AdaIN Forward

        Args:
            detect_feat (torch.Tensor): Detection feature [B, dim]
            fair_feat (torch.Tensor): Fairness feature [B, dim]

        Returns:
            torch.Tensor: AdaIN 적용된 feature [B, dim]
        """
        # detect_feat 정규화: [B, dim] -> [B, 1, dim] -> InstanceNorm -> [B, dim]
        normalized = self.norm(detect_feat.unsqueeze(1)).squeeze(1)

        # fair_feat에서 style 파라미터 추출
        style = self.style_transform(fair_feat)  # [B, dim*2]
        gamma, beta = style.chunk(2, dim=-1)     # [B, dim], [B, dim]

        # AdaIN: fair의 통계로 detect를 재조정
        return gamma * normalized + beta


class GatedAdaINFusion(nn.Module):
    """
    AdaIN + Gated Fusion

    AdaIN 결과와 원본 detect_feat를 gate로 혼합
    gate = sigmoid(W * [adain_out; detect_feat])
    output = gate * adain_out + (1-gate) * detect_feat
    """

    def __init__(self, dim=768, gate_init_bias=0.0):
        """
        Args:
            dim (int): Feature 차원
            gate_init_bias (float): Gate 초기 bias
                0.0 -> sigmoid(0.0) = 0.5 (균등 혼합)
        """
        super().__init__()

        self.adain = AdaptiveInstanceNorm(dim=dim)

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Gate 초기 bias 설정
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

    def forward(self, detect_feat, fair_feat):
        """
        Gated AdaIN Fusion Forward

        Args:
            detect_feat (torch.Tensor): Detection feature [B, dim]
            fair_feat (torch.Tensor): Fairness feature [B, dim]

        Returns:
            adain_out (torch.Tensor): AdaIN 출력 [B, dim]
            gate (torch.Tensor): Gate 값 [B, 1]
        """
        # AdaIN 적용
        adain_out = self.adain(detect_feat, fair_feat)

        return adain_out
