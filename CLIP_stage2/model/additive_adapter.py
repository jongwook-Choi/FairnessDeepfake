"""
Additive Adapter 모듈
CLIP feature에 더해질 additive feature를 생성하는 MLP
"""

import torch
import torch.nn as nn


class AdditiveAdapter(nn.Module):
    """
    Additive feature를 생성하는 MLP 기반 어댑터

    CLIP feature와 동일한 차원의 additive feature를 생성하여 더함으로써
    subgroup별 bias를 제거하는 역할을 수행

    Architecture:
        Input (768-dim) → Linear → ReLU → Linear → ReLU → Linear → Output (768-dim)
    """

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768, dropout=0.1):
        """
        Args:
            input_dim (int): 입력 feature 차원 (CLIP ViT-L/14: 768)
            hidden_dim (int): hidden layer 차원
            output_dim (int): 출력 feature 차원 (CLIP feature와 동일)
            dropout (float): dropout 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # MLP 레이어 정의
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 가중치 초기화 - 초기에 작은 값으로 시작
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 - 작은 값으로 시작하여 초기에 원본 CLIP feature에 가깝게 유지"""
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 마지막 레이어는 더 작은 값으로 초기화 (초기 additive feature가 작도록)
        last_linear = self.adapter[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CLIP feature [batch_size, input_dim]

        Returns:
            torch.Tensor: Additive feature [batch_size, output_dim]
        """
        return self.adapter(x)


class ResidualAdditiveAdapter(nn.Module):
    """
    Residual 연결을 포함한 Additive Adapter

    더 깊은 네트워크 학습을 위해 residual connection 추가
    """

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768,
                 num_blocks=2, dropout=0.1):
        """
        Args:
            input_dim (int): 입력 feature 차원
            hidden_dim (int): hidden layer 차원
            output_dim (int): 출력 feature 차원
            num_blocks (int): residual block 수
            dropout (float): dropout 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 입력 프로젝션 (input_dim != hidden_dim일 경우)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # 출력 프로젝션
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        # 출력 프로젝션은 작은 값으로 초기화
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CLIP feature [batch_size, input_dim]

        Returns:
            torch.Tensor: Additive feature [batch_size, output_dim]
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output_proj(x)


class ResidualBlock(nn.Module):
    """단일 Residual Block"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))
