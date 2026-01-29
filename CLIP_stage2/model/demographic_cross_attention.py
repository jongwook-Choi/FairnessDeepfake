"""
Demographic-Aware Multi-Head Cross-Attention Module

인구통계 속성별 dedicated head로 구성된 cross-attention.
fair_feat을 여러 "view"로 분할하여 seq_len > 1 확보 -> attention이 실질적으로 작동

핵심: seq_len=1 문제 해결
- 기존: [B, 1, 768] query -> [B, 1, 768] key/value -> trivial attention weight 1.0
- 신규: [B, 1, 768] query -> [B, K, 768] key/value (K개의 fair view)
"""

import torch
import torch.nn as nn


class DemographicAwareCrossAttention(nn.Module):
    """
    인구통계 속성별 dedicated view로 구성된 cross-attention.

    fair_feat을 여러 view로 확장하여 seq_len > 1 확보:
    - Gender view: 성별 관련 fairness 측면
    - Race view: 인종 관련 fairness 측면
    - General view 1: 일반적인 fairness 측면
    - General view 2: 상호작용 fairness 측면

    총 4 views -> seq_len=4로 attention이 의미있게 동작
    """

    def __init__(self, dim=768, num_views=4, num_heads=8, dropout=0.1):
        """
        Args:
            dim (int): Feature 차원
            num_views (int): Fair feature view 수 (seq_len 결정)
            num_heads (int): Multi-head attention의 head 수
            dropout (float): Dropout 비율
        """
        super().__init__()

        self.dim = dim
        self.num_views = num_views
        self.num_heads = num_heads

        # fair_feat을 여러 view로 확장하는 projection
        # 각 view는 fairness의 다른 측면을 캡처
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_views)
        ])

        # Multi-head cross-attention (seq_len = num_views)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-attention processing
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

        # View projection 초기화
        self._init_view_projections()

    def _init_view_projections(self):
        """View projection 초기화 - 각 view가 다른 측면을 캡처하도록"""
        for i, proj in enumerate(self.view_projections):
            for module in proj.modules():
                if isinstance(module, nn.Linear):
                    # 각 view에 약간 다른 초기화
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0, std=0.01)

    def forward(self, detect_feat, fair_feat):
        """
        Forward pass

        Args:
            detect_feat (torch.Tensor): Detection feature [B, dim]
            fair_feat (torch.Tensor): Fairness feature [B, dim]

        Returns:
            output (torch.Tensor): Cross-attention 출력 [B, dim]
            attn_weights (torch.Tensor): Attention weights [B, 1, num_views]
        """
        # fair_feat을 다중 view로 확장: [B, dim] -> [B, num_views, dim]
        views = [proj(fair_feat).unsqueeze(1) for proj in self.view_projections]
        fair_views = torch.cat(views, dim=1)  # [B, num_views, dim]

        # detect_feat: [B, dim] -> [B, 1, dim] (query)
        query = detect_feat.unsqueeze(1)  # [B, 1, dim]

        # Cross-Attention: detect가 fair의 num_views개 view를 참조
        # query: [B, 1, dim], key/value: [B, num_views, dim]
        # -> attn_out: [B, 1, dim], attn_weights: [B, 1, num_views]
        attn_out, attn_weights = self.cross_attn(query, fair_views, fair_views)

        # Residual + LayerNorm
        x = self.norm1(query + attn_out)  # [B, 1, dim]

        # FFN + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))   # [B, 1, dim]

        # [B, 1, dim] -> [B, dim]
        output = x.squeeze(1)

        return output, attn_weights


class DemographicFusionModule(nn.Module):
    """
    완전한 Demographic-Aware Fusion Module

    Step 1: AdaIN Normalization
    Step 2: Multi-Head Cross-Attention (demographic views)
    Step 3: Gated Fusion (AdaIN + CA 혼합)
    """

    def __init__(self, dim=768, num_views=4, num_heads=8,
                 dropout=0.1, gate_init_bias=0.0):
        """
        Args:
            dim (int): Feature 차원
            num_views (int): Fair feature view 수
            num_heads (int): Attention head 수
            dropout (float): Dropout 비율
            gate_init_bias (float): Gate 초기 bias (0.0 -> 균등 출발)
        """
        super().__init__()

        self.dim = dim

        # Step 1: AdaIN
        from model.adain_fusion import AdaptiveInstanceNorm
        self.adain = AdaptiveInstanceNorm(dim=dim)

        # Step 2: Demographic-Aware Cross-Attention
        self.cross_attention = DemographicAwareCrossAttention(
            dim=dim,
            num_views=num_views,
            num_heads=num_heads,
            dropout=dropout
        )

        # Step 3: Gated Fusion
        # gate = sigmoid(W * [adain_out; ca_out])
        # fused = gate * adain_out + (1 - gate) * ca_out
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Gate 초기 bias: 0.0 -> sigmoid(0.0) = 0.5 (균등 출발)
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

    def forward(self, detect_feat, fair_feat):
        """
        Forward pass

        Args:
            detect_feat (torch.Tensor): Stage 2 detection feature [B, dim]
            fair_feat (torch.Tensor): Stage 1 fairness feature [B, dim] (no grad)

        Returns:
            fused_feat (torch.Tensor): 융합된 feature [B, dim]
            gate (torch.Tensor): Gate 값 [B, 1]
            attn_weights (torch.Tensor): Attention weights [B, 1, num_views]
        """
        # Step 1: AdaIN - fair statistics로 detect_feat 정규화
        adain_out = self.adain(detect_feat, fair_feat)

        # Step 2: Cross-Attention - detect가 fair의 다양한 view 참조
        ca_out, attn_weights = self.cross_attention(detect_feat, fair_feat)

        # Step 3: Gated Fusion
        gate_input = torch.cat([adain_out, ca_out], dim=-1)  # [B, dim*2]
        gate = self.gate_net(gate_input)  # [B, 1]

        # Fused Feature = gate * AdaIN + (1-gate) * CrossAttn
        fused_feat = gate * adain_out + (1 - gate) * ca_out

        return fused_feat, gate, attn_weights
