"""
Cross-Attention Fusion with Dynamic Gate
Detection Feature가 Fairness Feature를 참조하여 공정성 정보 주입

철학:
- Stage 2 (Detection)가 Query: "Detection 관점에서 필요한 Fairness 정보 선택"
- Stage 1 (Fairness)이 Key/Value: "Global Fair 정보 제공"
- Dynamic Gate: Fairness 성능 저하 시 Stage 1 정보 더 많이 활용
"""

import torch
import torch.nn as nn


class CrossAttentionFusionWithDynamicGate(nn.Module):
    """
    Detection Feature가 Fairness Feature를 참조하여 공정성 정보 주입

    철학:
    - Stage 2 (Detection)가 Query: "Detection 관점에서 필요한 Fairness 정보 선택"
    - Stage 1 (Fairness)이 Key/Value: "Global Fair 정보 제공"
    - Dynamic Gate: Fairness 성능에 따라 Stage 1 활용도 자동 조절
    """

    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        """
        Args:
            dim (int): Feature 차원
            num_heads (int): Multi-head attention의 head 수
            dropout (float): Dropout 비율
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Cross-Attention: Detection queries Fairness
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

        # Dynamic Gate: Fairness 정보 활용도 조절
        # gate 값이 높으면 Stage1 (Fairness) 더 많이 사용
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # 초기화: Stage1과 CrossAttn 동등 출발 (sigmoid(0.0) = 0.5)
        # Exp1/Exp2에서 bias=1.0 → gate 0.96-0.99 포화 문제 해결
        nn.init.constant_(self.gate_net[-2].bias, 0.0)

    def forward(self, stage2_feat, stage1_feat):
        """
        Forward pass

        Args:
            stage2_feat: [B, 768] - Detection feature (from Stage 2 Adapter)
            stage1_feat: [B, 768] - Fairness feature (from Stage 1 Adapter, frozen)

        Returns:
            fused_feat: [B, 768] - 융합된 feature
            gate: [B, 1] - Dynamic gate 값 (모니터링용)
            attn_weights: [B, 1, 1] - Attention weights (시각화용)
        """
        # [B, 768] -> [B, 1, 768] for attention
        s2 = stage2_feat.unsqueeze(1)  # Query: Detection
        s1 = stage1_feat.unsqueeze(1)  # Key/Value: Fairness

        # Cross-Attention: Detection queries Fairness
        # Query = Stage2 (Detection), Key/Value = Stage1 (Fairness)
        attn_out, attn_weights = self.cross_attn(s2, s1, s1)

        # Residual connection + LayerNorm
        x = self.norm1(s2 + attn_out)

        # FFN + Residual connection + LayerNorm
        x = self.norm2(x + self.ffn(x))

        # [B, 1, 768] -> [B, 768]
        x = x.squeeze(1)

        # Dynamic Gate 계산
        # gate_input = concat(cross_attn_output, stage1_feat)
        gate_input = torch.cat([x, stage1_feat], dim=-1)  # [B, 768*2]
        gate = self.gate_net(gate_input)  # [B, 1]

        # Fused Feature = gate * Stage1 + (1-gate) * CrossAttn_result
        # gate가 높으면 Stage1 (Fairness) 정보 더 많이 활용
        fused = gate * stage1_feat + (1 - gate) * x

        return fused, gate, attn_weights


class SimpleFusion(nn.Module):
    """
    간단한 Fusion 모듈 (비교 실험용)

    Cross-Attention 대신 단순 가중합 또는 concatenation 사용
    """

    def __init__(self, dim=768, fusion_type='weighted_sum'):
        """
        Args:
            dim (int): Feature 차원
            fusion_type (str): 'weighted_sum', 'concat', 'gate'
        """
        super().__init__()

        self.dim = dim
        self.fusion_type = fusion_type

        if fusion_type == 'weighted_sum':
            # 학습 가능한 가중치
            self.alpha = nn.Parameter(torch.tensor(0.5))

        elif fusion_type == 'concat':
            # Concatenation 후 projection
            self.proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )

        elif fusion_type == 'gate':
            # 입력 기반 gate
            self.gate_net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

    def forward(self, stage2_feat, stage1_feat):
        """
        Forward pass

        Args:
            stage2_feat: [B, 768] - Detection feature
            stage1_feat: [B, 768] - Fairness feature

        Returns:
            fused_feat: [B, 768]
            gate: [B, 1] or None
            attn_weights: None
        """
        if self.fusion_type == 'weighted_sum':
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * stage1_feat + (1 - alpha) * stage2_feat
            return fused, torch.tensor([[alpha.item()]]).to(fused.device), None

        elif self.fusion_type == 'concat':
            concat = torch.cat([stage2_feat, stage1_feat], dim=-1)
            fused = self.proj(concat)
            return fused, None, None

        elif self.fusion_type == 'gate':
            gate_input = torch.cat([stage2_feat, stage1_feat], dim=-1)
            gate = self.gate_net(gate_input)
            fused = gate * stage1_feat + (1 - gate) * stage2_feat
            return fused, gate, None


if __name__ == "__main__":
    # 테스트 코드
    print("Cross-Attention Fusion Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    dim = 768

    # 더미 입력
    stage2_feat = torch.randn(batch_size, dim).to(device)
    stage1_feat = torch.randn(batch_size, dim).to(device)

    # CrossAttentionFusionWithDynamicGate 테스트
    print("\n1. CrossAttentionFusionWithDynamicGate")
    fusion = CrossAttentionFusionWithDynamicGate(dim=dim, num_heads=8).to(device)
    fused, gate, attn_weights = fusion(stage2_feat, stage1_feat)
    print(f"   Fused shape: {fused.shape}")
    print(f"   Gate shape: {gate.shape}, mean: {gate.mean().item():.4f}")
    print(f"   Attn weights shape: {attn_weights.shape}")

    # SimpleFusion 테스트
    print("\n2. SimpleFusion (weighted_sum)")
    simple_fusion = SimpleFusion(dim=dim, fusion_type='weighted_sum').to(device)
    fused, gate, _ = simple_fusion(stage2_feat, stage1_feat)
    print(f"   Fused shape: {fused.shape}")
    print(f"   Alpha (gate): {gate}")

    print("\n3. SimpleFusion (gate)")
    gate_fusion = SimpleFusion(dim=dim, fusion_type='gate').to(device)
    fused, gate, _ = gate_fusion(stage2_feat, stage1_feat)
    print(f"   Fused shape: {fused.shape}")
    print(f"   Gate mean: {gate.mean().item():.4f}")

    # 파라미터 수 계산
    print("\n4. Parameter count")
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"   CrossAttentionFusionWithDynamicGate: {total_params:,} params")
