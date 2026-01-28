"""
Stage 2 Fairness Loss
Real/Fake 각 클래스 내에서 subgroup 간 분포 정렬

목적: Deepfake Detection 학습 시 Local Fairness 보장
- Real 이미지 내에서 subgroup 간 feature 분포 동일화
- Fake 이미지 내에서 subgroup 간 feature 분포 동일화
"""

import torch
import torch.nn as nn

try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    raise ImportError("geomloss is required. Please run: pip install geomloss")


class Stage2FairnessLoss(nn.Module):
    """
    Stage 2 Fairness Loss using Sinkhorn Distance

    Real/Fake 각 클래스 내에서 subgroup 간 feature 분포 정렬
    - 각 클래스(Real/Fake) 내에서 모든 subgroup 쌍 간 Sinkhorn distance 최소화
    - Detection 성능을 유지하면서 fairness 보장

    Loss = (1/2) * (L_real + L_fake)
    where L_class = (1/M) * sum(Sinkhorn(subgroup_i, subgroup_j)) for all pairs
    """

    def __init__(self, sinkhorn_blur=1e-4, sinkhorn_p=2, scaling=0.9,
                 min_samples_per_subgroup=2, num_subgroups=8):
        """
        Args:
            sinkhorn_blur (float): Sinkhorn blur 파라미터 (regularization)
            sinkhorn_p (int): p-norm for distance (1 or 2)
            scaling (float): Sinkhorn scaling 파라미터
            min_samples_per_subgroup (int): subgroup별 최소 샘플 수
            num_subgroups (int): 총 subgroup 수 (기본 8: 2 gender × 4 race)
        """
        super().__init__()

        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_p = sinkhorn_p
        self.scaling = scaling
        self.min_samples_per_subgroup = min_samples_per_subgroup
        self.num_subgroups = num_subgroups

        # Sinkhorn Loss (geomloss)
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=sinkhorn_p,
            blur=sinkhorn_blur,
            scaling=scaling,
            backend="tensorized"  # GPU 가속
        )

    def forward(self, features, labels, subgroups):
        """
        Fairness loss 계산

        Args:
            features (torch.Tensor): Feature 텐서 [batch_size, feature_dim]
            labels (torch.Tensor): Binary labels [batch_size] (0: Real, 1: Fake)
            subgroups (torch.Tensor): Subgroup labels [batch_size] (0-7)

        Returns:
            torch.Tensor: Fairness loss (scalar)
            dict: 상세 loss 정보 (클래스별, 쌍별)
        """
        total_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        valid_classes = 0
        loss_info = {
            'real_loss': 0.0,
            'fake_loss': 0.0,
            'real_pairs': 0,
            'fake_pairs': 0,
            'pair_losses': {}
        }

        # 각 클래스(Real/Fake) 내에서 fairness loss 계산
        for label_val in [0, 1]:  # 0: Real, 1: Fake
            class_name = 'real' if label_val == 0 else 'fake'
            mask = (labels == label_val)

            if mask.sum() < self.min_samples_per_subgroup * 2:
                continue

            class_features = features[mask]
            class_subgroups = subgroups[mask]

            # Pairwise Sinkhorn distance 계산
            class_loss, num_pairs, pair_losses = self._pairwise_sinkhorn(
                class_features, class_subgroups, class_name
            )

            if num_pairs > 0:
                total_loss = total_loss + class_loss
                valid_classes += 1
                loss_info[f'{class_name}_loss'] = class_loss.item()
                loss_info[f'{class_name}_pairs'] = num_pairs
                loss_info['pair_losses'].update(pair_losses)

        # 평균
        if valid_classes > 0:
            total_loss = total_loss / valid_classes

        return total_loss, loss_info

    def _pairwise_sinkhorn(self, features, subgroups, class_name):
        """
        모든 subgroup 쌍 간 Sinkhorn distance 계산

        Args:
            features (torch.Tensor): 한 클래스 내 features [N, feature_dim]
            subgroups (torch.Tensor): 한 클래스 내 subgroups [N]
            class_name (str): 'real' or 'fake'

        Returns:
            total_loss: Sinkhorn distance 합
            num_pairs: 유효한 쌍 수
            pair_losses: 각 쌍별 loss 딕셔너리
        """
        total_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        num_pairs = 0
        pair_losses = {}

        # 각 subgroup의 feature 추출
        subgroup_features = {}
        for sg_id in range(self.num_subgroups):
            mask = (subgroups == sg_id)
            sg_features = features[mask]
            if sg_features.size(0) >= self.min_samples_per_subgroup:
                subgroup_features[sg_id] = sg_features

        # 유효한 subgroup이 2개 미만이면 loss 0
        if len(subgroup_features) < 2:
            return total_loss, 0, {}

        # 모든 쌍에 대해 Sinkhorn distance 계산
        sg_ids = list(subgroup_features.keys())
        for i, sg_i in enumerate(sg_ids):
            for sg_j in sg_ids[i + 1:]:
                feat_i = subgroup_features[sg_i]
                feat_j = subgroup_features[sg_j]

                # 크기 맞추기 (작은 쪽에 맞춤)
                min_size = min(feat_i.size(0), feat_j.size(0))
                feat_i_sample = feat_i[:min_size]
                feat_j_sample = feat_j[:min_size]

                # Sinkhorn distance 계산
                pair_loss = self.sinkhorn(feat_i_sample, feat_j_sample)
                total_loss = total_loss + pair_loss
                num_pairs += 1

                pair_key = f'{class_name}_sg{sg_i}_sg{sg_j}'
                pair_losses[pair_key] = pair_loss.item()

        # 쌍 수로 나누어 평균
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss, num_pairs, pair_losses


class Stage2GlobalFairnessLoss(nn.Module):
    """
    Stage 2 Global Fairness Loss

    클래스 구분 없이 전체 배치에서 subgroup 간 분포 정렬
    (Stage 1과 유사한 방식)
    """

    def __init__(self, sinkhorn_blur=1e-4, sinkhorn_p=2, scaling=0.9,
                 min_samples_per_subgroup=2, num_subgroups=8):
        super().__init__()

        self.min_samples_per_subgroup = min_samples_per_subgroup
        self.num_subgroups = num_subgroups

        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=sinkhorn_p,
            blur=sinkhorn_blur,
            scaling=scaling,
            backend="tensorized"
        )

    def forward(self, features, subgroups):
        """
        Global fairness loss 계산

        Args:
            features (torch.Tensor): [batch_size, feature_dim]
            subgroups (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: Loss
            dict: 상세 정보
        """
        total_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        num_pairs = 0
        pair_losses = {}

        # 각 subgroup의 feature 추출
        subgroup_features = {}
        for sg_id in range(self.num_subgroups):
            mask = (subgroups == sg_id)
            sg_features = features[mask]
            if sg_features.size(0) >= self.min_samples_per_subgroup:
                subgroup_features[sg_id] = sg_features

        if len(subgroup_features) < 2:
            return total_loss, {'num_pairs': 0}

        # 모든 쌍에 대해 Sinkhorn distance
        sg_ids = list(subgroup_features.keys())
        for i, sg_i in enumerate(sg_ids):
            for sg_j in sg_ids[i + 1:]:
                feat_i = subgroup_features[sg_i]
                feat_j = subgroup_features[sg_j]

                min_size = min(feat_i.size(0), feat_j.size(0))
                pair_loss = self.sinkhorn(feat_i[:min_size], feat_j[:min_size])

                total_loss = total_loss + pair_loss
                num_pairs += 1
                pair_losses[f'sg{sg_i}_sg{sg_j}'] = pair_loss.item()

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss, {'num_pairs': num_pairs, 'pair_losses': pair_losses}


class CombinedStage2FairnessLoss(nn.Module):
    """
    Combined Stage 2 Fairness Loss

    Local (클래스별) + Global fairness loss 조합
    """

    def __init__(self, local_weight=1.0, global_weight=0.0, **kwargs):
        """
        Args:
            local_weight (float): Local fairness loss 가중치
            global_weight (float): Global fairness loss 가중치
        """
        super().__init__()

        self.local_weight = local_weight
        self.global_weight = global_weight

        self.local_loss = Stage2FairnessLoss(**kwargs)

        if global_weight > 0:
            self.global_loss = Stage2GlobalFairnessLoss(**kwargs)
        else:
            self.global_loss = None

    def forward(self, features, labels, subgroups):
        """
        Combined fairness loss 계산

        Args:
            features (torch.Tensor): [batch_size, feature_dim]
            labels (torch.Tensor): [batch_size]
            subgroups (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: Combined loss
            dict: 상세 정보
        """
        loss_info = {}

        # Local fairness loss
        local_loss, local_info = self.local_loss(features, labels, subgroups)
        loss_info['local'] = local_info

        total_loss = self.local_weight * local_loss

        # Global fairness loss (옵션)
        if self.global_loss is not None:
            global_loss, global_info = self.global_loss(features, subgroups)
            total_loss = total_loss + self.global_weight * global_loss
            loss_info['global'] = global_info

        return total_loss, loss_info


if __name__ == "__main__":
    # 테스트 코드
    print("Stage 2 Fairness Loss Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    feature_dim = 768

    # 더미 데이터 생성
    features = torch.randn(batch_size, feature_dim).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)  # Real/Fake
    subgroups = torch.randint(0, 8, (batch_size,)).to(device)  # 8 subgroups

    # Stage2FairnessLoss 테스트
    print("\n1. Stage2FairnessLoss (Local)")
    fairness_loss = Stage2FairnessLoss(sinkhorn_blur=1e-4).to(device)
    loss, info = fairness_loss(features, labels, subgroups)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Real pairs: {info['real_pairs']}, Fake pairs: {info['fake_pairs']}")

    # Stage2GlobalFairnessLoss 테스트
    print("\n2. Stage2GlobalFairnessLoss")
    global_loss_fn = Stage2GlobalFairnessLoss(sinkhorn_blur=1e-4).to(device)
    g_loss, g_info = global_loss_fn(features, subgroups)
    print(f"   Loss: {g_loss.item():.6f}")
    print(f"   Num pairs: {g_info['num_pairs']}")

    # CombinedStage2FairnessLoss 테스트
    print("\n3. CombinedStage2FairnessLoss")
    combined_loss_fn = CombinedStage2FairnessLoss(
        local_weight=1.0, global_weight=0.5, sinkhorn_blur=1e-4
    ).to(device)
    c_loss, c_info = combined_loss_fn(features, labels, subgroups)
    print(f"   Loss: {c_loss.item():.6f}")

    # Gradient 확인
    print("\n4. Gradient check")
    features.requires_grad = True
    loss, _ = fairness_loss(features, labels, subgroups)
    loss.backward()
    print(f"   Gradient exists: {features.grad is not None}")
    print(f"   Gradient norm: {features.grad.norm().item():.6f}")
