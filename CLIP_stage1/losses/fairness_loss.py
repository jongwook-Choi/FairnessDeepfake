"""
Fairness Loss using Sinkhorn Distance
각 subgroup의 feature 분포가 동일해지도록 하는 loss

geomloss 라이브러리의 SamplesLoss 사용
"""

import torch
import torch.nn as nn
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("Warning: geomloss not installed. Please run: pip install geomloss")


class FairnessLoss(nn.Module):
    """
    Fairness Loss using Sinkhorn Distance

    각 subgroup의 feature 분포와 전체 분포 간의 Sinkhorn distance를 최소화하여
    subgroup별 feature 분포가 동일해지도록 학습

    Loss = (1/N) * sum(Sinkhorn(subgroup_features, all_features)) for all subgroups
    """

    def __init__(self, sinkhorn_blur=1e-4, sinkhorn_p=2, scaling=0.9,
                 min_samples_per_subgroup=2, num_subgroups=8):
        """
        Args:
            sinkhorn_blur (float): Sinkhorn blur 파라미터 (regularization)
            sinkhorn_p (int): p-norm for distance (1 or 2)
            scaling (float): Sinkhorn scaling 파라미터
            min_samples_per_subgroup (int): subgroup별 최소 샘플 수
            num_subgroups (int): 총 subgroup 수 (기본 8)
        """
        super().__init__()

        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_p = sinkhorn_p
        self.scaling = scaling
        self.min_samples_per_subgroup = min_samples_per_subgroup
        self.num_subgroups = num_subgroups

        if GEOMLOSS_AVAILABLE:
            self.sinkhorn = SamplesLoss(
                loss="sinkhorn",
                p=sinkhorn_p,
                blur=sinkhorn_blur,
                scaling=scaling,
                backend="tensorized"  # GPU 가속
            )
        else:
            self.sinkhorn = None

    def forward(self, features, subgroups):
        """
        Fairness loss 계산

        Args:
            features (torch.Tensor): Feature 텐서 [batch_size, feature_dim]
            subgroups (torch.Tensor): Subgroup 레이블 [batch_size]

        Returns:
            torch.Tensor: Fairness loss (scalar)
            dict: 각 subgroup별 loss 값
        """
        if self.sinkhorn is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}

        total_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        valid_subgroups = 0
        subgroup_losses = {}

        # 전체 feature를 reference로 사용
        all_features = features.detach()

        for sg_id in range(self.num_subgroups):
            mask = (subgroups == sg_id)
            sg_features = features[mask]

            if sg_features.size(0) >= self.min_samples_per_subgroup:
                # 해당 subgroup과 전체 분포 간의 Sinkhorn distance 계산
                # 전체 분포에서 같은 수만큼 샘플링
                num_samples = min(sg_features.size(0), all_features.size(0))

                if num_samples >= self.min_samples_per_subgroup:
                    # 동일한 크기로 맞추기 위해 랜덤 샘플링
                    if all_features.size(0) > num_samples:
                        indices = torch.randperm(all_features.size(0))[:num_samples]
                        ref_features = all_features[indices]
                    else:
                        ref_features = all_features

                    # 크기 맞추기
                    if sg_features.size(0) > ref_features.size(0):
                        sg_features_sample = sg_features[:ref_features.size(0)]
                    else:
                        sg_features_sample = sg_features
                        ref_features = ref_features[:sg_features.size(0)]

                    # Sinkhorn distance 계산
                    sg_loss = self.sinkhorn(sg_features_sample, ref_features)
                    total_loss = total_loss + sg_loss
                    valid_subgroups += 1
                    subgroup_losses[sg_id] = sg_loss.item()

        # 평균
        if valid_subgroups > 0:
            total_loss = total_loss / valid_subgroups

        return total_loss, subgroup_losses


class PairwiseFairnessLoss(nn.Module):
    """
    Pairwise Fairness Loss

    모든 subgroup 쌍에 대해 Sinkhorn distance를 계산하여
    subgroup 간 feature 분포가 서로 유사해지도록 학습
    """

    def __init__(self, sinkhorn_blur=1e-4, sinkhorn_p=2, scaling=0.9,
                 min_samples_per_subgroup=2, num_subgroups=8):
        """
        Args:
            sinkhorn_blur (float): Sinkhorn blur 파라미터
            sinkhorn_p (int): p-norm for distance
            scaling (float): Sinkhorn scaling 파라미터
            min_samples_per_subgroup (int): subgroup별 최소 샘플 수
            num_subgroups (int): 총 subgroup 수
        """
        super().__init__()

        self.sinkhorn_blur = sinkhorn_blur
        self.min_samples_per_subgroup = min_samples_per_subgroup
        self.num_subgroups = num_subgroups

        if GEOMLOSS_AVAILABLE:
            self.sinkhorn = SamplesLoss(
                loss="sinkhorn",
                p=sinkhorn_p,
                blur=sinkhorn_blur,
                scaling=scaling,
                backend="tensorized"
            )
        else:
            self.sinkhorn = None

    def forward(self, features, subgroups):
        """
        Pairwise fairness loss 계산

        Args:
            features (torch.Tensor): Feature 텐서 [batch_size, feature_dim]
            subgroups (torch.Tensor): Subgroup 레이블 [batch_size]

        Returns:
            torch.Tensor: Fairness loss (scalar)
            dict: subgroup 쌍별 loss 값
        """
        if self.sinkhorn is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}

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

        # 모든 쌍에 대해 Sinkhorn distance 계산
        sg_ids = list(subgroup_features.keys())
        for i, sg_i in enumerate(sg_ids):
            for sg_j in sg_ids[i+1:]:
                feat_i = subgroup_features[sg_i]
                feat_j = subgroup_features[sg_j]

                # 크기 맞추기
                min_size = min(feat_i.size(0), feat_j.size(0))
                feat_i_sample = feat_i[:min_size]
                feat_j_sample = feat_j[:min_size]

                # Sinkhorn distance 계산
                pair_loss = self.sinkhorn(feat_i_sample, feat_j_sample)
                total_loss = total_loss + pair_loss
                num_pairs += 1
                pair_losses[(sg_i, sg_j)] = pair_loss.item()

        # 평균
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss, pair_losses


class MMDFairnessLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) based Fairness Loss

    geomloss가 없을 경우 대안으로 사용 가능한 MMD 기반 fairness loss
    """

    def __init__(self, kernel='rbf', sigma=1.0, min_samples_per_subgroup=2, num_subgroups=8):
        """
        Args:
            kernel (str): 커널 타입 ('rbf' 또는 'linear')
            sigma (float): RBF 커널의 bandwidth
            min_samples_per_subgroup (int): subgroup별 최소 샘플 수
            num_subgroups (int): 총 subgroup 수
        """
        super().__init__()

        self.kernel = kernel
        self.sigma = sigma
        self.min_samples_per_subgroup = min_samples_per_subgroup
        self.num_subgroups = num_subgroups

    def rbf_kernel(self, x, y):
        """RBF 커널 계산"""
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yy = (y ** 2).sum(dim=1, keepdim=True)
        xy = x @ y.t()
        dist = xx - 2 * xy + yy.t()
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def mmd(self, x, y):
        """MMD 계산"""
        if self.kernel == 'rbf':
            kxx = self.rbf_kernel(x, x)
            kyy = self.rbf_kernel(y, y)
            kxy = self.rbf_kernel(x, y)
        else:  # linear
            kxx = x @ x.t()
            kyy = y @ y.t()
            kxy = x @ y.t()

        mmd_val = kxx.mean() + kyy.mean() - 2 * kxy.mean()
        return mmd_val

    def forward(self, features, subgroups):
        """
        MMD fairness loss 계산

        Args:
            features (torch.Tensor): Feature 텐서 [batch_size, feature_dim]
            subgroups (torch.Tensor): Subgroup 레이블 [batch_size]

        Returns:
            torch.Tensor: Fairness loss (scalar)
            dict: 각 subgroup별 loss 값
        """
        total_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        valid_subgroups = 0
        subgroup_losses = {}

        all_features = features.detach()

        for sg_id in range(self.num_subgroups):
            mask = (subgroups == sg_id)
            sg_features = features[mask]

            if sg_features.size(0) >= self.min_samples_per_subgroup:
                # MMD 계산
                sg_loss = self.mmd(sg_features, all_features)
                total_loss = total_loss + sg_loss
                valid_subgroups += 1
                subgroup_losses[sg_id] = sg_loss.item()

        if valid_subgroups > 0:
            total_loss = total_loss / valid_subgroups

        return total_loss, subgroup_losses
