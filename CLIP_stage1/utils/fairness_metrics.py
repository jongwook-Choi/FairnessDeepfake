"""
Fairness Metrics 유틸리티
Subgroup별 feature 분포 분석 및 공정성 메트릭 계산
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


SUBGROUP_NAMES = [
    "(Male, Asian)", "(Male, Black)", "(Male, White)", "(Male, Other)",
    "(Female, Asian)", "(Female, Black)", "(Female, White)", "(Female, Other)"
]

RACE_NAMES = ['Asian', 'Black', 'White', 'Other']
GENDER_NAMES = ['Male', 'Female']


def compute_subgroup_feature_stats(features, subgroups, num_subgroups=8):
    """
    Subgroup별 feature 통계 계산

    Args:
        features (torch.Tensor): Feature 텐서 [N, feature_dim]
        subgroups (torch.Tensor): Subgroup 레이블 [N]
        num_subgroups (int): Subgroup 수

    Returns:
        dict: Subgroup별 통계
            - 'mean': 평균 feature
            - 'std': 표준편차
            - 'count': 샘플 수
    """
    stats = {}

    for sg_id in range(num_subgroups):
        mask = (subgroups == sg_id)
        sg_features = features[mask]

        if sg_features.size(0) > 0:
            stats[sg_id] = {
                'name': SUBGROUP_NAMES[sg_id],
                'count': sg_features.size(0),
                'mean': sg_features.mean(dim=0),
                'std': sg_features.std(dim=0) if sg_features.size(0) > 1 else torch.zeros_like(sg_features[0]),
                'norm_mean': sg_features.norm(dim=1).mean().item(),
                'norm_std': sg_features.norm(dim=1).std().item() if sg_features.size(0) > 1 else 0.0
            }
        else:
            stats[sg_id] = {
                'name': SUBGROUP_NAMES[sg_id],
                'count': 0,
                'mean': None,
                'std': None,
                'norm_mean': 0.0,
                'norm_std': 0.0
            }

    return stats


def compute_subgroup_cosine_similarity(features, subgroups, num_subgroups=8):
    """
    Subgroup 간 cosine similarity 계산

    Args:
        features (torch.Tensor): Feature 텐서 [N, feature_dim]
        subgroups (torch.Tensor): Subgroup 레이블 [N]
        num_subgroups (int): Subgroup 수

    Returns:
        torch.Tensor: Cosine similarity matrix [num_subgroups, num_subgroups]
        dict: 통계 정보
    """
    # Subgroup별 mean feature 계산
    mean_features = []
    valid_subgroups = []

    for sg_id in range(num_subgroups):
        mask = (subgroups == sg_id)
        sg_features = features[mask]

        if sg_features.size(0) > 0:
            mean_feat = sg_features.mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=0)  # L2 normalize
            mean_features.append(mean_feat)
            valid_subgroups.append(sg_id)
        else:
            mean_features.append(None)

    # Cosine similarity matrix 계산
    n_valid = len(valid_subgroups)
    sim_matrix = torch.zeros(num_subgroups, num_subgroups, device=features.device)

    for i, sg_i in enumerate(valid_subgroups):
        for j, sg_j in enumerate(valid_subgroups):
            if mean_features[sg_i] is not None and mean_features[sg_j] is not None:
                sim = F.cosine_similarity(
                    mean_features[sg_i].unsqueeze(0),
                    mean_features[sg_j].unsqueeze(0)
                )
                sim_matrix[sg_i, sg_j] = sim.item()

    # 통계 계산 (대각선 제외한 off-diagonal 원소들)
    off_diagonal_mask = ~torch.eye(num_subgroups, dtype=bool, device=sim_matrix.device)
    valid_mask = (sim_matrix != 0) & off_diagonal_mask
    off_diagonal_values = sim_matrix[valid_mask]

    stats = {
        'mean_similarity': off_diagonal_values.mean().item() if len(off_diagonal_values) > 0 else 0.0,
        'min_similarity': off_diagonal_values.min().item() if len(off_diagonal_values) > 0 else 0.0,
        'max_similarity': off_diagonal_values.max().item() if len(off_diagonal_values) > 0 else 0.0,
        'std_similarity': off_diagonal_values.std().item() if len(off_diagonal_values) > 1 else 0.0,
    }

    return sim_matrix, stats


def compute_feature_variance_ratio(features, subgroups, num_subgroups=8):
    """
    Feature 분산 비율 계산

    Within-group variance와 Between-group variance의 비율로
    subgroup 간 feature 분포의 유사성을 측정

    비율이 낮을수록 subgroup 간 분포가 유사함

    Args:
        features (torch.Tensor): Feature 텐서 [N, feature_dim]
        subgroups (torch.Tensor): Subgroup 레이블 [N]
        num_subgroups (int): Subgroup 수

    Returns:
        dict: 분산 비율 통계
    """
    # 전체 평균
    global_mean = features.mean(dim=0)

    # Within-group variance 계산
    within_var = torch.tensor(0.0, device=features.device)
    total_samples = 0

    # Subgroup별 평균
    subgroup_means = []
    subgroup_counts = []

    for sg_id in range(num_subgroups):
        mask = (subgroups == sg_id)
        sg_features = features[mask]

        if sg_features.size(0) > 0:
            sg_mean = sg_features.mean(dim=0)
            subgroup_means.append(sg_mean)
            subgroup_counts.append(sg_features.size(0))

            # Within-group variance (각 sample과 group mean의 차이)
            within_var += ((sg_features - sg_mean) ** 2).sum()
            total_samples += sg_features.size(0)
        else:
            subgroup_means.append(None)
            subgroup_counts.append(0)

    if total_samples > 0:
        within_var = within_var / total_samples

    # Between-group variance 계산 (각 group mean과 global mean의 차이)
    between_var = torch.tensor(0.0, device=features.device)
    valid_groups = 0

    for i, (sg_mean, count) in enumerate(zip(subgroup_means, subgroup_counts)):
        if sg_mean is not None and count > 0:
            between_var += count * ((sg_mean - global_mean) ** 2).sum()
            valid_groups += 1

    if total_samples > 0:
        between_var = between_var / total_samples

    # Variance ratio
    variance_ratio = (between_var / (within_var + 1e-8)).item()

    stats = {
        'within_variance': within_var.item(),
        'between_variance': between_var.item(),
        'variance_ratio': variance_ratio,
        'valid_subgroups': valid_groups,
        'total_samples': total_samples
    }

    return stats


def compute_subgroup_sinkhorn_distance(features, subgroups, num_subgroups=8,
                                        sinkhorn_blur=1e-4, min_samples=2):
    """
    Sinkhorn distance 기반 subgroup 분포 유사도 계산

    각 subgroup과 전체 분포 간의 Sinkhorn distance 평균을 계산
    (fairness_loss와 동일한 기준으로 평가)

    Args:
        features (torch.Tensor): Feature 텐서 [N, feature_dim]
        subgroups (torch.Tensor): Subgroup 레이블 [N]
        num_subgroups (int): Subgroup 수
        sinkhorn_blur (float): Sinkhorn blur 파라미터
        min_samples (int): 최소 샘플 수

    Returns:
        dict: Sinkhorn distance 통계
            - 'mean_distance': 평균 Sinkhorn distance (낮을수록 좋음)
            - 'subgroup_distances': 각 subgroup별 distance
            - 'within_variance': 전체 feature의 within variance (collapse 감지용)
    """
    try:
        from geomloss import SamplesLoss
        sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur, backend="tensorized")
        geomloss_available = True
    except ImportError:
        geomloss_available = False

    if not geomloss_available:
        return {
            'mean_distance': -1.0,  # geomloss 미설치 표시
            'subgroup_distances': {},
            'within_variance': 0.0,
            'valid_subgroups': 0,
            'geomloss_available': False
        }

    device = features.device
    all_features = features.detach()

    subgroup_distances = {}
    total_distance = 0.0
    valid_count = 0

    for sg_id in range(num_subgroups):
        mask = (subgroups == sg_id)
        sg_features = features[mask]

        if sg_features.size(0) >= min_samples:
            # 동일 크기로 샘플링
            n_samples = min(sg_features.size(0), all_features.size(0))

            # Reference: 전체 분포에서 랜덤 샘플링
            indices = torch.randperm(all_features.size(0), device=device)[:n_samples]
            ref_features = all_features[indices]

            # Subgroup features 샘플링
            sg_sample = sg_features[:n_samples]

            with torch.no_grad():
                dist = sinkhorn(sg_sample, ref_features).item()

            subgroup_distances[sg_id] = {
                'name': SUBGROUP_NAMES[sg_id],
                'distance': dist,
                'count': sg_features.size(0)
            }
            total_distance += dist
            valid_count += 1

    # 평균 distance
    mean_distance = total_distance / valid_count if valid_count > 0 else 0.0

    # Within variance 계산 (collapse 감지용)
    within_var = features.var(dim=0).mean().item()

    stats = {
        'mean_distance': mean_distance,
        'subgroup_distances': subgroup_distances,
        'within_variance': within_var,
        'valid_subgroups': valid_count,
        'geomloss_available': True
    }

    return stats


def compute_classification_fairness_metrics(predictions, labels, subgroups, num_subgroups=8):
    """
    Classification 관점의 fairness metrics 계산

    Args:
        predictions (torch.Tensor): 예측 logits [N, num_classes]
        labels (torch.Tensor): Ground truth labels [N]
        subgroups (torch.Tensor): Subgroup labels [N]
        num_subgroups (int): Subgroup 수

    Returns:
        dict: Fairness metrics
    """
    preds = torch.argmax(predictions, dim=1)

    # Subgroup별 accuracy
    subgroup_acc = {}
    for sg_id in range(num_subgroups):
        mask = (subgroups == sg_id)
        if mask.sum() > 0:
            sg_preds = preds[mask]
            sg_labels = labels[mask]
            acc = (sg_preds == sg_labels).float().mean().item()
            subgroup_acc[sg_id] = {
                'name': SUBGROUP_NAMES[sg_id],
                'accuracy': acc,
                'count': mask.sum().item()
            }

    # Fairness metrics
    accuracies = [v['accuracy'] for v in subgroup_acc.values()]

    if len(accuracies) >= 2:
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        # Equal Opportunity difference (max - min)
        eod = max_acc - min_acc

        # Coefficient of Variation
        cv = std_acc / (mean_acc + 1e-8)

        fairness_metrics = {
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'accuracy_gap': eod,
            'coefficient_of_variation': cv,
            'subgroup_accuracies': subgroup_acc
        }
    else:
        fairness_metrics = {
            'max_accuracy': 0.0,
            'min_accuracy': 0.0,
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'accuracy_gap': 0.0,
            'coefficient_of_variation': 0.0,
            'subgroup_accuracies': subgroup_acc
        }

    return fairness_metrics


def print_fairness_summary(stats_dict, sim_stats, var_stats, race_metrics=None, gender_metrics=None):
    """
    Fairness 분석 결과 출력

    Args:
        stats_dict: Subgroup별 feature 통계
        sim_stats: Cosine similarity 통계
        var_stats: Variance ratio 통계
        race_metrics: Race classification fairness metrics
        gender_metrics: Gender classification fairness metrics
    """
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS SUMMARY")
    print("=" * 60)

    # Feature Statistics
    print("\n[Subgroup Feature Statistics]")
    for sg_id, stats in stats_dict.items():
        if stats['count'] > 0:
            print(f"  {stats['name']}: "
                  f"count={stats['count']}, "
                  f"norm_mean={stats['norm_mean']:.4f}, "
                  f"norm_std={stats['norm_std']:.4f}")

    # Cosine Similarity
    print(f"\n[Subgroup Cosine Similarity]")
    print(f"  Mean: {sim_stats['mean_similarity']:.4f}")
    print(f"  Min: {sim_stats['min_similarity']:.4f}")
    print(f"  Max: {sim_stats['max_similarity']:.4f}")
    print(f"  Std: {sim_stats['std_similarity']:.4f}")

    # Variance Ratio
    print(f"\n[Feature Variance Analysis]")
    print(f"  Within-group variance: {var_stats['within_variance']:.6f}")
    print(f"  Between-group variance: {var_stats['between_variance']:.6f}")
    print(f"  Variance ratio (lower is better): {var_stats['variance_ratio']:.6f}")

    # Classification Fairness
    if race_metrics:
        print(f"\n[Race Classification Fairness]")
        print(f"  Accuracy gap: {race_metrics['accuracy_gap']:.4f}")
        print(f"  Mean accuracy: {race_metrics['mean_accuracy']:.4f}")
        print(f"  CV: {race_metrics['coefficient_of_variation']:.4f}")

    if gender_metrics:
        print(f"\n[Gender Classification Fairness]")
        print(f"  Accuracy gap: {gender_metrics['accuracy_gap']:.4f}")
        print(f"  Mean accuracy: {gender_metrics['mean_accuracy']:.4f}")
        print(f"  CV: {gender_metrics['coefficient_of_variation']:.4f}")

    print("=" * 60)
