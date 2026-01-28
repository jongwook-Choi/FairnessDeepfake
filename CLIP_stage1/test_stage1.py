"""
CLIP Stage1 Testing Script
학습된 Additive Adapter의 Fairness 성능 평가

Usage:
    python test_stage1.py --checkpoint path/to/checkpoint.pth --config config/train_stage1.yaml
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.fairness_adapter import FairnessAdapter
from dataset.fairness_dataset import FairnessDataset
from losses.combined_loss import CombinedLoss
from utils.fairness_metrics import (
    compute_subgroup_feature_stats,
    compute_subgroup_cosine_similarity,
    compute_feature_variance_ratio,
    compute_classification_fairness_metrics,
    print_fairness_summary,
    SUBGROUP_NAMES
)


def set_seed(seed):
    """Random seed 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config):
    """중첩된 config를 평탄화"""
    flat_config = {}

    for section in ['model', 'dataset', 'loss', 'training', 'logging']:
        if section in config:
            flat_config.update(config[section])

    for key in ['device', 'seed']:
        if key in config:
            flat_config[key] = config[key]

    return flat_config


def create_test_dataloader(config, dataset_name=None):
    """테스트 데이터로더 생성"""
    if dataset_name:
        config['dataset_name'] = dataset_name

    test_dataset = FairnessDataset(config, mode='test')

    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FairnessDataset.collate_fn,
        pin_memory=True
    )

    return test_loader


def load_model(checkpoint_path, config, device):
    """체크포인트에서 모델 로드"""
    model = FairnessAdapter(
        clip_name=config.get('clip_name', 'ViT-L/14'),
        adapter_hidden_dim=config.get('adapter_hidden_dim', 512),
        feature_dim=config.get('feature_dim', 768),
        num_races=config.get('num_races', 4),
        num_genders=config.get('num_genders', 2),
        dropout=config.get('dropout', 0.1),
        normalize_features=config.get('normalize_features', True),
        device=device,
        clip_download_root=config.get('clip_download_root', '/data/cuixinjie/weights')
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metrics: {checkpoint.get('best_metrics', 'N/A')}")

    return model


@torch.no_grad()
def evaluate(model, test_loader, device, config):
    """
    모델 평가

    Args:
        model: FairnessAdapter 모델
        test_loader: 테스트 데이터 로더
        device: 평가 device
        config: 설정 딕셔너리

    Returns:
        dict: 평가 결과
    """
    model.eval()

    # Loss 함수
    criterion = CombinedLoss(
        lambda_race=config.get('lambda_race', 1.0),
        lambda_gender=config.get('lambda_gender', 0.1),
        lambda_fairness=config.get('lambda_fairness', 1e-4),
        sinkhorn_blur=config.get('sinkhorn_blur', 1e-4)
    )

    # 결과 수집
    all_features = []
    all_clip_features = []
    all_additive_features = []
    all_race_logits = []
    all_gender_logits = []
    all_races = []
    all_genders = []
    all_subgroups = []
    all_labels = []

    total_loss = 0.0
    num_batches = 0

    print("\n[Evaluating...]")
    for data_dict in tqdm(test_loader):
        # 데이터를 device로 이동
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        # Forward pass
        pred_dict = model(data_dict)
        loss_dict = criterion(pred_dict, data_dict)

        # 결과 수집
        all_features.append(pred_dict['final_features'].cpu())
        all_clip_features.append(pred_dict['clip_features'].cpu())
        all_additive_features.append(pred_dict['additive_features'].cpu())
        all_race_logits.append(pred_dict['race_logits'].cpu())
        all_gender_logits.append(pred_dict['gender_logits'].cpu())
        all_races.append(data_dict['race'].cpu())
        all_genders.append(data_dict['gender'].cpu())
        all_subgroups.append(data_dict['subgroup'].cpu())
        all_labels.append(data_dict['label'].cpu())

        total_loss += loss_dict['overall'].item()
        num_batches += 1

    # 텐서 연결
    all_features = torch.cat(all_features, dim=0)
    all_clip_features = torch.cat(all_clip_features, dim=0)
    all_additive_features = torch.cat(all_additive_features, dim=0)
    all_race_logits = torch.cat(all_race_logits, dim=0)
    all_gender_logits = torch.cat(all_gender_logits, dim=0)
    all_races = torch.cat(all_races, dim=0)
    all_genders = torch.cat(all_genders, dim=0)
    all_subgroups = torch.cat(all_subgroups, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # =========================================================================
    # 1. Classification Metrics
    # =========================================================================
    race_preds = torch.argmax(all_race_logits, dim=1)
    gender_preds = torch.argmax(all_gender_logits, dim=1)
    race_acc = (race_preds == all_races).float().mean().item()
    gender_acc = (gender_preds == all_genders).float().mean().item()

    # =========================================================================
    # 2. Fairness Metrics for Final Features (CLIP + Additive)
    # =========================================================================
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS: Final Features (CLIP + Additive)")
    print("=" * 60)

    final_stats = compute_subgroup_feature_stats(all_features, all_subgroups)
    final_sim_matrix, final_sim_stats = compute_subgroup_cosine_similarity(all_features, all_subgroups)
    final_var_stats = compute_feature_variance_ratio(all_features, all_subgroups)
    final_race_fairness = compute_classification_fairness_metrics(all_race_logits, all_races, all_subgroups)
    final_gender_fairness = compute_classification_fairness_metrics(all_gender_logits, all_genders, all_subgroups)

    print_fairness_summary(final_stats, final_sim_stats, final_var_stats,
                          final_race_fairness, final_gender_fairness)

    # =========================================================================
    # 3. Fairness Metrics for CLIP Features (Baseline comparison)
    # =========================================================================
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS: Original CLIP Features (Baseline)")
    print("=" * 60)

    clip_stats = compute_subgroup_feature_stats(all_clip_features, all_subgroups)
    clip_sim_matrix, clip_sim_stats = compute_subgroup_cosine_similarity(all_clip_features, all_subgroups)
    clip_var_stats = compute_feature_variance_ratio(all_clip_features, all_subgroups)

    print(f"\n[Subgroup Cosine Similarity]")
    print(f"  Mean: {clip_sim_stats['mean_similarity']:.4f}")
    print(f"  Min: {clip_sim_stats['min_similarity']:.4f}")
    print(f"  Max: {clip_sim_stats['max_similarity']:.4f}")

    print(f"\n[Feature Variance Analysis]")
    print(f"  Within-group variance: {clip_var_stats['within_variance']:.6f}")
    print(f"  Between-group variance: {clip_var_stats['between_variance']:.6f}")
    print(f"  Variance ratio: {clip_var_stats['variance_ratio']:.6f}")

    # =========================================================================
    # 4. Improvement Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY (Final vs CLIP Baseline)")
    print("=" * 60)

    print(f"\n[Cosine Similarity]")
    print(f"  Mean: {clip_sim_stats['mean_similarity']:.4f} -> {final_sim_stats['mean_similarity']:.4f} "
          f"({'↑' if final_sim_stats['mean_similarity'] > clip_sim_stats['mean_similarity'] else '↓'})")

    print(f"\n[Variance Ratio (lower is better)]")
    print(f"  Ratio: {clip_var_stats['variance_ratio']:.6f} -> {final_var_stats['variance_ratio']:.6f} "
          f"({'↓ (improved)' if final_var_stats['variance_ratio'] < clip_var_stats['variance_ratio'] else '↑'})")

    # =========================================================================
    # 5. Per-Subgroup Accuracy
    # =========================================================================
    print("\n" + "=" * 60)
    print("PER-SUBGROUP ACCURACY")
    print("=" * 60)

    print("\n[Race Classification by Subgroup]")
    for sg_id in range(8):
        mask = (all_subgroups == sg_id)
        if mask.sum() > 0:
            sg_race_acc = (race_preds[mask] == all_races[mask]).float().mean().item()
            print(f"  {SUBGROUP_NAMES[sg_id]}: {sg_race_acc:.4f} (n={mask.sum().item()})")

    print("\n[Gender Classification by Subgroup]")
    for sg_id in range(8):
        mask = (all_subgroups == sg_id)
        if mask.sum() > 0:
            sg_gender_acc = (gender_preds[mask] == all_genders[mask]).float().mean().item()
            print(f"  {SUBGROUP_NAMES[sg_id]}: {sg_gender_acc:.4f} (n={mask.sum().item()})")

    # 결과 딕셔너리
    results = {
        'loss': total_loss / num_batches,
        'race_acc': race_acc,
        'gender_acc': gender_acc,
        'final_mean_cosine_sim': final_sim_stats['mean_similarity'],
        'final_variance_ratio': final_var_stats['variance_ratio'],
        'clip_mean_cosine_sim': clip_sim_stats['mean_similarity'],
        'clip_variance_ratio': clip_var_stats['variance_ratio'],
        'race_acc_gap': final_race_fairness['accuracy_gap'],
        'gender_acc_gap': final_gender_fairness['accuracy_gap'],
    }

    return results


def main(args):
    """메인 테스트 함수"""
    # 설정 로드
    config = load_config(args.config)
    flat_config = flatten_config(config)

    # Random seed 설정
    seed = flat_config.get('seed', 42)
    set_seed(seed)

    # Device 설정
    device = args.device if args.device else flat_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("\n" + "=" * 60)
    print("CLIP Stage1 Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")

    # 모델 로드
    print("\n[Loading Model]")
    model = load_model(args.checkpoint, flat_config, device)

    # 테스트 데이터로더 생성
    print("\n[Creating Test DataLoader]")
    test_loader = create_test_dataloader(flat_config, args.dataset)

    # 평가 수행
    results = evaluate(model, test_loader, device, flat_config)

    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Stage1 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/train_stage1.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to evaluate on')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()
    main(args)
