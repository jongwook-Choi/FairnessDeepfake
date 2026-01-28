"""
Stage 2 Independent Adapter Training Script
Independent Dual Adapter with Cross-Attention Fusion

사용법:
    python train_stage2_independent_adapter.py --config config/train_stage2_independent_adapter.yaml

    # 옵션 오버라이드:
    python train_stage2_independent_adapter.py --config config/train_stage2_independent_adapter.yaml \
        --stage1-checkpoint /path/to/stage1.pth \
        --lr 0.0001 \
        --epochs 30 \
        --batch-size 64
"""

import argparse
import os
import yaml
import torch
import random
import numpy as np

from trainer.stage2_independent_adapter_trainer import Stage2IndependentAdapterTrainer


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Stage 2 Independent Adapter Training',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 설정 파일
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/train_stage2_independent_adapter.yaml',
        help='Path to the config file'
    )

    # Stage 1 체크포인트
    parser.add_argument(
        '--stage1-checkpoint',
        type=str,
        help='Path to Stage 1 checkpoint (overrides config)'
    )

    # 학습 파라미터 오버라이드
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size (overrides config)'
    )

    # 데이터셋
    parser.add_argument(
        '--train-dataset',
        type=str,
        nargs='+',
        help='Training dataset names (overrides config)'
    )

    # 로깅
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Log directory (overrides config)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Experiment name (overrides config)'
    )

    # 기타
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )

    # Loss weighting
    parser.add_argument(
        '--loss-weighting',
        type=str,
        choices=['dynamic', 'fixed'],
        help='Loss weighting type (overrides config)'
    )

    parser.add_argument(
        '--lambda-fair',
        type=float,
        help='Fixed fairness loss weight (only for fixed weighting)'
    )

    return parser.parse_args()


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config, args):
    """명령행 인자로 설정 업데이트"""

    # Stage 1 체크포인트
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint

    # 학습 파라미터
    if args.lr:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['lr'] = args.lr

    if args.epochs:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_epochs'] = args.epochs

    if args.batch_size:
        if 'training' not in config:
            config['training'] = {}
        config['training']['train_batch_size'] = args.batch_size

    # 데이터셋
    if args.train_dataset:
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['train_dataset'] = args.train_dataset

    # 로깅
    if args.log_dir:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_dir'] = args.log_dir

    if args.experiment_name:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['experiment_name'] = args.experiment_name

    # 기타
    if args.device:
        config['device'] = args.device

    if args.seed:
        config['seed'] = args.seed

    # Loss weighting
    if args.loss_weighting:
        if 'loss_weighting' not in config:
            config['loss_weighting'] = {}
        config['loss_weighting']['type'] = args.loss_weighting

    if args.lambda_fair:
        if 'loss_weighting' not in config:
            config['loss_weighting'] = {}
        config['loss_weighting']['lambda_fair'] = args.lambda_fair

    return config


def print_config(config, indent=0):
    """설정 정보 출력"""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def set_seed(seed):
    """Random seed 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """메인 함수"""
    args = parse_args()

    # 설정 파일 로드
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    # 명령행 인자로 설정 오버라이드
    config = update_config_from_args(config, args)

    # Seed 설정
    seed = config.get('seed', 42)
    set_seed(seed)

    # 설정 출력
    print("\n" + "=" * 60)
    print("Stage 2 Independent Adapter Training Configuration")
    print("=" * 60)
    print_config(config)
    print("=" * 60 + "\n")

    # Stage 1 체크포인트 확인
    stage1_checkpoint = config.get('stage1_checkpoint')
    if stage1_checkpoint:
        if os.path.exists(stage1_checkpoint):
            print(f"Stage 1 checkpoint found: {stage1_checkpoint}")
        else:
            print(f"Warning: Stage 1 checkpoint not found: {stage1_checkpoint}")
            print("Stage 1 Adapter will be randomly initialized")
    else:
        print("Warning: No Stage 1 checkpoint specified")
        print("Stage 1 Adapter will be randomly initialized")

    # Trainer 생성 및 학습
    print("\nInitializing trainer...")
    trainer = Stage2IndependentAdapterTrainer(config)

    print("Starting training...")
    best_metric = trainer.train()

    print(f"\nTraining completed! Best metric: {best_metric:.4f}")

    return best_metric


if __name__ == "__main__":
    main()
