"""
CLIP Stage 2 Training Script v2
AdaIN + Demographic-Aware Cross-Attention + CVaR/LDAM/Sinkhorn Fairness

Usage:
    python train_stage2_v2.py --config config/train_stage2_v2.yaml
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.stage2_trainer_v2 import Stage2TrainerV2


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


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """메인 학습 함수"""
    # 설정 로드
    config = load_config(args.config)

    # Command line 오버라이드
    if args.batch_size:
        config.setdefault('training', {})['train_batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('optimizer', {})['lr'] = args.lr
    if args.epochs:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    if args.device:
        config['device'] = args.device
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    if args.experiment_name:
        config.setdefault('logging', {})['experiment_name'] = args.experiment_name

    # Fairness strategy preset 오버라이드
    if args.preset:
        fairness_strategy = config.get('loss', {}).get('fairness_strategy', {})
        if args.preset == 'full':
            fairness_strategy['use_cvar'] = True
            fairness_strategy['use_ldam'] = True
            fairness_strategy['use_sinkhorn'] = True
        elif args.preset == 'cvar':
            fairness_strategy['use_cvar'] = True
            fairness_strategy['use_ldam'] = False
            fairness_strategy['use_sinkhorn'] = True
        elif args.preset == 'ldam':
            fairness_strategy['use_cvar'] = False
            fairness_strategy['use_ldam'] = True
            fairness_strategy['use_sinkhorn'] = True
        elif args.preset == 'minimal':
            fairness_strategy['use_cvar'] = False
            fairness_strategy['use_ldam'] = False
            fairness_strategy['use_sinkhorn'] = True
        elif args.preset == 'none':
            fairness_strategy['use_cvar'] = False
            fairness_strategy['use_ldam'] = False
            fairness_strategy['use_sinkhorn'] = False
        else:
            raise ValueError(f"Unknown preset: {args.preset}")

        config.setdefault('loss', {})['fairness_strategy'] = fairness_strategy
        config.setdefault('logging', {})['experiment_name'] = f"stage2_v2_{args.preset}"
        print(f"Using fairness preset: {args.preset}")

    # Seed 설정
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to {seed}")

    # Device 확인
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['device'] = 'cpu'

    print("\n" + "=" * 60)
    print("CLIP Stage 2 Training v2")
    print("AdaIN + Demographic-Aware Cross-Attention")
    print("=" * 60)
    print(f"Config: {args.config}")

    # Trainer 생성 및 학습
    trainer = Stage2TrainerV2(config)
    best_metrics = trainer.train()

    print("\n[Training Complete]")
    print(f"Best metrics: {best_metrics}")

    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Stage 2 Training v2')
    parser.add_argument('--config', type=str, default='config/train_stage2_v2.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Override Stage 1 checkpoint path')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Override experiment name')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['full', 'cvar', 'ldam', 'minimal', 'none'],
                        help='Fairness strategy preset '
                             '(full: CVaR+LDAM+Sinkhorn, '
                             'cvar: CVaR+Sinkhorn, '
                             'ldam: LDAM+Sinkhorn, '
                             'minimal: Sinkhorn only, '
                             'none: no fairness)')

    args = parser.parse_args()
    main(args)
