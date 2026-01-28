#!/usr/bin/env python3
"""
Stage 2 Layerwise Learning Rate Decay Full Fine-tuning 학습 스크립트

CLIP (unfrozen, layerwise LR) + Additive Adapter (unfrozen) + Binary Classifier
Deepfake Detection을 위한 Full Fine-tuning with Layerwise LR Decay 수행

Usage:
    python train_stage2_layerwise.py --config config/train_stage2_layerwise.yaml
    python train_stage2_layerwise.py --config config/train_stage2_layerwise.yaml \
        --stage1-checkpoint /path/to/stage1.pth \
        --head-lr 1e-4 \
        --layer-decay 0.65
"""

import os
import sys
import argparse
import yaml

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.stage2_layerwise_trainer import Stage2LayerwiseTrainer


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Stage 2 Full Fine-tuning with Layerwise LR Decay Training'
    )

    parser.add_argument('--config', type=str,
                        default='config/train_stage2_layerwise.yaml',
                        help='Path to config file')

    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                        help='Path to Stage 1 checkpoint (overrides config)')

    # Layerwise LR Decay 설정
    parser.add_argument('--head-lr', type=float, default=None,
                        help='Learning rate for classification head (overrides config)')

    parser.add_argument('--layer-decay', type=float, default=None,
                        help='Layer-wise decay rate (overrides config)')

    # Training 설정
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')

    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')

    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Warmup steps for LayerWiseWarmupScheduler (overrides config)')

    # Freeze 설정
    parser.add_argument('--freeze-clip', action='store_true',
                        help='Freeze CLIP visual encoder (default: False)')

    parser.add_argument('--freeze-adapter', action='store_true',
                        help='Freeze Additive Adapter (default: False)')

    # Dataset 설정
    parser.add_argument('--train-dataset', type=str, nargs='+', default=None,
                        help='Training dataset(s) (overrides config)')

    # Logging 설정
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Log directory (overrides config)')

    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (overrides config)')

    # Hardware 설정
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    return parser.parse_args()


def load_config(config_path):
    """설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def update_config_from_args(config, args):
    """명령행 인자로 설정 업데이트"""

    # Stage 1 Checkpoint
    if args.stage1_checkpoint is not None:
        config['stage1_checkpoint'] = args.stage1_checkpoint

    # Layerwise LR Decay 설정
    if args.head_lr is not None:
        config['head_lr'] = args.head_lr

    if args.layer_decay is not None:
        config['layer_decay'] = args.layer_decay

    # Freeze 설정
    if args.freeze_clip:
        config['freeze_clip'] = True

    if args.freeze_adapter:
        config['freeze_adapter'] = True

    # Epochs
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_epochs'] = args.epochs

    # Batch Size
    if args.batch_size is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['train_batch_size'] = args.batch_size
        config['training']['val_batch_size'] = args.batch_size

    # Warmup Steps
    if args.warmup_steps is not None:
        if 'scheduler' not in config:
            config['scheduler'] = {}
        config['scheduler']['warmup_steps'] = args.warmup_steps

    # Train Dataset
    if args.train_dataset is not None:
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['train_dataset'] = args.train_dataset

    # Log Directory
    if args.log_dir is not None:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_dir'] = args.log_dir

    # Experiment Name
    if args.experiment_name is not None:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['experiment_name'] = args.experiment_name

    # Device
    if args.device is not None:
        config['device'] = args.device

    # Seed
    if args.seed is not None:
        config['seed'] = args.seed

    return config


def print_config(config):
    """설정 출력"""
    print("\n" + "=" * 70)
    print("Stage 2 Full Fine-tuning with Layerwise LR Decay Configuration")
    print("=" * 70)

    # Model
    model_config = config.get('model', {})
    print(f"\n[Model]")
    print(f"  CLIP Model: {model_config.get('clip_name', 'ViT-L/14')}")
    print(f"  Feature Dim: {model_config.get('feature_dim', 768)}")
    print(f"  Classifier: {model_config.get('classifier_hidden_dims', [384, 192])}")

    # Stage 1 & Freeze
    print(f"\n[Stage 1 & Freeze]")
    print(f"  Stage 1 Checkpoint: {config.get('stage1_checkpoint', 'None')}")
    print(f"  Freeze CLIP: {config.get('freeze_clip', False)}")
    print(f"  Freeze Adapter: {config.get('freeze_adapter', False)}")

    # Layerwise LR Decay
    print(f"\n[Layerwise LR Decay]")
    print(f"  Head LR: {config.get('head_lr', 1e-4)}")
    print(f"  Layer Decay: {config.get('layer_decay', 0.65)}")

    # 예상 LR 범위 계산
    head_lr = config.get('head_lr', 1e-4)
    layer_decay = config.get('layer_decay', 0.65)
    # CLIP ViT-L/14: 24 layers + embeddings + projection
    # Classifier: 3 layers, Adapter: 3 layers
    # Total decay_steps for embeddings: ~29
    min_lr = head_lr * (layer_decay ** 29)
    print(f"  Estimated LR Range: {min_lr:.2e} ~ {head_lr:.2e}")

    # Dataset
    dataset_config = config.get('dataset', {})
    print(f"\n[Dataset]")
    print(f"  Train: {dataset_config.get('train_dataset', ['ff++'])}")
    print(f"  Val: {dataset_config.get('validation_dataset', ['ff++'])}")
    print(f"  Resolution: {dataset_config.get('resolution', 256)}")

    # Training
    training_config = config.get('training', {})
    scheduler_config = config.get('scheduler', {})
    print(f"\n[Training]")
    print(f"  Epochs: {training_config.get('num_epochs', 10)}")
    print(f"  Batch Size: {training_config.get('train_batch_size', 16)}")
    print(f"  Scheduler: {scheduler_config.get('name', 'LayerWiseWarmupScheduler')}")
    if scheduler_config.get('name') == 'LayerWiseWarmupScheduler':
        print(f"  Warmup Steps: {scheduler_config.get('warmup_steps', 100)}")

    # Logging
    logging_config = config.get('logging', {})
    print(f"\n[Logging]")
    print(f"  Log Dir: {logging_config.get('log_dir', 'logs')}")
    print(f"  Experiment: {logging_config.get('experiment_name', 'stage2_full_finetuning_lw')}")

    print("=" * 70 + "\n")


def main():
    """메인 함수"""
    args = parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 명령행 인자로 설정 업데이트
    config = update_config_from_args(config, args)

    # 설정 출력
    print_config(config)

    # Stage 1 체크포인트 확인
    stage1_checkpoint = config.get('stage1_checkpoint')
    if stage1_checkpoint and not os.path.exists(stage1_checkpoint):
        print(f"Warning: Stage 1 checkpoint not found: {stage1_checkpoint}")
        print("Proceeding with randomly initialized Adapter weights...")

    # Trainer 생성 및 학습
    trainer = Stage2LayerwiseTrainer(config)

    try:
        best_metric = trainer.train()
        print(f"\nTraining completed! Best metric: {best_metric:.4f}")
        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
