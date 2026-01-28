"""
CLIP Stage1 Text Training Script
Text Encoder를 활용한 Subgroup bias 제거를 위한 Additive Adapter 학습

Usage:
    python train_stage1_txt.py --config config/train_stage1_txt.yaml
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.fairness_adapter_txt import FairnessAdapterWithText
from dataset.fairness_dataset import FairnessDataset, SubgroupBalancedBatchSampler
from trainer.stage1_txt_trainer import Stage1TxtTrainer


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


def flatten_config(config):
    """중첩된 config를 평탄화"""
    flat_config = {}

    # Model config
    if 'model' in config:
        flat_config.update(config['model'])

    # Text config (신규)
    if 'text' in config:
        flat_config.update(config['text'])

    # Dataset config
    if 'dataset' in config:
        flat_config.update(config['dataset'])

    # Loss config
    if 'loss' in config:
        flat_config.update(config['loss'])

    # Training config
    if 'training' in config:
        flat_config.update(config['training'])

    # Logging config
    if 'logging' in config:
        flat_config.update(config['logging'])

    # Top-level config
    for key in ['device', 'seed']:
        if key in config:
            flat_config[key] = config[key]

    return flat_config


def create_dataloaders(config):
    """데이터로더 생성"""
    # 학습 데이터셋
    train_dataset = FairnessDataset(config, mode='train')

    # 검증 데이터셋
    val_dataset = FairnessDataset(config, mode='validation')

    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)

    # Subgroup balanced batch sampler 사용
    train_sampler = SubgroupBalancedBatchSampler(
        train_dataset.get_subgroup_list(),
        batch_size=batch_size,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=FairnessDataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=FairnessDataset.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(config, device):
    """모델 생성 (FairnessAdapterWithText)"""
    model = FairnessAdapterWithText(
        clip_name=config.get('clip_name', 'ViT-L/14'),
        adapter_hidden_dim=config.get('adapter_hidden_dim', 512),
        feature_dim=config.get('feature_dim', 768),
        num_races=config.get('num_races', 4),
        num_genders=config.get('num_genders', 2),
        num_subgroups=config.get('num_subgroups', 8),
        num_prompts_per_subgroup=config.get('num_prompts_per_subgroup', 6),
        dropout=config.get('dropout', 0.1),
        normalize_features=config.get('normalize_features', True),
        device=device,
        clip_download_root=config.get('clip_download_root', '/data/cuixinjie/weights')
    )

    return model


def create_optimizer(model, config):
    """옵티마이저 생성"""
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0001)
    betas = tuple(optimizer_config.get('betas', [0.9, 0.999]))

    # 학습 가능한 파라미터만 선택
    trainable_params = model.get_trainable_params()

    if optimizer_type == 'adam':
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(optimizer, config):
    """스케줄러 생성"""
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine').lower()

    if scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', config.get('num_epochs', 10))
        eta_min = scheduler_config.get('eta_min', 1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 5)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def main(args):
    """메인 학습 함수"""
    # 설정 로드
    config = load_config(args.config)
    flat_config = flatten_config(config)

    # Command line 인자로 설정 오버라이드
    if args.batch_size:
        flat_config['batch_size'] = args.batch_size
    if args.lr:
        flat_config['optimizer']['lr'] = args.lr
    if args.epochs:
        flat_config['num_epochs'] = args.epochs
    if args.device:
        flat_config['device'] = args.device

    # Random seed 설정
    seed = flat_config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to {seed}")

    # Device 설정
    device = flat_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    flat_config['device'] = device

    print("\n" + "=" * 60)
    print("CLIP Stage1 Text Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config file: {args.config}")
    print(f"Text-Visual Alignment Loss: lambda={flat_config.get('lambda_text_align', 0.1)}")
    print(f"Text Consistency Loss: lambda={flat_config.get('lambda_text_consist', 0.01)}")

    # 데이터로더 생성
    print("\n[Creating DataLoaders]")
    train_loader, val_loader = create_dataloaders(flat_config)

    # 모델 생성
    print("\n[Creating Model]")
    model = create_model(flat_config, device)

    # 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # 옵티마이저 및 스케줄러 생성
    print("\n[Creating Optimizer and Scheduler]")
    optimizer = create_optimizer(model, flat_config)
    scheduler = create_scheduler(optimizer, flat_config)

    # Trainer 생성
    print("\n[Creating Trainer]")
    trainer = Stage1TxtTrainer(
        config=flat_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=flat_config.get('log_dir', './logs')
    )

    # 체크포인트에서 재개
    if args.resume:
        print(f"\n[Resuming from checkpoint: {args.resume}]")
        start_epoch = trainer.load_checkpoint(args.resume)
    else:
        start_epoch = 0

    # 학습 시작
    num_epochs = flat_config.get('num_epochs', 10)
    best_metrics = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    print("\n[Training Complete]")
    print(f"Best metrics: {best_metrics}")

    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Stage1 Text Training')
    parser.add_argument('--config', type=str, default='config/train_stage1_txt.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)
