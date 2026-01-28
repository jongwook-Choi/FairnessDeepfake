"""
Stage 2 Layerwise Learning Rate Decay Trainer
CLIP visual encoder의 각 layer에 다른 learning rate 적용하여 Full Fine-tuning 수행
하위 layer (초기 layer)일수록 낮은 learning rate 적용
"""

import os
import time
import datetime
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from model.stage2_model import Stage2LinearProbingModel
from dataset.fairness_dataset import FairnessDataset, SubgroupBalancedBatchSampler
from losses.detection_loss import DetectionLoss
from utils.training_logger import TrainingLogger


class LayerWiseWarmupScheduler:
    """
    Layer-wise warm-up learning rate scheduler (per-batch)

    각 레이어별로 서로 다른 base_lr을 가지며, warmup 기간 동안
    linear하게 0에서 base_lr까지 증가
    """

    def __init__(self, optimizer, warmup_steps, param_groups_info):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps (int): Warmup step 수
            param_groups_info (list): 각 param group의 정보
                [{'base_lr': float, 'layer_name': str, 'layer_id': int}, ...]
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.param_groups_info = param_groups_info
        self.step_count = 0

    def step(self):
        """매 배치마다 호출하여 learning rate 업데이트"""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Warmup phase: linear하게 LR 증가
            lr_scale = self.step_count / self.warmup_steps

            for param_group, info in zip(self.optimizer.param_groups, self.param_groups_info):
                base_lr = info['base_lr']
                current_lr = base_lr * lr_scale
                param_group['lr'] = current_lr

    def get_last_lr(self):
        """현재 learning rates 반환"""
        return [group['lr'] for group in self.optimizer.param_groups]


class Stage2LayerwiseTrainer:
    """
    Stage 2 Layerwise Learning Rate Decay Trainer

    Features:
    - CLIP visual encoder unfreeze + Layerwise LR decay
    - Stage 1 체크포인트에서 Adapter 가중치 로드
    - Layer별 다른 learning rate 적용 (head가 가장 높고, 하위 layer로 갈수록 decay)
    - LayerWiseWarmupScheduler 지원
    """

    def __init__(self, config):
        """
        Args:
            config (dict): 학습 설정 딕셔너리
        """
        self.config = config

        # 수치 안정성 설정
        torch.backends.cudnn.deterministic = config.get('deterministic', True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_default_dtype(torch.float32)

        # Seed 설정
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device 설정
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Layerwise LR decay 설정
        self.layer_decay = float(config.get('layer_decay', 0.65))
        self.head_lr = float(config.get('head_lr', 1e-4))

        print(f"Layer-wise decay rate: {self.layer_decay}")
        print(f"Head LR: {self.head_lr}")

        # 로그 디렉토리 설정
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_config = config.get('logging', {})
        base_log_dir = log_config.get('log_dir', '/workspace/code/CLIP_stage2/logs')
        experiment_name = log_config.get('experiment_name', 'stage2_full_finetuning_lw')
        self.log_dir = os.path.join(base_log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"Log directory: {self.log_dir}")

        # Logger 초기화
        self.logger = TrainingLogger(self.log_dir, experiment_name)

        # 모델, 옵티마이저, 스케줄러 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.param_groups_info = []  # LayerWiseWarmupScheduler용 정보

    def extract_layer_number(self, param_name):
        """
        파라미터 이름에서 layer 번호 추출

        Returns:
            -1: CLIP embeddings (conv1, class_embedding, positional_embedding)
            0-23: CLIP transformer blocks (resblocks.0 ~ resblocks.23)
            1000: CLIP projection (proj, ln_post)
            2000-2002: Adapter layers (3개 Linear)
            3000-3002: Classifier layers (3개 Linear)
        """
        # CLIP Transformer block patterns
        if 'resblocks' in param_name:
            match = re.search(r'resblocks\.(\d+)', param_name)
            if match:
                return int(match.group(1))

        # CLIP Embedding layers (가장 낮은 레벨)
        if any(x in param_name for x in ['conv1', 'class_embedding', 'positional_embedding']):
            return -1

        # CLIP Projection layer
        if 'proj' in param_name or 'ln_post' in param_name:
            return 1000

        # Adapter layers
        if 'additive_adapter' in param_name:
            # additive_adapter.fc1, fc2, fc3 등을 구분
            if 'fc1' in param_name or 'layer.0' in param_name:
                return 2000
            elif 'fc2' in param_name or 'layer.2' in param_name:
                return 2001
            elif 'fc3' in param_name or 'layer.4' in param_name:
                return 2002
            return 2000  # 기본값

        # Classifier layers
        if 'binary_classifier' in param_name:
            # binary_classifier.0, .3, .6 등을 구분
            match = re.search(r'binary_classifier\.(\d+)', param_name)
            if match:
                layer_idx = int(match.group(1))
                # 0-2: layer 0, 3-5: layer 1, 6+: layer 2
                return 3000 + (layer_idx // 3)
            return 3000  # 기본값

        return 0  # 기본값

    def get_layerwise_params(self):
        """
        Layer별로 다른 learning rate를 적용한 parameter groups 생성

        LR 계산 공식 (layer_decay=0.65, head_lr=1e-4 기준):
        - Classifier: decay_steps=0 → LR=1.00e-4
        - Adapter Layer 2: decay_steps=1 → LR=6.50e-5
        - Adapter Layer 1: decay_steps=2 → LR=4.23e-5
        - Adapter Layer 0: decay_steps=3 → LR=2.75e-5
        - CLIP Projection: decay_steps=4 → LR=1.79e-5
        - CLIP Layer 23: decay_steps=5 → LR=1.16e-5
        - ...
        - CLIP Layer 0: decay_steps=28 → LR=3.50e-9
        - CLIP Embeddings: decay_steps=29 → LR=2.28e-9

        Returns:
            dict: layer_name -> {'params': list, 'lr': float, 'layer_id': int, 'name': str}
        """
        layer_params = {}

        # 1. Classifier layers - 가장 높은 learning rate (decay_steps = 0, 1, 2)
        classifier_layers = {}
        for name, param in self.model.binary_classifier.named_parameters():
            if not param.requires_grad:
                continue
            layer_num = self.extract_layer_number(f'binary_classifier.{name}')
            if layer_num not in classifier_layers:
                classifier_layers[layer_num] = []
            classifier_layers[layer_num].append((name, param))

        for layer_num in sorted(classifier_layers.keys(), reverse=True):
            params_list = [param for _, param in classifier_layers[layer_num]]
            decay_steps = layer_num - 3000  # 3000 → 0, 3001 → 1, 3002 → 2
            layer_lr = self.head_lr * (self.layer_decay ** decay_steps)
            layer_name = f"classifier_layer_{layer_num - 3000}"

            layer_params[layer_name] = {
                'params': params_list,
                'lr': layer_lr,
                'layer_id': layer_num,
                'name': layer_name
            }

        # 2. Adapter layers (decay_steps = 1, 2, 3 from classifier)
        adapter_layers = {}
        for name, param in self.model.additive_adapter.named_parameters():
            if not param.requires_grad:
                continue
            layer_num = self.extract_layer_number(f'additive_adapter.{name}')
            if layer_num not in adapter_layers:
                adapter_layers[layer_num] = []
            adapter_layers[layer_num].append((name, param))

        # Adapter layers의 decay_steps 계산 (classifier 다음)
        classifier_max_decay = len(classifier_layers)
        for layer_num in sorted(adapter_layers.keys(), reverse=True):
            params_list = [param for _, param in adapter_layers[layer_num]]
            adapter_idx = layer_num - 2000  # 2000 → 0, 2001 → 1, 2002 → 2
            decay_steps = classifier_max_decay + (2 - adapter_idx)  # 역순
            layer_lr = self.head_lr * (self.layer_decay ** decay_steps)
            layer_name = f"adapter_layer_{adapter_idx}"

            layer_params[layer_name] = {
                'params': params_list,
                'lr': layer_lr,
                'layer_id': layer_num,
                'name': layer_name
            }

        # 3. CLIP visual encoder layers
        clip_layers = {}
        for name, param in self.model.clip_model.visual.named_parameters():
            if not param.requires_grad:
                continue
            layer_num = self.extract_layer_number(name)
            if layer_num not in clip_layers:
                clip_layers[layer_num] = []
            clip_layers[layer_num].append((name, param))

        # CLIP layers의 decay_steps 계산
        adapter_max_decay = classifier_max_decay + len(adapter_layers)

        # Regular transformer layers (0-23) 중 최대값 찾기
        regular_layers = [k for k in clip_layers.keys() if k >= 0 and k != 1000]
        max_clip_layer = max(regular_layers) if regular_layers else 23

        for layer_num in sorted(clip_layers.keys(), reverse=True):
            params_list = [param for _, param in clip_layers[layer_num]]

            if layer_num == 1000:  # Projection layer
                decay_steps = adapter_max_decay + 1
                layer_name = "clip_projection"
            elif layer_num == -1:  # Embedding layers
                decay_steps = adapter_max_decay + max_clip_layer + 3
                layer_name = "clip_embeddings"
            else:  # Regular transformer blocks (0-23)
                decay_steps = adapter_max_decay + 1 + (max_clip_layer - layer_num) + 1
                layer_name = f"clip_layer_{layer_num}"

            layer_lr = self.head_lr * (self.layer_decay ** decay_steps)

            layer_params[layer_name] = {
                'params': params_list,
                'lr': layer_lr,
                'layer_id': layer_num,
                'name': layer_name
            }

        return layer_params

    def create_model(self):
        """모델 생성, CLIP unfreeze, Stage 1 가중치 로드"""
        model_config = self.config.get('model', {})

        # Stage 2 모델 생성
        self.model = Stage2LinearProbingModel(
            clip_name=model_config.get('clip_name', 'ViT-L/14'),
            feature_dim=model_config.get('feature_dim', 768),
            adapter_hidden_dim=model_config.get('adapter_hidden_dim', 512),
            classifier_hidden_dims=model_config.get('classifier_hidden_dims', [384, 192]),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.1),
            normalize_features=model_config.get('normalize_features', True),
            device=str(self.device),
            clip_download_root=self.config.get('clip_download_root', '/data/cuixinjie/weights')
        )

        # Stage 1 체크포인트 로드
        stage1_checkpoint = self.config.get('stage1_checkpoint')
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            self.model.load_stage1_checkpoint(stage1_checkpoint)
        else:
            self.logger.log_warning(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
            self.logger.log_warning("Using randomly initialized Adapter weights")

        # CLIP visual encoder unfreeze (Full fine-tuning)
        freeze_clip = self.config.get('freeze_clip', False)
        if not freeze_clip:
            self.model.unfreeze_clip()
            self.logger.log_info("CLIP visual encoder unfrozen for full fine-tuning")

        # Adapter unfreeze
        freeze_adapter = self.config.get('freeze_adapter', False)
        if not freeze_adapter:
            self.model.unfreeze_adapter()
            self.logger.log_info("Additive Adapter unfrozen for fine-tuning")

        self.model.to(self.device)
        self.model = self.model.float()

        # 모든 파라미터가 Float32인지 확인
        for name, param in self.model.named_parameters():
            if param.dtype != torch.float32:
                self.logger.log_warning(f"Parameter {name} is not float32: {param.dtype}")
                param.data = param.data.float()

        # 파라미터 정보 출력
        self.model.print_trainable_parameters()
        self.logger.log_model_info(self.model)

        return self.model

    def create_optimizer(self):
        """Layerwise learning rate를 적용한 Optimizer 생성"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()

        # Layerwise parameter groups 생성
        layerwise_params = self.get_layerwise_params()

        # Optimizer parameter groups 구성
        optimizer_params = []
        self.param_groups_info = []

        # Layer별로 정렬하여 출력 (layer_id 역순)
        for layer_name in sorted(layerwise_params.keys(),
                                  key=lambda x: layerwise_params[x]['layer_id'],
                                  reverse=True):
            layer_info = layerwise_params[layer_name]

            optimizer_params.append({
                'params': layer_info['params'],
                'lr': layer_info['lr']
            })
            self.param_groups_info.append({
                'base_lr': layer_info['lr'],
                'layer_name': layer_name,
                'layer_id': layer_info['layer_id']
            })

            num_params = sum(p.numel() for p in layer_info['params'])
            print(f"Layer {layer_name}: LR={layer_info['lr']:.2e}, Params={num_params:,}")

        # Optimizer 생성
        weight_decay = float(opt_config.get('weight_decay', 0.01))

        if opt_type == 'adamw':
            betas = tuple(opt_config.get('betas', [0.9, 0.999]))
            eps = float(opt_config.get('eps', 1e-8))
            self.optimizer = optim.AdamW(
                optimizer_params,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        elif opt_type == 'adam':
            betas = tuple(opt_config.get('betas', [0.9, 0.999]))
            eps = float(opt_config.get('eps', 1e-8))
            self.optimizer = optim.Adam(
                optimizer_params,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        elif opt_type == 'sgd':
            momentum = float(opt_config.get('momentum', 0.9))
            self.optimizer = optim.SGD(
                optimizer_params,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        # Learning rate 분포 요약
        lrs = [group['lr'] for group in optimizer_params]
        self.logger.log_info(f"Optimizer: {opt_type}")
        self.logger.log_info(f"Total parameter groups: {len(optimizer_params)}")
        self.logger.log_info(f"Learning rate range: {min(lrs):.2e} - {max(lrs):.2e}")
        self.logger.log_info(f"LR decay factor: {self.layer_decay}")

        return self.optimizer

    def create_scheduler(self):
        """Learning Rate Scheduler 생성"""
        sched_config = self.config.get('scheduler', {})

        if not sched_config.get('use_scheduler', True):
            self.scheduler = None
            return None

        sched_name = sched_config.get('name', 'LayerWiseWarmupScheduler')
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 10)

        if sched_name == 'LayerWiseWarmupScheduler':
            warmup_steps = sched_config.get('warmup_steps', 100)
            self.scheduler = LayerWiseWarmupScheduler(
                self.optimizer,
                warmup_steps,
                self.param_groups_info
            )
            self.logger.log_info(f"Scheduler: LayerWiseWarmupScheduler (warmup_steps={warmup_steps})")

        elif sched_name == 'StepLR':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 5),
                gamma=sched_config.get('gamma', 0.5)
            )
            self.logger.log_info(f"Scheduler: StepLR")

        elif sched_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=sched_config.get('min_lr', 1e-8)
            )
            self.logger.log_info(f"Scheduler: CosineAnnealingLR")

        else:
            self.logger.log_warning(f"Unknown scheduler: {sched_name}, using no scheduler")
            self.scheduler = None

        return self.scheduler

    def create_loss_function(self):
        """Loss Function 생성"""
        loss_config = self.config.get('loss', {})

        self.loss_fn = DetectionLoss(
            loss_type=loss_config.get('type', 'cross_entropy'),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            class_weights=loss_config.get('class_weights')
        )

        self.logger.log_info(f"Loss: {loss_config.get('type', 'cross_entropy')}")

        return self.loss_fn

    def create_dataloaders(self):
        """DataLoader 생성"""
        dataset_config = self.config.get('dataset', {})
        training_config = self.config.get('training', {})

        # 데이터셋 설정 병합
        train_config = {**self.config, **dataset_config}

        # Train 데이터셋
        train_dataset = FairnessDataset(
            config=train_config,
            mode='train'
        )

        # Validation 데이터셋
        val_config = train_config.copy()
        val_dataset = FairnessDataset(
            config=val_config,
            mode='validation'
        )

        # Subgroup Balanced Batch Sampler 사용 여부
        use_subgroup_sampler = dataset_config.get('use_subgroup_sampler', False)
        batch_size = training_config.get('train_batch_size', 16)  # Full fine-tuning은 메모리 사용량 증가

        if use_subgroup_sampler:
            # Batch-level subgroup 균형 샘플링
            batch_sampler = SubgroupBalancedBatchSampler(
                train_dataset.subgroup_list,
                batch_size=batch_size,
                drop_last=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=training_config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=train_dataset.collate_fn
            )
            self.logger.log_info(f"Using SubgroupBalancedBatchSampler (batch_size={batch_size})")
        else:
            # 기본 DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=training_config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=train_dataset.collate_fn,
                drop_last=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get('val_batch_size', 16),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
            drop_last=False
        )

        self.logger.log_info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        self.logger.log_info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """한 에포크 학습 (per-batch scheduler step 포함)"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0
        skipped_batches = 0

        log_config = self.config.get('logging', {})
        print_freq = log_config.get('print_freq', 50)
        grad_clip = self.config.get('gradient_clip_max_norm', 1.0)

        for batch_idx, data_dict in enumerate(train_loader):
            # 데이터를 device로 이동
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            try:
                pred_dict = self.model(data_dict, inference=False)
            except Exception as e:
                self.logger.log_error(f"Forward pass error at batch {batch_idx}", e)
                skipped_batches += 1
                continue

            # Loss 계산
            try:
                cls_logits = pred_dict['cls']
                labels = data_dict['label']
                loss = self.loss_fn(cls_logits, labels)
            except Exception as e:
                self.logger.log_error(f"Loss calculation error at batch {batch_idx}", e)
                skipped_batches += 1
                continue

            # NaN/Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.log_warning(f"NaN/Inf loss at batch {batch_idx}, skipping...")
                skipped_batches += 1
                continue

            # Backward pass
            try:
                loss.backward()
            except Exception as e:
                self.logger.log_error(f"Backward pass error at batch {batch_idx}", e)
                skipped_batches += 1
                continue

            # Gradient 체크 및 클리핑
            total_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if grad_clip > 0 and total_norm > grad_clip * 5:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            # Optimizer step
            self.optimizer.step()

            # Per-batch scheduler step (LayerWiseWarmupScheduler)
            if self.scheduler and isinstance(self.scheduler, LayerWiseWarmupScheduler):
                self.scheduler.step()

            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = torch.max(cls_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 로깅
            if batch_idx % print_freq == 0:
                current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                accuracy = 100. * correct / total
                self.logger.log_batch_progress(
                    epoch, batch_idx, len(train_loader),
                    loss.item(), accuracy,
                    current_lrs=current_lrs,
                    log_interval=1
                )

        processed_batches = len(train_loader) - skipped_batches
        if processed_batches > 0:
            avg_loss = total_loss / processed_batches
            accuracy = 100. * correct / total
        else:
            avg_loss = float('inf')
            accuracy = 0.0

        if skipped_batches > 0:
            self.logger.log_warning(f"Epoch {epoch}: {skipped_batches}/{len(train_loader)} batches skipped")

        return avg_loss, accuracy

    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data_dict in val_loader:
                # 데이터를 device로 이동
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                # Forward pass
                pred_dict = self.model(data_dict, inference=True)

                # Loss 계산
                cls_logits = pred_dict['cls']
                labels = data_dict['label']
                loss = self.loss_fn(cls_logits, labels)
                total_loss += loss.item()

        # 메트릭 계산
        metrics = self.model.get_test_metrics()
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

        return avg_loss, metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler and hasattr(self.scheduler, 'state_dict') else None,
            'metrics': metrics,
            'config': self.config,
            'model_type': 'stage2_full_finetuning_layerwise',
            'layer_decay': self.layer_decay,
            'head_lr': self.head_lr
        }

        # 최신 체크포인트 저장
        latest_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Best 모델 저장
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

            # 점수가 포함된 파일명으로도 저장
            auc = metrics.get('auc', 0)
            acc = metrics.get('acc', 0)
            eer = metrics.get('eer', 0)
            score_filename = f"best_ep{epoch}_auc{auc:.4f}_acc{acc:.4f}_eer{eer:.4f}.pth"
            score_path = os.path.join(self.log_dir, score_filename)
            torch.save(checkpoint, score_path)

            # 최대 3개만 유지
            self._cleanup_old_best_checkpoints(max_keep=3)

            self.logger.log_info(f"Best model saved: {score_filename}")

        return latest_path

    def _cleanup_old_best_checkpoints(self, max_keep=3):
        """오래된 best 체크포인트 삭제"""
        import glob

        pattern = os.path.join(self.log_dir, 'best_ep*.pth')
        best_files = glob.glob(pattern)

        if len(best_files) <= max_keep:
            return

        best_files.sort(key=lambda x: os.path.getmtime(x))
        files_to_delete = best_files[:-max_keep]

        for f in files_to_delete:
            os.remove(f)
            self.logger.log_info(f"Removed old checkpoint: {os.path.basename(f)}")

    def train(self):
        """전체 학습 프로세스"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 2 Layerwise Full Fine-tuning Training Started")
        self.logger.log_info("=" * 60)

        # 모델, 옵티마이저, 스케줄러, Loss 생성
        self.create_model()
        self.create_optimizer()
        self.create_scheduler()
        self.create_loss_function()

        # 설정 로깅
        self.logger.log_config(self.config)

        # DataLoader 생성
        train_loader, val_loader = self.create_dataloaders()

        # 학습 파라미터
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 10)
        eval_config = self.config.get('evaluation', {})
        metric_for_best = eval_config.get('metric_for_best', 'auc')

        best_metric = 0.0 if metric_for_best != 'eer' else 1.0
        start_time = time.time()

        self.logger.log_training_start(num_epochs, len(train_loader))

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            self.logger.log_epoch_start(epoch, num_epochs)

            # 학습
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 검증
            val_loss, val_metrics = self.validate(val_loader)

            # Epoch-wise scheduler step (LayerWiseWarmupScheduler 제외)
            if self.scheduler and not isinstance(self.scheduler, LayerWiseWarmupScheduler):
                self.scheduler.step()

            # Best 모델 판단
            current_metric = val_metrics.get(metric_for_best, 0)
            if metric_for_best == 'eer':
                is_best = current_metric < best_metric
            else:
                is_best = current_metric > best_metric

            if is_best:
                best_metric = current_metric

            # 체크포인트 저장
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # 로깅
            elapsed_time = time.time() - epoch_start
            self.logger.log_epoch_results(
                epoch, train_loss, train_acc,
                val_loss, val_metrics,
                is_best=is_best,
                elapsed_time=elapsed_time
            )

        # 학습 완료
        total_time = time.time() - start_time
        self.logger.log_training_complete(
            best_metric,
            best_metric_name=metric_for_best.upper(),
            total_time=total_time,
            final_model_path=os.path.join(self.log_dir, 'checkpoint_best.pth')
        )

        self.logger.close()

        return best_metric


if __name__ == "__main__":
    # 테스트 코드
    print("Stage 2 Layerwise Trainer Test")

    config = {
        'model': {
            'clip_name': 'ViT-L/14',
            'feature_dim': 768,
        },
        'training': {
            'num_epochs': 1,
            'train_batch_size': 8,
            'val_batch_size': 8,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 0.0001,
        },
        'layer_decay': 0.65,
        'head_lr': 0.0001,
        'freeze_clip': False,
        'freeze_adapter': False,
        'dataset': {
            'fairness_root': '/workspace/datasets/fairness',
            'train_dataset': ['ff++'],
            'validation_dataset': ['ff++'],
            'resolution': 256,
        },
        'logging': {
            'log_dir': '/tmp/stage2_lw_test',
            'experiment_name': 'test',
        }
    }

    trainer = Stage2LayerwiseTrainer(config)
    print("Trainer created successfully")
