"""
Stage 2 Independent Adapter Trainer
Independent Dual Adapter with Cross-Attention Fusion 학습을 위한 Trainer

특징:
- Stage 1 Adapter (Frozen): Global Fairness
- Stage 2 Adapter (Trainable): Detection + Local Fairness
- Cross-Attention Fusion: Detection queries Fairness
- Dynamic Loss Weighting: Uncertainty-based automatic loss balancing
- Stage 2 Fairness Loss: Real/Fake 각 클래스 내에서 subgroup 간 분포 정렬
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import math

from model.stage2_independent_adapter_model import Stage2IndependentAdapterModel
from dataset.fairness_dataset import FairnessDataset, SubgroupBalancedBatchSampler
from losses.detection_loss import DetectionLoss
from losses.stage2_fairness_loss import Stage2FairnessLoss
from losses.dynamic_loss_weighting import DynamicFairnessLossWeighting, FixedLossWeighting
from utils.training_logger import TrainingLogger


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Scheduler (per-epoch)"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = max(self.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class Stage2IndependentAdapterTrainer:
    """
    Stage 2 Independent Adapter Trainer

    Features:
    - Independent Dual Adapter (Stage 1 Frozen, Stage 2 Trainable)
    - Cross-Attention Fusion with Dynamic Gate
    - Dynamic Loss Weighting (Detection + Fairness)
    - Stage 2 Fairness Loss (per-class subgroup alignment)
    - Comprehensive monitoring (gate, loss weights, fairness metrics)
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

        # 로그 디렉토리 설정
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_config = config.get('logging', {})
        base_log_dir = log_config.get('log_dir', '/workspace/code/CLIP_stage2/logs')
        experiment_name = log_config.get('experiment_name', 'stage2_independent')
        self.log_dir = os.path.join(base_log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Logger 초기화
        self.logger = TrainingLogger(self.log_dir, experiment_name)

        # 모델, 옵티마이저, 스케줄러 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.cls_loss_fn = None
        self.fairness_loss_fn = None
        self.loss_weighting = None

        # 학습 데이터셋 참조 (subgroup 정보 접근용)
        self.train_dataset = None

    def create_model(self):
        """모델 생성 및 Stage 1 가중치 로드"""
        model_config = self.config.get('model', {})

        # Stage 2 Independent Adapter 모델 생성
        self.model = Stage2IndependentAdapterModel(
            clip_name=model_config.get('clip_name', 'ViT-L/14'),
            feature_dim=model_config.get('feature_dim', 768),
            stage1_hidden_dim=model_config.get('stage1_hidden_dim', 512),
            stage2_hidden_dim=model_config.get('stage2_hidden_dim', 384),
            classifier_hidden_dims=model_config.get('classifier_hidden_dims', [384, 192]),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.1),
            fusion_num_heads=model_config.get('fusion_num_heads', 8),
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
            self.logger.log_warning("Using randomly initialized Stage 1 Adapter weights")

        self.model.to(self.device)
        self.model = self.model.float()

        # 파라미터 정보 출력
        param_info = self.model.print_trainable_parameters()
        self.logger.log_model_info(self.model)

        return self.model

    def create_optimizer(self):
        """Optimizer 생성"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()

        # 학습할 파라미터: Stage 2 Adapter + Fusion + Classifier
        params = list(self.model.get_trainable_params())

        # Loss weighting 파라미터도 추가 (DynamicFairnessLossWeighting인 경우)
        if self.loss_weighting is not None and hasattr(self.loss_weighting, 'parameters'):
            params.extend(list(self.loss_weighting.parameters()))

        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_type == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=opt_config.get('lr', 0.01),
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        self.logger.log_info(f"Optimizer: {opt_type}, LR: {opt_config.get('lr', 1e-4)}")

        return self.optimizer

    def create_scheduler(self):
        """Learning Rate Scheduler 생성"""
        sched_config = self.config.get('scheduler', {})

        if not sched_config.get('use_scheduler', True):
            self.scheduler = None
            return None

        sched_name = sched_config.get('name', 'CosineAnnealingWarmRestarts')
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 30)

        if sched_name == 'CosineAnnealingWarmRestarts':
            # Cosine Annealing with Warm Restarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 5),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_name == 'WarmupCosine':
            warmup_epochs = sched_config.get('warmup_epochs', 1)
            min_lr = sched_config.get('eta_min', 1e-6)
            base_lr = self.config.get('optimizer', {}).get('lr', 1e-4)
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs,
                min_lr_ratio=min_lr / base_lr
            )
        else:
            self.logger.log_warning(f"Unknown scheduler: {sched_name}, using no scheduler")
            self.scheduler = None

        if self.scheduler:
            self.logger.log_info(f"Scheduler: {sched_name}")

        return self.scheduler

    def create_loss_functions(self):
        """Loss Functions 생성"""
        loss_config = self.config.get('loss', {})

        # Classification Loss
        self.cls_loss_fn = DetectionLoss(
            loss_type=loss_config.get('type', 'cross_entropy'),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0),
            class_weights=loss_config.get('class_weights')
        )

        # Fairness Loss
        fairness_config = self.config.get('fairness_loss', {})
        self.fairness_loss_fn = Stage2FairnessLoss(
            sinkhorn_blur=fairness_config.get('sinkhorn_blur', 1e-4),
            sinkhorn_p=fairness_config.get('sinkhorn_p', 2),
            scaling=fairness_config.get('scaling', 0.9),
            min_samples_per_subgroup=fairness_config.get('min_samples_per_subgroup', 2),
            num_subgroups=fairness_config.get('num_subgroups', 8)
        )

        # Loss Weighting
        weighting_config = self.config.get('loss_weighting', {})
        weighting_type = weighting_config.get('type', 'dynamic')

        if weighting_type == 'dynamic':
            self.loss_weighting = DynamicFairnessLossWeighting(
                init_log_var_cls=weighting_config.get('init_log_var_cls', 0.0),
                init_log_var_fair=weighting_config.get('init_log_var_fair', 0.0)
            )
            self.loss_weighting.to(self.device)
        elif weighting_type == 'fixed':
            self.loss_weighting = FixedLossWeighting(
                lambda_cls=weighting_config.get('lambda_cls', 1.0),
                lambda_fair=weighting_config.get('lambda_fair', 0.1)
            )
        else:
            raise ValueError(f"Unknown loss weighting type: {weighting_type}")

        self.logger.log_info(f"Classification Loss: {loss_config.get('type', 'cross_entropy')}")
        self.logger.log_info(f"Fairness Loss: Stage2FairnessLoss (sinkhorn)")
        self.logger.log_info(f"Loss Weighting: {weighting_type}")

        return self.cls_loss_fn, self.fairness_loss_fn, self.loss_weighting

    def create_dataloaders(self):
        """DataLoader 생성"""
        dataset_config = self.config.get('dataset', {})
        training_config = self.config.get('training', {})

        # 데이터셋 설정 병합
        train_config = {**self.config, **dataset_config}

        # Train 데이터셋
        self.train_dataset = FairnessDataset(
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
        use_subgroup_sampler = dataset_config.get('use_subgroup_sampler', True)
        batch_size = training_config.get('train_batch_size', 64)

        if use_subgroup_sampler:
            batch_sampler = SubgroupBalancedBatchSampler(
                self.train_dataset.subgroup_list,
                batch_size=batch_size,
                drop_last=True
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=training_config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=self.train_dataset.collate_fn
            )
            self.logger.log_info(f"Using SubgroupBalancedBatchSampler (batch_size={batch_size})")
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=training_config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=self.train_dataset.collate_fn,
                drop_last=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get('val_batch_size', 64),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
            drop_last=False
        )

        self.logger.log_info(f"Train: {len(self.train_dataset)} samples, {len(train_loader)} batches")
        self.logger.log_info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

        return train_loader, val_loader

    def _get_batch_subgroups(self, batch_indices):
        """배치 내 샘플들의 subgroup 정보 반환"""
        if self.train_dataset is None:
            return None

        subgroups = []
        for idx in batch_indices:
            if idx < len(self.train_dataset.subgroup_list):
                subgroups.append(self.train_dataset.subgroup_list[idx])
            else:
                subgroups.append(-1)  # Unknown

        return torch.LongTensor(subgroups)

    def train_epoch(self, train_loader, epoch):
        """한 에포크 학습"""
        self.model.train()

        # Backbone은 항상 eval 모드 유지
        self.model.clip_model.eval()
        self.model.stage1_adapter.eval()

        total_loss = 0
        total_cls_loss = 0
        total_fair_loss = 0
        total_gate = 0
        total_lambda_fair = 0
        correct = 0
        total = 0

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
            pred_dict = self.model(data_dict, inference=False)

            # 1. Classification Loss
            cls_logits = pred_dict['cls']
            labels = data_dict['label']
            cls_loss = self.cls_loss_fn(cls_logits, labels)

            # 2. Fairness Loss (fused_features 사용)
            fused_features = pred_dict['fused_features_norm']

            # Subgroup 정보는 이제 data_dict에서 직접 가져옴
            subgroups = data_dict.get('subgroup')
            fair_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            fair_info = {}

            if subgroups is not None:
                try:
                    fair_loss, fair_info = self.fairness_loss_fn(fused_features, labels, subgroups)
                except Exception as e:
                    # Fairness loss 계산 실패 시 0으로 설정
                    fair_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # 3. Dynamic Loss Weighting
            total_loss_batch, lambda_fair, weighting_info = self.loss_weighting(cls_loss, fair_loss)

            # NaN 체크
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                self.logger.log_warning(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
                continue

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.optimizer.step()

            # 통계 업데이트
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_fair_loss += fair_loss.item() if isinstance(fair_loss, torch.Tensor) else fair_loss

            # Gate 통계
            gate_mean = pred_dict['gate'].mean().item()
            total_gate += gate_mean

            # Lambda fair 통계
            if isinstance(lambda_fair, torch.Tensor):
                total_lambda_fair += lambda_fair.item()
            else:
                total_lambda_fair += lambda_fair

            _, predicted = torch.max(cls_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 로깅
            if batch_idx % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                accuracy = 100. * correct / total

                print(f"  Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}]  "
                      f"Loss: {total_loss_batch.item():.4f} (cls: {cls_loss.item():.4f}, fair: {fair_loss.item():.4f})  "
                      f"Acc: {accuracy:.2f}%  Gate: {gate_mean:.3f}  λ_fair: {lambda_fair:.3f}  "
                      f"LR: {current_lr:.6f}")

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0
        avg_fair_loss = total_fair_loss / num_batches if num_batches > 0 else 0
        avg_gate = total_gate / num_batches if num_batches > 0 else 0
        avg_lambda_fair = total_lambda_fair / num_batches if num_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0

        train_stats = {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'fair_loss': avg_fair_loss,
            'accuracy': accuracy,
            'gate_mean': avg_gate,
            'lambda_fair': avg_lambda_fair,
        }

        return train_stats

    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data_dict in val_loader:
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                pred_dict = self.model(data_dict, inference=True)

                cls_logits = pred_dict['cls']
                labels = data_dict['label']
                loss = self.cls_loss_fn(cls_logits, labels)
                total_loss += loss.item()

        metrics = self.model.get_test_metrics()
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        metrics['loss'] = avg_loss

        return metrics

    def evaluate_test_datasets(self, test_datasets: list = None) -> dict:
        """여러 테스트 데이터셋 평가

        Args:
            test_datasets: 평가할 데이터셋 이름 리스트 (기본값: ['celebdf', 'dfd', 'dfdc'])

        Returns:
            dict: 데이터셋별 메트릭 딕셔너리
        """
        if test_datasets is None:
            test_datasets = ['celebdf', 'dfd', 'dfdc']

        results = {}
        for dataset_name in test_datasets:
            try:
                dataloader = self._create_test_dataloader(dataset_name)
                if dataloader is not None:
                    metrics = self._evaluate_single_dataset(dataloader, dataset_name)
                    results[dataset_name] = metrics
                else:
                    self.logger.log_warning(f"Test dataloader for '{dataset_name}' is None, skipping")
            except Exception as e:
                self.logger.log_warning(f"Failed to evaluate {dataset_name}: {str(e)}")

        return results

    def _create_test_dataloader(self, dataset_name: str) -> DataLoader:
        """단일 테스트 DataLoader 생성

        Args:
            dataset_name: 테스트 데이터셋 이름

        Returns:
            DataLoader 또는 None
        """
        from dataset.fairness_dataset import FairnessDataset

        dataset_config = self.config.get('dataset', {})
        training_config = self.config.get('training', {})

        # 테스트용 config 생성
        test_config = {**self.config, **dataset_config}
        test_config['test_dataset'] = [dataset_name]

        try:
            test_dataset = FairnessDataset(
                config=test_config,
                mode='test'
            )

            if len(test_dataset) == 0:
                self.logger.log_warning(f"Test dataset '{dataset_name}' is empty")
                return None

            test_loader = DataLoader(
                test_dataset,
                batch_size=training_config.get('val_batch_size', 64),
                shuffle=False,
                num_workers=training_config.get('num_workers', 4),
                pin_memory=True,
                collate_fn=test_dataset.collate_fn,
                drop_last=False
            )

            return test_loader

        except Exception as e:
            self.logger.log_warning(f"Failed to create test dataloader for {dataset_name}: {str(e)}")
            return None

    def _evaluate_single_dataset(self, dataloader: DataLoader, dataset_name: str) -> dict:
        """단일 데이터셋 평가 (AUC, ACC, EER, AP)

        Args:
            dataloader: 테스트 DataLoader
            dataset_name: 데이터셋 이름

        Returns:
            dict: 메트릭 딕셔너리
        """
        self.model.eval()

        with torch.no_grad():
            for data_dict in dataloader:
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                _ = self.model(data_dict, inference=True)

        # 모델의 테스트 메트릭 수집
        metrics = self.model.get_test_metrics()
        metrics['num_samples'] = len(dataloader.dataset)

        return metrics

    def save_checkpoint(self, epoch, metrics, train_stats, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler and hasattr(self.scheduler, 'state_dict') else None,
            'loss_weighting_state_dict': self.loss_weighting.state_dict() if hasattr(self.loss_weighting, 'state_dict') else None,
            'metrics': metrics,
            'train_stats': train_stats,
            'config': self.config,
        }

        # 최신 체크포인트 저장
        latest_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Best 모델 저장
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

            auc = metrics.get('auc', 0)
            acc = metrics.get('acc', 0)
            score_filename = f"best_ep{epoch}_auc{auc:.4f}_acc{acc:.4f}.pth"
            score_path = os.path.join(self.log_dir, score_filename)
            torch.save(checkpoint, score_path)

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
        self.logger.log_info("Stage 2 Independent Adapter Training Started")
        self.logger.log_info("=" * 60)

        # 모델 생성
        self.create_model()

        # Loss functions 생성 (optimizer 전에)
        self.create_loss_functions()

        # 옵티마이저, 스케줄러 생성
        self.create_optimizer()
        self.create_scheduler()

        # 설정 로깅
        self.logger.log_config(self.config)

        # DataLoader 생성
        train_loader, val_loader = self.create_dataloaders()

        # 학습 파라미터
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 30)
        eval_config = self.config.get('evaluation', {})
        metric_for_best = eval_config.get('metric_for_best', 'auc')

        best_metric = 0.0 if metric_for_best != 'eer' else 1.0
        start_time = time.time()

        self.logger.log_info(f"\nStarting training for {num_epochs} epochs")
        self.logger.log_info(f"Best metric: {metric_for_best}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # 학습
            train_stats = self.train_epoch(train_loader, epoch)

            # 검증
            val_metrics = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, WarmupCosineScheduler):
                    self.scheduler.step(epoch)
                else:
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
            self.save_checkpoint(epoch, val_metrics, train_stats, is_best=is_best)

            # 로깅
            elapsed_time = time.time() - epoch_start

            print(f"\n[Epoch {epoch} Summary]")
            print(f"  Train - Loss: {train_stats['loss']:.4f}, "
                  f"Cls: {train_stats['cls_loss']:.4f}, "
                  f"Fair: {train_stats['fair_loss']:.6f}, "
                  f"Acc: {train_stats['accuracy']:.2f}%")
            print(f"  Train - Gate: {train_stats['gate_mean']:.3f}, "
                  f"λ_fair: {train_stats['lambda_fair']:.4f}")
            print(f"  Val   - AUC: {val_metrics['auc']:.4f}, "
                  f"ACC: {val_metrics['acc']:.4f}, "
                  f"EER: {val_metrics['eer']:.4f}")
            print(f"  Time: {elapsed_time:.1f}s, Best {metric_for_best.upper()}: {best_metric:.4f}")

            # CSV/Summary 로깅 - log_epoch_results 호출
            self.logger.log_epoch_results(
                epoch=epoch,
                train_loss=train_stats['loss'],
                train_acc=train_stats['accuracy'],
                val_loss=val_metrics.get('loss', 0),
                val_metrics={
                    'auc': val_metrics.get('auc', 0),
                    'acc': val_metrics.get('acc', 0),
                    'eer': val_metrics.get('eer', 0),
                    'ap': val_metrics.get('ap', 0)
                },
                is_best=is_best,
                elapsed_time=elapsed_time
            )

            if is_best:
                print(f"  *** New best model! ***")

            # 테스트 데이터셋 평가 (Best 모델일 때 또는 N 에포크마다)
            test_every_n_epochs = eval_config.get('test_every_n_epochs', 5)
            if is_best or (epoch % test_every_n_epochs == 0):
                test_datasets = eval_config.get('test_datasets', ['celebdf', 'dfd', 'dfdc'])
                if test_datasets:
                    print(f"\n  [Test Evaluation]")
                    test_results = self.evaluate_test_datasets(test_datasets)
                    for ds_name, ds_metrics in test_results.items():
                        self.logger.log_test_dataset_results(epoch, ds_name, ds_metrics)

        # 학습 완료
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {metric_for_best.upper()}: {best_metric:.4f}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Model saved at: {self.log_dir}")
        print(f"{'='*60}")

        # log_training_complete 호출
        self.logger.log_training_complete(
            best_metric_value=best_metric,
            best_metric_name=metric_for_best.upper(),
            total_time=total_time,
            final_model_path=os.path.join(self.log_dir, 'checkpoint_best.pth')
        )

        self.logger.close()

        return best_metric


if __name__ == "__main__":
    print("Stage 2 Independent Adapter Trainer Test")

    config = {
        'model': {
            'clip_name': 'ViT-L/14',
            'feature_dim': 768,
            'stage1_hidden_dim': 512,
            'stage2_hidden_dim': 384,
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
        'dataset': {
            'fairness_root': '/workspace/datasets/fairness',
            'train_dataset': ['ff++'],
            'validation_dataset': ['ff++'],
            'resolution': 256,
        },
        'logging': {
            'log_dir': '/tmp/stage2_independent_test',
            'experiment_name': 'test',
        }
    }

    trainer = Stage2IndependentAdapterTrainer(config)
    print("Trainer created successfully")
