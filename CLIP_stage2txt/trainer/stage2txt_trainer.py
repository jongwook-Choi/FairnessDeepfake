"""
Stage 2 TXT Trainer
Linear Probing / Fine-tuning 학습을 위한 Trainer 클래스
Stage1_txt에서 학습된 Adapter + Text Anchors 로드
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
import math

from model.stage2txt_model import Stage2TxtModel
from dataset.fairness_dataset import FairnessDataset, SubgroupBalancedBatchSampler
from losses.detection_loss import DetectionLoss, CombinedDetectionLoss
from utils.training_logger import TrainingLogger


class WarmupCosineScheduler(LambdaLR):
    """Warmup + Cosine Annealing Scheduler"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.0):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, cosine_decay)

        super().__init__(optimizer, lr_lambda)


class Stage2TxtTrainer:
    """
    Stage 2 TXT Linear Probing / Fine-tuning Trainer

    Features:
    - CLIP + Additive Adapter (frozen) + Binary Classifier (trainable)
    - Stage 1_txt 체크포인트에서 Adapter + Text Anchors 가중치 로드
    - Warmup scheduler 지원
    - Gradient clipping
    - AUC, EER, AP 메트릭 계산
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
        base_log_dir = log_config.get('log_dir', '/workspace/code/CLIP_stage2txt/logs')
        experiment_name = log_config.get('experiment_name', 'stage2txt')
        self.log_dir = os.path.join(base_log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Logger 초기화
        self.logger = TrainingLogger(self.log_dir, experiment_name)

        # 모델, 옵티마이저, 스케줄러 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

    def create_model(self):
        """모델 생성 및 Stage 1_txt 가중치 로드"""
        model_config = self.config.get('model', {})

        # Stage 2 TXT 모델 생성
        self.model = Stage2TxtModel(
            clip_name=model_config.get('clip_name', 'ViT-L/14'),
            feature_dim=model_config.get('feature_dim', 768),
            adapter_hidden_dim=model_config.get('adapter_hidden_dim', 512),
            classifier_hidden_dims=model_config.get('classifier_hidden_dims', [384, 192]),
            num_classes=model_config.get('num_classes', 2),
            num_subgroups=model_config.get('num_subgroups', 8),
            dropout=model_config.get('dropout', 0.1),
            normalize_features=model_config.get('normalize_features', True),
            device=str(self.device),
            clip_download_root=self.config.get('clip_download_root', '/data/cuixinjie/weights')
        )

        # Stage 1_txt 체크포인트 로드 (Adapter + Text Anchors)
        stage1txt_checkpoint = self.config.get('stage1txt_checkpoint')
        if stage1txt_checkpoint and os.path.exists(stage1txt_checkpoint):
            self.model.load_stage1txt_checkpoint(stage1txt_checkpoint)
        else:
            self.logger.log_warning(f"Stage 1_txt checkpoint not found: {stage1txt_checkpoint}")
            self.logger.log_warning("Using randomly initialized Adapter weights (no Text Anchors)")

        # Adapter 동결 여부
        freeze_adapter = self.config.get('freeze_adapter', True)
        if freeze_adapter:
            self.model.freeze_backbone()

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

        # 학습할 파라미터 선택
        freeze_adapter = self.config.get('freeze_adapter', True)
        if freeze_adapter:
            params = self.model.get_trainable_params()
        else:
            params = self.model.get_all_trainable_params()

        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config.get('lr', 0.001),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=opt_config.get('lr', 0.001),
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

        self.logger.log_info(f"Optimizer: {opt_type}, LR: {opt_config.get('lr', 0.001)}")

        return self.optimizer

    def create_scheduler(self):
        """Learning Rate Scheduler 생성"""
        sched_config = self.config.get('scheduler', {})

        if not sched_config.get('use_scheduler', False):
            self.scheduler = None
            return None

        sched_name = sched_config.get('name', 'StepLR')
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 10)

        if sched_name == 'StepLR':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 5),
                gamma=sched_config.get('gamma', 0.5)
            )
        elif sched_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_name == 'WarmupCosine':
            warmup_epochs = sched_config.get('warmup_epochs', 1)
            min_lr = sched_config.get('min_lr', 1e-6)
            base_lr = self.config.get('optimizer', {}).get('lr', 0.001)
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
        batch_size = training_config.get('train_batch_size', 64)

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
            batch_size=training_config.get('val_batch_size', 64),
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
        """한 에포크 학습"""
        self.model.train()

        # Backbone은 항상 eval 모드 유지
        self.model.clip_model.eval()
        if self.config.get('freeze_adapter', True):
            self.model.additive_adapter.eval()

        total_loss = 0
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

            # Loss 계산
            cls_logits = pred_dict['cls']
            labels = data_dict['label']
            loss = self.loss_fn(cls_logits, labels)

            # NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.log_warning(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.optimizer.step()

            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = torch.max(cls_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 로깅
            if batch_idx % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                accuracy = 100. * correct / total
                self.logger.log_batch_progress(
                    epoch, batch_idx, len(train_loader),
                    loss.item(), accuracy,
                    current_lrs=[current_lr],
                    log_interval=1
                )

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0

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
        """체크포인트 저장

        저장 방식:
        - checkpoint_latest.pth: 항상 최신 상태 저장
        - checkpoint_best.pth: best 모델일 때만 저장
        - best_ep{epoch}_auc{auc}_acc{acc}.pth: 점수 포함 파일명 (최대 3개 유지)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
        }

        # Text Anchors도 함께 저장
        if self.model.text_anchors is not None:
            checkpoint['text_anchors'] = self.model.text_anchors

        # 최신 체크포인트 저장 (latest)
        latest_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Best 모델 저장
        if is_best:
            # 기본 best 체크포인트
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

            # 점수가 포함된 파일명으로도 저장
            auc = metrics.get('auc', 0)
            acc = metrics.get('accuracy', 0)
            eer = metrics.get('eer', 0)
            score_filename = f"best_ep{epoch}_auc{auc:.4f}_acc{acc:.4f}_eer{eer:.4f}.pth"
            score_path = os.path.join(self.log_dir, score_filename)
            torch.save(checkpoint, score_path)

            # 최대 3개만 유지 (오래된 파일 삭제)
            self._cleanup_old_best_checkpoints(max_keep=3)

            self.logger.log_info(f"Best model saved: {score_filename}")

        return latest_path

    def _cleanup_old_best_checkpoints(self, max_keep=3):
        """오래된 best 체크포인트 삭제 (최대 max_keep개 유지)"""
        import glob

        # best_ep*.pth 파일들 찾기
        pattern = os.path.join(self.log_dir, 'best_ep*.pth')
        best_files = glob.glob(pattern)

        if len(best_files) <= max_keep:
            return

        # 수정 시간 기준 정렬
        best_files.sort(key=lambda x: os.path.getmtime(x))

        # 오래된 파일 삭제
        files_to_delete = best_files[:-max_keep]
        for f in files_to_delete:
            os.remove(f)
            self.logger.log_info(f"Removed old checkpoint: {os.path.basename(f)}")

    def train(self):
        """전체 학습 프로세스"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 2 TXT Linear Probing Training Started")
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

            # Scheduler step
            if self.scheduler:
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
    print("Stage 2 TXT Trainer Test")

    # 기본 설정
    config = {
        'model': {
            'clip_name': 'ViT-L/14',
            'feature_dim': 768,
            'num_subgroups': 8,
        },
        'training': {
            'num_epochs': 1,
            'train_batch_size': 8,
            'val_batch_size': 8,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 0.001,
        },
        'dataset': {
            'fairness_root': '/workspace/datasets/fairness',
            'train_dataset': ['ff++'],
            'validation_dataset': ['ff++'],
            'resolution': 256,
        },
        'logging': {
            'log_dir': '/tmp/stage2txt_test',
            'experiment_name': 'test',
        }
    }

    trainer = Stage2TxtTrainer(config)
    print("Trainer created successfully")
