"""
Stage 2 Trainer v2
AdaIN + Demographic-Aware Cross-Attention + CVaR/LDAM/Sinkhorn Fairness

핵심 변경점:
1. Fairness warmup: 초기 N epoch는 fairness loss=0, 이후 선형 증가
2. Config 기반 fairness strategy 선택 (CVaR, LDAM, Sinkhorn 조합)
3. Stage2ModelV2 사용 (AdaIN + DemographicCA + Gate)
4. Best model: AUC - w_fpr*F_FPR - w_meo*F_MEO
"""

import os
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from model.stage2_model_v2 import Stage2ModelV2
from dataset.fairness_dataset import FairnessDataset, SubgroupBalancedBatchSampler
from losses.detection_loss import DetectionLoss
from losses.stage2_fairness_loss import Stage2FairnessLoss
from losses.subgroup_conditional_loss import SubgroupConditionalDetectionLoss
from losses.contrastive_loss import FeatureDisentanglementLoss
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
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr_scale = max(self.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class Stage2TrainerV2:
    """
    Stage 2 Trainer v2

    Features:
    - Stage2ModelV2: AdaIN + DemographicCA + Gate
    - Config-driven fairness strategy (CVaR, LDAM, Sinkhorn)
    - Fairness warmup scheduling
    - Disentanglement loss for Stage1/Stage2 features
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
        torch.set_default_dtype(torch.float32)

        # Seed 설정
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device
        self.device = torch.device(
            config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Log directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_config = config.get('logging', {})
        base_log_dir = log_config.get('log_dir', '/workspace/code/CLIP_stage2/logs')
        experiment_name = log_config.get('experiment_name', 'stage2_v2')
        self.log_dir = os.path.join(base_log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Logger
        self.logger = TrainingLogger(self.log_dir, experiment_name)

        # Components (setup에서 초기화)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.cls_loss_fn = None
        self.cvar_loss_fn = None
        self.sinkhorn_loss_fn = None
        self.contrastive_loss_fn = None
        self.train_dataset = None

        # Fairness strategy config
        self.fairness_config = config.get('loss', {}).get('fairness_strategy', {})
        self.use_cvar = self.fairness_config.get('use_cvar', True)
        self.use_ldam = self.fairness_config.get('use_ldam', True)
        self.use_sinkhorn = self.fairness_config.get('use_sinkhorn', True)

        # Fairness warmup config
        warmup_config = self.fairness_config.get('warmup', {})
        self.fairness_warmup_epochs = warmup_config.get('warmup_epochs', 5)
        self.fairness_warmup_end = warmup_config.get('warmup_end_epoch', 10)

        # Best model tracking
        best_config = config.get('best_model_selection', {})
        self.best_score = float('-inf')
        self.best_epoch = 0
        self.best_metrics = {}
        self.score_weights = {
            'w_auc': best_config.get('w_auc', 1.0),
            'w_f_fpr': best_config.get('w_f_fpr', -0.001),
            'w_f_meo': best_config.get('w_f_meo', -0.001),
        }

    def _get_fairness_weight(self, epoch):
        """
        Fairness loss warmup scheduling

        epoch < warmup_epochs: 0
        warmup_epochs <= epoch < warmup_end: 선형 증가 (0 -> 1)
        epoch >= warmup_end: 1.0
        """
        if epoch < self.fairness_warmup_epochs:
            return 0.0
        elif epoch < self.fairness_warmup_end:
            progress = (epoch - self.fairness_warmup_epochs) / max(
                1, self.fairness_warmup_end - self.fairness_warmup_epochs
            )
            return progress
        else:
            return 1.0

    def create_model(self):
        """모델 생성"""
        model_config = self.config.get('model', {})
        loss_config = self.config.get('loss', {})
        ldam_config = self.fairness_config.get('ldam', {})

        self.model = Stage2ModelV2(
            clip_name=model_config.get('clip_name', 'ViT-L/14'),
            feature_dim=model_config.get('feature_dim', 768),
            stage1_hidden_dim=model_config.get('stage1_hidden_dim', 512),
            stage2_hidden_dim=model_config.get('stage2_hidden_dim', 384),
            classifier_hidden_dims=model_config.get('classifier_hidden_dims', [384, 192]),
            num_classes=model_config.get('num_classes', 2),
            num_ca_views=model_config.get('num_ca_views', 4),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            gate_init_bias=model_config.get('gate_init_bias', 0.0),
            normalize_features=model_config.get('normalize_features', True),
            use_ldam_head=self.use_ldam,
            ldam_cls_num_list=ldam_config.get('cls_num_list'),
            ldam_max_margin=ldam_config.get('max_margin', 0.5),
            ldam_s=ldam_config.get('ldam_s', 30.0),
            device=str(self.device),
            clip_download_root=self.config.get('clip_download_root', '/data/cuixinjie/weights')
        )

        # Stage 1 checkpoint 로드
        stage1_checkpoint = self.config.get('stage1_checkpoint')
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            self.model.load_stage1_checkpoint(stage1_checkpoint)
        else:
            self.logger.log_warning(f"Stage 1 checkpoint not found: {stage1_checkpoint}")

        self.model.to(self.device)
        self.model = self.model.float()
        self.model.print_trainable_parameters()

        return self.model

    def create_loss_functions(self):
        """Loss functions 생성 (config 기반)"""
        loss_config = self.config.get('loss', {})
        detection_config = loss_config.get('detection', {})

        # 1. Detection Loss (기본)
        self.cls_loss_fn = DetectionLoss(
            loss_type=loss_config.get('type', 'cross_entropy'),
            label_smoothing=detection_config.get('label_smoothing', 0.1)
        )

        # 2. CVaR Loss (optional)
        if self.use_cvar:
            cvar_config = self.fairness_config.get('cvar', {})
            self.cvar_loss_fn = SubgroupConditionalDetectionLoss(
                num_subgroups=8,
                inner_alpha=cvar_config.get('inner_alpha', 0.9),
                outer_alpha=cvar_config.get('outer_alpha', 0.5),
                label_smoothing=detection_config.get('label_smoothing', 0.1)
            )
            print("[Loss] CVaR (Subgroup-Conditional) enabled")

        # 3. Sinkhorn Loss (optional)
        if self.use_sinkhorn:
            sinkhorn_config = self.fairness_config.get('sinkhorn', {})
            self.sinkhorn_loss_fn = Stage2FairnessLoss(
                sinkhorn_blur=sinkhorn_config.get('sinkhorn_blur', 1e-4),
                num_subgroups=8
            )
            print("[Loss] Sinkhorn (per-class) enabled")

        # 4. Contrastive Loss (disentanglement)
        contrastive_config = loss_config.get('contrastive', {})
        if contrastive_config.get('lambda_contrastive', 0.05) > 0:
            self.contrastive_loss_fn = FeatureDisentanglementLoss(
                margin=contrastive_config.get('margin', 3.0),
                loss_type='cosine'
            )
            print("[Loss] Contrastive (disentanglement) enabled")

    def create_optimizer(self):
        """Optimizer 생성"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()
        params = list(self.model.get_trainable_params())

        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=opt_config.get('lr', 3e-4),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=opt_config.get('lr', 3e-4),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        print(f"[Optimizer] {opt_type}, lr={opt_config.get('lr', 3e-4)}")
        return self.optimizer

    def create_scheduler(self):
        """Scheduler 생성"""
        sched_config = self.config.get('scheduler', {})
        if not sched_config.get('use_scheduler', True):
            return None

        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 50)
        sched_name = sched_config.get('name', 'WarmupCosine')

        if sched_name == 'WarmupCosine':
            warmup_epochs = sched_config.get('warmup_epochs', 5)
            min_lr = sched_config.get('eta_min', 1e-6)
            base_lr = self.config.get('optimizer', {}).get('lr', 3e-4)
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs,
                min_lr_ratio=min_lr / base_lr
            )
        elif sched_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None

        return self.scheduler

    def create_dataloaders(self):
        """DataLoader 생성"""
        dataset_config = self.config.get('dataset', {})
        training_config = self.config.get('training', {})
        merged_config = {**self.config, **dataset_config}

        # Normalization: CLIP 표준 (Stage 1과 일치)
        if 'mean' in self.config:
            merged_config['mean'] = self.config['mean']
        if 'std' in self.config:
            merged_config['std'] = self.config['std']

        # Train dataset
        self.train_dataset = FairnessDataset(config=merged_config, mode='train')

        # Val dataset
        val_config = merged_config.copy()
        val_dataset = FairnessDataset(config=val_config, mode='validation')

        train_batch_size = training_config.get('train_batch_size', 256)
        val_batch_size = training_config.get('val_batch_size', 64)
        num_workers = training_config.get('num_workers', 4)

        # Subgroup balanced sampler
        use_sampler = dataset_config.get('use_subgroup_sampler', True)
        if use_sampler:
            train_sampler = SubgroupBalancedBatchSampler(
                self.train_dataset.get_subgroup_list(),
                batch_size=train_batch_size,
                drop_last=True
            )
            train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=FairnessDataset.collate_fn,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=FairnessDataset.collate_fn,
                pin_memory=True,
                drop_last=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=FairnessDataset.collate_fn,
            pin_memory=True
        )

        return train_loader, val_loader

    def compute_losses(self, pred_dict, data_dict, epoch):
        """
        모든 loss 계산 (config 기반)

        Args:
            pred_dict (dict): 모델 예측
            data_dict (dict): 입력 데이터
            epoch (int): 현재 epoch

        Returns:
            total_loss: 총 loss
            loss_dict: 각 loss 값 딕셔너리
        """
        loss_config = self.config.get('loss', {})
        loss_dict = {}

        # Fairness warmup multiplier
        fair_weight = self._get_fairness_weight(epoch)

        # 1. Detection loss (기본)
        cls_logits = pred_dict['cls']
        labels = data_dict['label']
        cls_loss = self.cls_loss_fn(cls_logits, labels)
        loss_dict['cls'] = cls_loss.item()
        total_loss = cls_loss

        # 2. CVaR loss (optional)
        if self.use_cvar and self.cvar_loss_fn is not None and fair_weight > 0:
            subgroups = data_dict['subgroup']
            cvar_loss, cvar_info = self.cvar_loss_fn(cls_logits, labels, subgroups)
            # CVaR는 detection loss를 대체하므로 별도 가중치 없이 사용
            # 기본 CE와 CVaR를 혼합: (1-fair_weight)*CE + fair_weight*CVaR
            total_loss = (1 - fair_weight) * cls_loss + fair_weight * cvar_loss
            loss_dict['cvar'] = cvar_loss.item()

        # 3. LDAM loss (optional)
        if self.use_ldam and self.model.ldam_head is not None and fair_weight > 0:
            ldam_config = self.fairness_config.get('ldam', {})
            lambda_ldam = ldam_config.get('lambda_ldam', 0.1)
            subgroups = data_dict['subgroup']
            ldam_loss = self.model.ldam_head.compute_loss(
                pred_dict['fused_features_norm'], subgroups
            )
            total_loss = total_loss + fair_weight * lambda_ldam * ldam_loss
            loss_dict['ldam'] = ldam_loss.item()

        # 4. Sinkhorn loss (optional)
        if self.use_sinkhorn and self.sinkhorn_loss_fn is not None and fair_weight > 0:
            sinkhorn_config = self.fairness_config.get('sinkhorn', {})
            lambda_sinkhorn = sinkhorn_config.get('lambda_sinkhorn', 0.1)
            features = pred_dict['fused_features_norm']
            labels_for_sink = data_dict['label']
            subgroups = data_dict['subgroup']
            sink_loss, sink_info = self.sinkhorn_loss_fn(features, labels_for_sink, subgroups)
            total_loss = total_loss + fair_weight * lambda_sinkhorn * sink_loss
            loss_dict['sinkhorn'] = sink_loss.item()

        # 5. Contrastive loss (disentanglement)
        contrastive_config = loss_config.get('contrastive', {})
        lambda_contrastive = contrastive_config.get('lambda_contrastive', 0.05)
        if self.contrastive_loss_fn is not None and lambda_contrastive > 0:
            stage1_feat = pred_dict['stage1_features']
            stage2_feat = pred_dict['stage2_features']
            contra_loss, contra_info = self.contrastive_loss_fn(stage1_feat, stage2_feat)
            total_loss = total_loss + lambda_contrastive * contra_loss
            loss_dict['contrastive'] = contra_loss.item()

        loss_dict['total'] = total_loss.item()
        loss_dict['fair_weight'] = fair_weight

        # Gate 모니터링
        if 'gate' in pred_dict and pred_dict['gate'] is not None:
            loss_dict['gate_mean'] = pred_dict['gate'].mean().item()
            loss_dict['gate_std'] = pred_dict['gate'].std().item()

        return total_loss, loss_dict

    def train_epoch(self, epoch, train_loader):
        """한 epoch 학습"""
        self.model.train()

        epoch_losses = {}
        num_correct = 0
        num_total = 0
        start_time = time.time()

        gradient_clip = self.config.get('gradient_clip_max_norm',
                                        self.config.get('gradient_clip', 1.0))

        for iteration, data_dict in enumerate(train_loader):
            # Device로 이동
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            pred_dict = self.model(data_dict)

            # Loss 계산
            total_loss, loss_dict = self.compute_losses(pred_dict, data_dict, epoch)

            # Backward
            total_loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip
                )

            self.optimizer.step()

            # Accuracy
            preds = pred_dict['cls'].argmax(dim=1)
            num_correct += (preds == data_dict['label']).sum().item()
            num_total += len(data_dict['label'])

            # Loss 누적
            for key, val in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(val)

            # 로깅
            print_freq = self.config.get('logging', {}).get('print_freq', 50)
            if iteration % print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                loss_str = ' | '.join([f"{k}:{v:.4f}" for k, v in loss_dict.items()
                                       if k not in ['fair_weight']])
                print(f"  [{iteration}/{len(train_loader)}] lr={lr:.2e} | {loss_str} | "
                      f"fair_w={loss_dict.get('fair_weight', 0):.2f}")

        # Epoch 평균
        epoch_time = time.time() - start_time
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        acc = num_correct / max(num_total, 1)

        print(f"\n[Train Epoch {epoch}] time={epoch_time:.1f}s | "
              f"loss={avg_losses['total']:.4f} | acc={acc:.4f} | "
              f"fair_w={avg_losses.get('fair_weight', 0):.2f}")

        # TensorBoard
        self.logger.log_epoch_train(epoch, avg_losses, acc)
        if 'gate_mean' in avg_losses:
            self.logger.writer.add_scalar('monitor/gate_mean', avg_losses['gate_mean'], epoch)
            self.logger.writer.add_scalar('monitor/gate_std', avg_losses.get('gate_std', 0), epoch)

        return avg_losses, acc

    @torch.no_grad()
    def validate(self, epoch, val_loader):
        """검증 수행"""
        self.model.eval()

        epoch_losses = {}
        num_correct = 0
        num_total = 0

        for data_dict in val_loader:
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            pred_dict = self.model(data_dict, inference=True)

            # Loss
            cls_logits = pred_dict['cls']
            labels = data_dict['label']
            cls_loss = self.cls_loss_fn(cls_logits, labels)

            if 'cls' not in epoch_losses:
                epoch_losses['cls'] = []
            epoch_losses['cls'].append(cls_loss.item())

            # Accuracy
            preds = cls_logits.argmax(dim=1)
            num_correct += (preds == labels).sum().item()
            num_total += len(labels)

        # Metrics
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        acc = num_correct / max(num_total, 1)

        # AUC & fairness
        test_metrics = self.model.get_test_metrics()
        fairness_metrics = self.model.get_fairness_metrics()

        auc = test_metrics.get('auc', 0.5)
        eer = test_metrics.get('eer', 0.5)

        print(f"\n[Val Epoch {epoch}] loss={avg_losses.get('cls', 0):.4f} | "
              f"acc={acc:.4f} | auc={auc:.4f} | eer={eer:.4f}")

        if fairness_metrics:
            print(f"  F_FPR={fairness_metrics['F_FPR']:.4f} | "
                  f"F_OAE={fairness_metrics['F_OAE']:.4f} | "
                  f"F_DP={fairness_metrics['F_DP']:.4f} | "
                  f"F_MEO={fairness_metrics['F_MEO']:.4f}")

        # Best model score
        f_fpr = fairness_metrics['F_FPR'] if fairness_metrics else 0
        f_meo = fairness_metrics['F_MEO'] if fairness_metrics else 0
        score = (
            self.score_weights['w_auc'] * auc +
            self.score_weights['w_f_fpr'] * f_fpr +
            self.score_weights['w_f_meo'] * f_meo
        )

        # Best model 저장
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metrics = {
                'auc': auc, 'acc': acc, 'eer': eer,
                'F_FPR': f_fpr, 'F_MEO': f_meo,
                'score': score,
            }
            self._save_checkpoint(epoch, is_best=True)
            print(f"  ** New Best! score={score:.6f} (auc={auc:.4f})")

        # Logging
        self.logger.log_epoch_val(epoch, avg_losses, acc, auc, eer)
        if fairness_metrics:
            for k, v in fairness_metrics.items():
                if isinstance(v, (int, float)):
                    self.logger.writer.add_scalar(f'val_fairness/{k}', v, epoch)

        return test_metrics, fairness_metrics

    def _save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            try:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            except AttributeError:
                pass

        # Latest
        torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoint_latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoint_best.pth'))
            auc = self.best_metrics.get('auc', 0)
            f_fpr = self.best_metrics.get('F_FPR', 0)
            filename = f"best_ep{epoch}_auc{auc:.4f}_fpr{f_fpr:.2f}.pth"
            torch.save(checkpoint, os.path.join(self.log_dir, filename))

    def test(self, test_loader, dataset_name='test'):
        """테스트 수행"""
        self.model.eval()

        with torch.no_grad():
            for data_dict in test_loader:
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)
                self.model(data_dict, inference=True)

        test_metrics = self.model.get_test_metrics()
        fairness_metrics = self.model.get_fairness_metrics()

        print(f"\n[Test: {dataset_name}] AUC={test_metrics['auc']:.4f} | "
              f"ACC={test_metrics['acc']:.4f} | EER={test_metrics['eer']:.4f}")

        if fairness_metrics:
            print(f"  F_FPR={fairness_metrics['F_FPR']:.4f} | "
                  f"F_MEO={fairness_metrics['F_MEO']:.4f}")

        return test_metrics, fairness_metrics

    def train(self, num_epochs=None):
        """전체 학습"""
        training_config = self.config.get('training', {})
        if num_epochs is None:
            num_epochs = training_config.get('num_epochs', 50)

        # Setup
        self.create_model()
        self.create_loss_functions()
        self.create_optimizer()
        self.create_scheduler()
        train_loader, val_loader = self.create_dataloaders()

        print(f"\n{'='*60}")
        print(f"Stage 2 Training v2 (AdaIN + DemographicCA)")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Fairness strategy: CVaR={self.use_cvar}, "
              f"LDAM={self.use_ldam}, Sinkhorn={self.use_sinkhorn}")
        print(f"Fairness warmup: {self.fairness_warmup_epochs}-{self.fairness_warmup_end} epochs")
        print(f"Best model: {self.score_weights}")

        eval_freq = self.config.get('evaluation', {}).get('eval_freq', 1)
        test_every = self.config.get('evaluation', {}).get('test_every_n_epochs', 5)

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*60)

            # Train
            train_losses, train_acc = self.train_epoch(epoch, train_loader)

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, WarmupCosineScheduler):
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()

            # Validate
            if epoch % eval_freq == 0:
                self.validate(epoch, val_loader)

            # Periodic checkpoint
            self._save_checkpoint(epoch)

        # 학습 완료
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best metrics: {self.best_metrics}")
        print('='*60)

        self.logger.close()

        return self.best_metrics
