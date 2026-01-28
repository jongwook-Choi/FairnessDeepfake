"""
Stage1 Text Trainer
CLIP + Text Encoder + Additive Adapter 학습을 위한 Trainer
"""

import os
import sys
import json
import datetime
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

sys.path.append('.')
from losses.combined_loss_txt import CombinedLossWithText
from utils.fairness_metrics import (
    compute_subgroup_feature_stats,
    compute_subgroup_cosine_similarity,
    compute_feature_variance_ratio,
    compute_subgroup_sinkhorn_distance,
    compute_classification_fairness_metrics,
    print_fairness_summary
)


class Recorder:
    """값 기록을 위한 유틸리티 클래스"""

    def __init__(self):
        self.values = []

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)

    def average(self):
        if len(self.values) == 0:
            return None
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values = []


class Stage1TxtTrainer:
    """
    Stage1 Text 학습 Trainer

    CLIP frozen + Text Encoder frozen + Additive Adapter를 학습하여
    Race/Gender classification + Fairness loss + Text Alignment loss를 통해
    subgroup bias를 제거
    """

    def __init__(self,
                 config,
                 model,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 log_dir=None):
        """
        Args:
            config (dict): 설정 딕셔너리
            model: FairnessAdapterWithText 모델
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러 (optional)
            device: 학습 device
            log_dir: 로그 디렉토리
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # 모델을 device로 이동
        self.model.to(device)

        # Text Anchors 초기화
        print("[Stage1TxtTrainer] Initializing text anchors...")
        self.model.initialize_text_anchors()

        # Loss 함수 초기화 (Text loss 포함)
        self.criterion = CombinedLossWithText(
            lambda_race=config.get('lambda_race', 1.0),
            lambda_gender=config.get('lambda_gender', 0.1),
            lambda_fairness=config.get('lambda_fairness', 0.5),
            lambda_pairwise_fairness=config.get('lambda_pairwise_fairness', 0.0),
            lambda_text_align=config.get('lambda_text_align', 0.1),
            lambda_text_consist=config.get('lambda_text_consist', 0.01),
            sinkhorn_blur=config.get('sinkhorn_blur', 1e-4),
            label_smoothing=config.get('label_smoothing', 0.0)
        )

        # 로그 디렉토리 설정
        self.timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if log_dir is None:
            log_dir = config.get('log_dir', './logs')
        self.log_dir = os.path.join(log_dir, f"stage1_txt_{self.timenow}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Config 저장 (학습 재현성을 위해)
        self._save_config(config)

        # TensorBoard writer
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tensorboard'))

        # Best metrics 추적
        self.best_metrics = {
            'race_acc': 0.0,
            'gender_acc': 0.0,
            'cosine_sim': 0.0,               # 높을수록 좋음
            'text_visual_sim': 0.0,          # 높을수록 좋음
            'variance_ratio': float('inf'),  # 낮을수록 좋음
            'combined_score': float('-inf'),
            'epoch': 0
        }

        # 종합 점수 가중치 (config에서 설정 가능)
        # score = w_race*race_acc + w_gender*gender_acc + w_cosine*cosine_sim
        #       + w_text_sim*text_visual_sim - w_var*variance_ratio
        self.score_weights = {
            'race_acc': config.get('weight_race_acc', 1.0),
            'gender_acc': config.get('weight_gender_acc', 0.1),
            'cosine_sim': config.get('weight_cosine_sim', 1.0),
            'text_visual_sim': config.get('weight_text_visual_sim', 0.5),
            'variance_ratio': config.get('weight_variance_ratio', 0.5),
        }

        # Gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.get('use_amp', False) else None

        # 실시간 loss 추적을 위한 deque (최근 100개 iteration)
        self.loss_history = {
            'overall': deque(maxlen=100),
            'race': deque(maxlen=100),
            'gender': deque(maxlen=100),
            'fairness': deque(maxlen=100),
            'pairwise_fairness': deque(maxlen=100),
            'text_align': deque(maxlen=100),
            'text_consist': deque(maxlen=100),
        }

        # CSV 로그 파일 초기화
        self.csv_log_path = os.path.join(self.log_dir, 'loss_log.csv')
        with open(self.csv_log_path, 'w') as f:
            f.write('step,epoch,loss_overall,loss_race,loss_gender,loss_fairness,'
                    'loss_text_align,loss_text_consist,race_acc,gender_acc,text_visual_sim\n')

        print(f"[Stage1TxtTrainer] Initialized")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Lambda race: {config.get('lambda_race', 1.0)}")
        print(f"  Lambda gender: {config.get('lambda_gender', 0.1)}")
        print(f"  Lambda fairness: {config.get('lambda_fairness', 0.5)}")
        print(f"  Lambda pairwise fairness: {config.get('lambda_pairwise_fairness', 0.0)}")
        print(f"  Lambda text_align: {config.get('lambda_text_align', 0.1)}")
        print(f"  Lambda text_consist: {config.get('lambda_text_consist', 0.01)}")

    def _save_config(self, config):
        """Config를 로그 디렉토리에 저장"""
        config_path = os.path.join(self.log_dir, 'config.json')

        # Tensor나 non-serializable 객체 처리
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_config = convert_to_serializable(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        print(f"  Config saved to: {config_path}")

    def train_epoch(self, epoch, train_loader, val_loader=None):
        """
        한 에포크 학습

        Args:
            epoch (int): 현재 에포크
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (optional)

        Returns:
            dict: 학습 메트릭
        """
        self.model.train()

        # 레코더 초기화
        loss_recorder = defaultdict(Recorder)
        metric_recorder = defaultdict(Recorder)

        # 학습 루프
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for iteration, data_dict in enumerate(pbar):
            # 데이터를 device로 이동
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_dict = self.model(data_dict)
                    loss_dict = self.criterion(pred_dict, data_dict)

                # Backward pass with gradient scaling
                self.scaler.scale(loss_dict['overall']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_dict = self.model(data_dict)
                loss_dict = self.criterion(pred_dict, data_dict)

                # Backward pass
                loss_dict['overall'].backward()
                self.optimizer.step()

            # 메트릭 계산
            metrics = self.criterion.get_metrics(pred_dict, data_dict)

            # 레코더 업데이트
            for key, value in loss_dict.items():
                if key not in ['subgroup_losses', 'text_align_info', 'text_consist_info'] and isinstance(value, torch.Tensor):
                    loss_recorder[key].update(value)

            metric_recorder['race_acc'].update(metrics['race_acc'])
            metric_recorder['gender_acc'].update(metrics['gender_acc'])
            if 'text_visual_sim' in metrics:
                metric_recorder['text_visual_sim'].update(metrics['text_visual_sim'])

            # 실시간 loss history 업데이트
            self.loss_history['overall'].append(loss_dict['overall'].item())
            self.loss_history['race'].append(loss_dict['race'].item())
            self.loss_history['gender'].append(loss_dict['gender'].item())
            self.loss_history['fairness'].append(loss_dict['fairness'].item())
            self.loss_history['pairwise_fairness'].append(loss_dict['pairwise_fairness'].item())
            self.loss_history['text_align'].append(loss_dict['text_align'].item())
            self.loss_history['text_consist'].append(loss_dict['text_consist'].item())

            # Moving average 계산
            ma_overall = sum(self.loss_history['overall']) / len(self.loss_history['overall'])
            ma_text_align = sum(self.loss_history['text_align']) / len(self.loss_history['text_align'])

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss_dict['overall'].item():.4f}",
                'ma_loss': f"{ma_overall:.4f}",
                'race': f"{metrics['race_acc']:.3f}",
                'gender': f"{metrics['gender_acc']:.3f}",
                'txt_sim': f"{metrics.get('text_visual_sim', 0):.3f}"
            })

            # Global step 계산
            global_step = epoch * len(train_loader) + iteration

            # CSV 로그 저장 (10 iteration마다)
            if iteration % 10 == 0:
                with open(self.csv_log_path, 'a') as f:
                    f.write(f"{global_step},{epoch},{loss_dict['overall'].item():.6f},"
                            f"{loss_dict['race'].item():.6f},{loss_dict['gender'].item():.6f},"
                            f"{loss_dict['fairness'].item():.6f},"
                            f"{loss_dict['text_align'].item():.6f},{loss_dict['text_consist'].item():.6f},"
                            f"{metrics['race_acc']:.4f},{metrics['gender_acc']:.4f},"
                            f"{metrics.get('text_visual_sim', 0):.4f}\n")

            # TensorBoard 로깅 (100 iteration마다)
            if iteration % 100 == 0:
                self.writer.add_scalar('train/loss_overall', loss_dict['overall'].item(), global_step)
                self.writer.add_scalar('train/loss_race', loss_dict['race'].item(), global_step)
                self.writer.add_scalar('train/loss_gender', loss_dict['gender'].item(), global_step)
                self.writer.add_scalar('train/loss_fairness', loss_dict['fairness'].item(), global_step)
                self.writer.add_scalar('train/loss_pairwise_fairness', loss_dict['pairwise_fairness'].item(), global_step)
                self.writer.add_scalar('train/loss_text_align', loss_dict['text_align'].item(), global_step)
                self.writer.add_scalar('train/loss_text_consist', loss_dict['text_consist'].item(), global_step)
                self.writer.add_scalar('train/race_acc', metrics['race_acc'], global_step)
                self.writer.add_scalar('train/gender_acc', metrics['gender_acc'], global_step)
                if 'text_visual_sim' in metrics:
                    self.writer.add_scalar('train/text_visual_sim', metrics['text_visual_sim'], global_step)

        # 에포크 평균 계산
        epoch_metrics = {
            'loss_overall': loss_recorder['overall'].average(),
            'loss_race': loss_recorder['race'].average(),
            'loss_gender': loss_recorder['gender'].average(),
            'loss_fairness': loss_recorder['fairness'].average(),
            'loss_pairwise_fairness': loss_recorder['pairwise_fairness'].average(),
            'loss_text_align': loss_recorder['text_align'].average(),
            'loss_text_consist': loss_recorder['text_consist'].average(),
            'race_acc': metric_recorder['race_acc'].average(),
            'gender_acc': metric_recorder['gender_acc'].average(),
            'text_visual_sim': metric_recorder['text_visual_sim'].average() if metric_recorder['text_visual_sim'].values else 0,
        }

        # 스케줄러 스텝
        if self.scheduler is not None:
            self.scheduler.step()
            epoch_metrics['lr'] = self.scheduler.get_last_lr()[0]

        # 검증 수행
        if val_loader is not None:
            val_metrics = self.validate(epoch, val_loader)
            epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            # 종합 점수 계산 (CLIP_stage1과 동일한 방식 + text_visual_sim)
            # score = w_race*race_acc + w_gender*gender_acc + w_cosine*cosine_sim
            #       + w_text_sim*text_visual_sim - w_var*variance_ratio
            combined_score = (
                self.score_weights['race_acc'] * val_metrics['race_acc'] +
                self.score_weights['gender_acc'] * val_metrics['gender_acc'] +
                self.score_weights['cosine_sim'] * val_metrics['mean_cosine_sim'] +
                self.score_weights['text_visual_sim'] * val_metrics.get('text_visual_sim', 0) -
                self.score_weights['variance_ratio'] * val_metrics['variance_ratio']
            )

            # Best 모델 저장
            if combined_score > self.best_metrics['combined_score']:
                self.best_metrics.update({
                    'race_acc': val_metrics['race_acc'],
                    'gender_acc': val_metrics['gender_acc'],
                    'cosine_sim': val_metrics['mean_cosine_sim'],
                    'text_visual_sim': val_metrics.get('text_visual_sim', 0),
                    'variance_ratio': val_metrics['variance_ratio'],
                    'combined_score': combined_score,
                    'epoch': epoch
                })
                self.save_checkpoint(epoch, is_best=True)
                print(f"\n[Best Model] Epoch {epoch} | Score: {combined_score:.4f} "
                      f"(race: {val_metrics['race_acc']:.4f}, gender: {val_metrics['gender_acc']:.4f}, "
                      f"cosine: {val_metrics['mean_cosine_sim']:.4f}, "
                      f"txt_sim: {val_metrics.get('text_visual_sim', 0):.4f}, "
                      f"var: {val_metrics['variance_ratio']:.4f})")

        # TensorBoard에 에포크 메트릭 로깅
        for key, value in epoch_metrics.items():
            if value is not None:
                self.writer.add_scalar(f'epoch/{key}', value, epoch)

        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch, val_loader):
        """
        검증 수행

        Args:
            epoch (int): 현재 에포크
            val_loader: 검증 데이터 로더

        Returns:
            dict: 검증 메트릭
        """
        self.model.eval()

        # 모든 예측과 레이블 수집
        all_features = []
        all_race_logits = []
        all_gender_logits = []
        all_races = []
        all_genders = []
        all_subgroups = []

        loss_recorder = defaultdict(Recorder)
        text_visual_sims = []

        for data_dict in tqdm(val_loader, desc="Validating"):
            # 데이터를 device로 이동
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            # Forward pass
            pred_dict = self.model(data_dict)
            loss_dict = self.criterion(pred_dict, data_dict)

            # 결과 수집
            all_features.append(pred_dict['final_features'].cpu())
            all_race_logits.append(pred_dict['race_logits'].cpu())
            all_gender_logits.append(pred_dict['gender_logits'].cpu())
            all_races.append(data_dict['race'].cpu())
            all_genders.append(data_dict['gender'].cpu())
            all_subgroups.append(data_dict['subgroup'].cpu())

            # Text-Visual similarity
            if 'text_align_info' in loss_dict and 'mean_cosine_sim' in loss_dict['text_align_info']:
                text_visual_sims.append(loss_dict['text_align_info']['mean_cosine_sim'])

            for key, value in loss_dict.items():
                if key not in ['subgroup_losses', 'text_align_info', 'text_consist_info'] and isinstance(value, torch.Tensor):
                    loss_recorder[key].update(value)

        # 텐서 연결
        all_features = torch.cat(all_features, dim=0)
        all_race_logits = torch.cat(all_race_logits, dim=0)
        all_gender_logits = torch.cat(all_gender_logits, dim=0)
        all_races = torch.cat(all_races, dim=0)
        all_genders = torch.cat(all_genders, dim=0)
        all_subgroups = torch.cat(all_subgroups, dim=0)

        # Classification metrics
        race_preds = torch.argmax(all_race_logits, dim=1)
        gender_preds = torch.argmax(all_gender_logits, dim=1)
        race_acc = (race_preds == all_races).float().mean().item()
        gender_acc = (gender_preds == all_genders).float().mean().item()

        # Fairness metrics
        stats_dict = compute_subgroup_feature_stats(all_features, all_subgroups)
        sim_matrix, sim_stats = compute_subgroup_cosine_similarity(all_features, all_subgroups)
        var_stats = compute_feature_variance_ratio(all_features, all_subgroups)

        # Sinkhorn distance 기반 fairness metric
        sinkhorn_stats = compute_subgroup_sinkhorn_distance(
            all_features, all_subgroups,
            sinkhorn_blur=self.config.get('sinkhorn_blur', 1e-4)
        )

        # Race/Gender classification fairness
        race_fairness = compute_classification_fairness_metrics(all_race_logits, all_races, all_subgroups)
        gender_fairness = compute_classification_fairness_metrics(all_gender_logits, all_genders, all_subgroups)

        # 결과 출력
        print_fairness_summary(stats_dict, sim_stats, var_stats, race_fairness, gender_fairness)

        # Sinkhorn distance 출력
        if sinkhorn_stats['geomloss_available']:
            print(f"\n[Sinkhorn Distance (lower is better)]")
            print(f"  Mean distance: {sinkhorn_stats['mean_distance']:.6f}")
            print(f"  Within variance: {sinkhorn_stats['within_variance']:.6f}")
            for sg_id, sg_info in sinkhorn_stats['subgroup_distances'].items():
                print(f"    {sg_info['name']}: {sg_info['distance']:.6f} (n={sg_info['count']})")
        else:
            print(f"\n[Sinkhorn Distance] geomloss not installed, using variance_ratio as fallback")

        # Text-Visual Similarity 출력
        mean_text_visual_sim = sum(text_visual_sims) / len(text_visual_sims) if text_visual_sims else 0
        print(f"\n[Text-Visual Similarity]")
        print(f"  Mean similarity: {mean_text_visual_sim:.4f}")

        metrics = {
            'loss_overall': loss_recorder['overall'].average(),
            'loss_race': loss_recorder['race'].average(),
            'loss_gender': loss_recorder['gender'].average(),
            'loss_fairness': loss_recorder['fairness'].average(),
            'loss_pairwise_fairness': loss_recorder['pairwise_fairness'].average(),
            'loss_text_align': loss_recorder['text_align'].average(),
            'loss_text_consist': loss_recorder['text_consist'].average(),
            'race_acc': race_acc,
            'gender_acc': gender_acc,
            'mean_cosine_sim': sim_stats['mean_similarity'],
            'variance_ratio': var_stats['variance_ratio'],
            'sinkhorn_distance': sinkhorn_stats['mean_distance'],
            'sinkhorn_within_var': sinkhorn_stats['within_variance'],
            'race_acc_gap': race_fairness['accuracy_gap'],
            'gender_acc_gap': gender_fairness['accuracy_gap'],
            'text_visual_sim': mean_text_visual_sim,
        }

        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 최신 체크포인트 저장
        save_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, save_path)

        # Best 모델 저장
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

            # 점수가 포함된 파일명
            race_acc = self.best_metrics['race_acc']
            gender_acc = self.best_metrics['gender_acc']
            var = self.best_metrics['variance_ratio']
            score = self.best_metrics['combined_score']
            score_filename = f"best_ep{epoch}_score{score:.4f}_race{race_acc:.4f}_gender{gender_acc:.4f}_var{var:.6f}.pth"
            score_path = os.path.join(self.log_dir, score_filename)
            torch.save(checkpoint, score_path)

            # 최대 3개만 유지
            self._cleanup_old_best_checkpoints(max_keep=3)

            print(f"\n[Checkpoint] Best model saved: {score_filename}")

        # 에포크별 저장 (선택적)
        if self.config.get('save_every_epoch', False):
            epoch_path = os.path.join(self.log_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def _cleanup_old_best_checkpoints(self, max_keep=3):
        """오래된 best 체크포인트 삭제"""
        import glob

        pattern = os.path.join(self.log_dir, 'best_ep*.pth')
        best_files = glob.glob(pattern)

        if len(best_files) <= max_keep:
            return

        best_files.sort(key=os.path.getmtime)
        files_to_delete = best_files[:-max_keep]
        for f in files_to_delete:
            os.remove(f)
            print(f"[Checkpoint] Removed old checkpoint: {os.path.basename(f)}")

    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)

        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best metrics: {self.best_metrics}")

        return checkpoint['epoch']

    def train(self, train_loader, val_loader=None, num_epochs=None):
        """
        전체 학습 수행

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (optional)
            num_epochs: 학습 에포크 수

        Returns:
            dict: 최종 메트릭
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 10)

        print(f"\n[Training] Starting training for {num_epochs} epochs")
        print(f"  Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Val samples: {len(val_loader.dataset)}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*60)

            epoch_metrics = self.train_epoch(epoch, train_loader, val_loader)

            # 메트릭 출력
            print(f"\n[Epoch {epoch} Summary]")
            print(f"  Train Loss: {epoch_metrics['loss_overall']:.4f}")
            print(f"  Race Acc: {epoch_metrics['race_acc']:.4f}")
            print(f"  Gender Acc: {epoch_metrics['gender_acc']:.4f}")
            print(f"  Text-Visual Sim: {epoch_metrics['text_visual_sim']:.4f}")

            if val_loader:
                print(f"  Val Loss: {epoch_metrics.get('val_loss_overall', 'N/A')}")
                print(f"  Val Race Acc: {epoch_metrics.get('val_race_acc', 'N/A')}")
                print(f"  Val Cosine Sim: {epoch_metrics.get('val_mean_cosine_sim', 'N/A')}")
                print(f"  Val Text-Visual Sim: {epoch_metrics.get('val_text_visual_sim', 'N/A')}")
                print(f"  Val Variance Ratio: {epoch_metrics.get('val_variance_ratio', 'N/A')}")

            # 체크포인트 저장
            self.save_checkpoint(epoch)

        # 학습 완료
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best model at epoch {self.best_metrics['epoch']}")
        print(f"  Combined Score: {self.best_metrics['combined_score']:.4f}")
        print(f"  Race Acc: {self.best_metrics['race_acc']:.4f}")
        print(f"  Gender Acc: {self.best_metrics['gender_acc']:.4f}")
        print(f"  Cosine Sim: {self.best_metrics['cosine_sim']:.4f}")
        print(f"  Text-Visual Sim: {self.best_metrics['text_visual_sim']:.4f}")
        print(f"  Variance Ratio: {self.best_metrics['variance_ratio']:.6f}")
        print('='*60)

        self.writer.close()

        return self.best_metrics
