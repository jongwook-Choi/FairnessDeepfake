#!/usr/bin/env python3
"""
Training Logger Module for CLIP_Merging
훈련 과정 및 결과를 종합적으로 로깅하는 클래스
"""

import os
import json
import logging
import datetime
import time
import csv
from typing import Dict, Any, Optional, List


class TrainingLogger:
    """훈련 과정 및 결과를 로깅하는 클래스"""

    def __init__(self, log_dir: str, experiment_name: str = "training"):
        """
        Args:
            log_dir: 로그 파일들을 저장할 디렉토리
            experiment_name: 실험 이름 (로그 파일명에 사용)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = datetime.datetime.now()

        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)

        # 로그 파일 경로 설정
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_files = {
            'training_log': os.path.join(log_dir, f'{experiment_name}_training_{timestamp}.log'),
            'config_log': os.path.join(log_dir, f'{experiment_name}_config_{timestamp}.json'),
            'metrics_csv': os.path.join(log_dir, f'{experiment_name}_metrics_{timestamp}.csv'),
            'summary_log': os.path.join(log_dir, f'{experiment_name}_summary_{timestamp}.txt'),
            'error_log': os.path.join(log_dir, f'{experiment_name}_errors_{timestamp}.log')
        }

        # 파이썬 로거 설정
        self.logger = logging.getLogger(f'{experiment_name}_logger')
        self.logger.setLevel(logging.INFO)

        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 파일 핸들러 추가
        file_handler = logging.FileHandler(self.log_files['training_log'], encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 에러 파일 핸들러 추가
        error_handler = logging.FileHandler(self.log_files['error_log'], encoding='utf-8')
        error_handler.setLevel(logging.ERROR)

        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)

        # 메트릭 저장을 위한 리스트
        self.epoch_metrics = []

        self.logger.info(f"Training logger initialized for {experiment_name}")
        self.logger.info(f"Log directory: {log_dir}")

    def log_config(self, config: Dict[str, Any], model_info: Optional[Dict] = None,
                   optimizer_info: Optional[Dict] = None):
        """설정 정보를 JSON 파일로 저장"""
        config_data = {
            'experiment_info': {
                'name': self.experiment_name,
                'start_time': self.start_time.isoformat(),
                'log_directory': self.log_dir,
                'log_files': self.log_files
            },
            'training_config': config,
            'model_info': model_info or {},
            'optimizer_info': optimizer_info or {}
        }

        with open(self.log_files['config_log'], 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Configuration saved to: {self.log_files['config_log']}")

    def log_model_info(self, model, total_params: int = None, model_size_mb: float = None):
        """모델 정보 로그"""
        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("Model Information:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")

        if model_size_mb:
            self.logger.info(f"  Model size: {model_size_mb:.2f} MB")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'model_size_mb': model_size_mb
        }

    def log_layer_info(self, layer_info: Dict[str, Any]):
        """Layer-wise 정보 로그"""
        self.logger.info("Layer-wise Learning Rate Configuration:")
        for layer_name, info in layer_info.items():
            lr = info.get('lr', 'N/A')
            param_count = len(info.get('params', []))
            layer_id = info.get('layer_id', 'N/A')

            self.logger.info(f"  {layer_name} (ID: {layer_id}): LR={lr:.2e}, Params={param_count}")

            # 샘플 파라미터 이름 출력
            if 'param_names' in info and info['param_names']:
                sample_names = ', '.join(info['param_names'][:3])
                self.logger.info(f"    Sample params: {sample_names}")

    def log_training_start(self, total_epochs: int, total_batches_per_epoch: int = None):
        """훈련 시작 로그"""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Total epochs: {total_epochs}")
        if total_batches_per_epoch:
            self.logger.info(f"Batches per epoch: {total_batches_per_epoch}")
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_epoch_start(self, epoch: int, total_epochs: int, learning_rates: List[float] = None):
        """에포크 시작 로그"""
        self.logger.info(f"Starting Epoch {epoch}/{total_epochs}")

        if learning_rates:
            lr_info = f"Learning rates - Head: {learning_rates[0]:.2e}"
            if len(learning_rates) > 1:
                lr_info += f", Range: {min(learning_rates):.2e} - {max(learning_rates):.2e}"
            self.logger.info(lr_info)

        self.logger.info("-" * 60)

    def log_batch_progress(self, epoch: int, batch_idx: int, total_batches: int,
                          loss: float, accuracy: float, grad_norm: float = None,
                          max_grad: float = None, current_lrs: List[float] = None,
                          skipped_batches: int = 0, log_interval: int = 50):
        """배치 진행 상황 로그"""
        if batch_idx % log_interval == 0:
            msg = (f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, "
                   f"Loss: {loss:.6f}, Acc: {accuracy:.2f}%")

            if grad_norm is not None and max_grad is not None:
                msg += f", Grad norm: {grad_norm:.6f}, Max grad: {max_grad:.6f}"

            if current_lrs is not None and len(current_lrs) > 0:
                msg += f", Head LR: {current_lrs[0]:.2e}"
                if len(current_lrs) > 1:
                    msg += f", Range: {min(current_lrs):.2e}-{max(current_lrs):.2e}"

            if skipped_batches > 0:
                msg += f", Skipped: {skipped_batches}"

            self.logger.info(msg)

    def log_epoch_results(self, epoch: int, train_loss: float, train_acc: float,
                         val_loss: float, val_metrics: Dict[str, float],
                         is_best: bool = False, model_saved_path: str = None,
                         elapsed_time: float = None):
        """에포크 결과 로그"""
        # 콘솔/파일 로그
        self.logger.info(f"Epoch {epoch} Results:")
        self.logger.info(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        val_info = f"  Validation - Loss: {val_loss:.4f}"
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, float):
                val_info += f", {metric_name.upper()}: {metric_value:.4f}"
        self.logger.info(val_info)

        if elapsed_time:
            self.logger.info(f"  Epoch time: {elapsed_time:.2f}s")

        if is_best:
            # 다양한 메트릭 키 지원 (auc, avg_auc, top1_acc, acc 등)
            best_metric = val_metrics.get('auc',
                          val_metrics.get('avg_auc',
                          val_metrics.get('top1_acc',
                          val_metrics.get('acc', None))))

            if best_metric is not None:
                self.logger.info(f"  *** NEW BEST MODEL - {best_metric:.4f} ***")
            else:
                self.logger.info(f"  *** NEW BEST MODEL ***")

            if model_saved_path:
                self.logger.info(f"  Model saved: {model_saved_path}")

        # CSV 메트릭 저장
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'is_best': is_best,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # validation metrics 추가
        for metric_name, metric_value in val_metrics.items():
            epoch_data[f'val_{metric_name}'] = metric_value

        self.epoch_metrics.append(epoch_data)
        self._save_metrics_csv()

    def log_training_complete(self, best_metric_value: float, best_metric_name: str = 'AUC',
                             total_time: float = None, final_model_path: str = None):
        """훈련 완료 로그"""
        end_time = datetime.datetime.now()
        if total_time is None:
            total_time = (end_time - self.start_time).total_seconds()

        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Best validation {best_metric_name}: {best_metric_value:.4f}")
        self.logger.info(f"Total training time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if final_model_path:
            self.logger.info(f"Final model saved: {final_model_path}")

        # 요약 파일 생성
        self._save_training_summary(best_metric_value, best_metric_name, total_time, end_time)

    def log_error(self, message: str, exception: Exception = None):
        """에러 로그"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)

    def log_warning(self, message: str):
        """경고 로그"""
        self.logger.warning(message)

    def log_info(self, message: str):
        """일반 정보 로그"""
        self.logger.info(message)

    def log_debug(self, message: str):
        """디버그 로그"""
        self.logger.debug(message)

    def log_validation_details(self, val_metrics: Dict[str, float], detailed_results: Dict = None):
        """검증 세부 결과 로그"""
        self.logger.info("Validation Details:")
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, float):
                self.logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")

        if detailed_results:
            for key, value in detailed_results.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value}")

    def _save_metrics_csv(self):
        """메트릭을 CSV 파일로 저장"""
        if not self.epoch_metrics:
            return

        # 모든 에포크의 필드를 수집하여 전체 필드명 생성 (순서 유지)
        all_fieldnames = []
        seen_fields = set()

        for epoch_data in self.epoch_metrics:
            for key in epoch_data.keys():
                if key not in seen_fields:
                    all_fieldnames.append(key)
                    seen_fields.add(key)

        with open(self.log_files['metrics_csv'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.epoch_metrics)

    def _save_training_summary(self, best_metric_value: float, best_metric_name: str,
                              total_time: float, end_time: datetime.datetime):
        """훈련 요약 파일 저장"""
        summary = [
            f"Training Summary - {self.experiment_name}",
            "=" * 60,
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration: {total_time:.2f}s ({total_time/3600:.2f}h)",
            f"Best Validation {best_metric_name}: {best_metric_value:.4f}",
            "",
            "Epoch-wise Results:",
            "-" * 40
        ]

        for metric in self.epoch_metrics:
            main_metric = metric.get('val_auc', metric.get('val_acc', 'N/A'))
            summary.append(
                f"Epoch {metric['epoch']:2d}: "
                f"Train Loss: {metric['train_loss']:.4f}, "
                f"Val Metric: {main_metric:.4f}, "
                f"Val ACC: {metric.get('val_accuracy', metric.get('val_acc', 0)):.4f}"
                f"{' (BEST)' if metric['is_best'] else ''}"
            )

        summary.extend([
            "",
            f"Log files saved in: {self.log_dir}",
            f"- Training log: {os.path.basename(self.log_files['training_log'])}",
            f"- Config log: {os.path.basename(self.log_files['config_log'])}",
            f"- Metrics CSV: {os.path.basename(self.log_files['metrics_csv'])}",
            f"- Error log: {os.path.basename(self.log_files['error_log'])}",
            f"- Summary: {os.path.basename(self.log_files['summary_log'])}"
        ])

        with open(self.log_files['summary_log'], 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))

    def get_log_files(self) -> Dict[str, str]:
        """생성된 로그 파일 경로들 반환"""
        return self.log_files.copy()

    def close(self):
        """로거 핸들러 정리"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)