#!/usr/bin/env python3
"""
Stage 2 TXT Cross-dataset + Fairness 평가 스크립트

주요 기능:
1. Cross-dataset 평가: ff++ (Train) → celebdf, dfd, dfdc (Test)
2. Generalization 성능: AUC, ACC, EER, AP
3. Fairness 성능:
   - Intra-group: Gender별, Race별 공정성
   - Inter-group: Gender × Race 8개 서브그룹 간 공정성
   - 메트릭: F_FPR, F_OAE, F_DP, F_MEO

Usage:
    python test_stage2txt.py --checkpoint logs/xxx/checkpoint_best.pth
    python test_stage2txt.py --checkpoint logs/xxx/checkpoint_best.pth --test-datasets ff++ celebdf dfd dfdc
"""

import os
import sys
import argparse
import yaml
import json
import datetime
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.stage2txt_model import Stage2TxtModel
from dataset.fairness_dataset import FairnessDataset
from utils.fairness_metrics import compute_fairness_metrics, print_fairness_report, compute_eer
from utils.fairness_metrics_intersection import (
    compute_fairness_with_indices,
    print_intersection_fairness_report
)


class Stage2TxtCrossDatasetTester:
    """
    Stage 2 TXT Cross-dataset + Fairness 평가 클래스

    Features:
    - Cross-dataset 평가 (ff++ → celebdf, dfd, dfdc)
    - Intra-group Fairness (Gender, Race)
    - Inter-group Fairness (Gender × Race)
    - Text Anchors 기반 similarity 분석
    """

    def __init__(self, config, checkpoint_path):
        """
        Args:
            config (dict): 테스트 설정
            checkpoint_path (str): 체크포인트 경로
        """
        self.config = config
        self.checkpoint_path = checkpoint_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 결과 저장 디렉토리
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_result_dir = config.get('result_dir', '/workspace/code/CLIP_stage2txt/results')
        self.result_dir = os.path.join(base_result_dir, timestamp)
        os.makedirs(self.result_dir, exist_ok=True)

        # Fairness 데이터셋 설정
        self.fairness_root = config.get('fairness_root', '/workspace/datasets/fairness')

        # Intersection label 매핑
        self.intersec_label_mappings = {
            'ff++': {
                0: ('male', 'asian'), 1: ('male', 'white'), 2: ('male', 'black'), 3: ('male', 'others'),
                4: ('nonmale', 'asian'), 5: ('nonmale', 'white'), 6: ('nonmale', 'black'), 7: ('nonmale', 'others')
            },
            'dfdc': {
                0: ('male', 'asian'), 1: ('male', 'white'), 2: ('male', 'black'), 3: ('male', 'others'),
                4: ('nonmale', 'asian'), 5: ('nonmale', 'white'), 6: ('nonmale', 'black'), 7: ('nonmale', 'others')
            },
            'dfd': {
                0: ('male', 'white'), 1: ('male', 'black'), 2: ('male', 'others'),
                3: ('nonmale', 'white'), 4: ('nonmale', 'black'), 5: ('nonmale', 'others')
            },
            'celebdf': {
                0: ('male', 'white'), 1: ('male', 'black'), 2: ('male', 'others'),
                3: ('nonmale', 'white'), 4: ('nonmale', 'black'), 5: ('nonmale', 'others')
            }
        }

        # 결과 저장용
        self.results = {}
        self.fairness_results = {}

        print(f"Results will be saved to: {self.result_dir}")

    def load_model(self):
        """체크포인트에서 모델 로드"""
        print(f"\nLoading model from: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Config에서 모델 설정 추출
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            model_config = self.config.get('model', {})

        # 모델 생성
        model = Stage2TxtModel(
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

        # 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Text Anchors 로드 (체크포인트에 별도로 저장된 경우)
        if 'text_anchors' in checkpoint and model.text_anchors is None:
            model.register_buffer('text_anchors', checkpoint['text_anchors'])
            print(f"  Loaded Text Anchors from checkpoint: {model.text_anchors.shape}")

        model.to(self.device)
        model.eval()

        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"  Best AUC: {metrics.get('auc', 0):.4f}")

        # Text Anchors 정보 출력
        if model.text_anchors is not None:
            print(f"  Text Anchors: {model.text_anchors.shape}")
        else:
            print(f"  Text Anchors: Not available")

        return model

    def create_test_dataloader(self, dataset_name):
        """테스트 데이터로더 생성"""
        test_config = {
            'fairness_root': self.fairness_root,
            'test_dataset': [dataset_name],
            'resolution': self.config.get('resolution', 256),
            'mean': self.config.get('mean', [0.485, 0.456, 0.406]),
            'std': self.config.get('std', [0.229, 0.224, 0.225]),
            'use_data_augmentation': False,
            'use_balance_sampling': False,
            'skip_unknown_attributes': False,
        }

        dataset = FairnessDataset(config=test_config, mode='test')

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('test_batch_size', 64),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            drop_last=False
        )

        print(f"  Created dataloader: {len(dataset)} samples, {len(dataloader)} batches")

        return dataloader, dataset

    def compute_basic_metrics(self, labels, probs, preds):
        """기본 평가 메트릭 계산"""
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5

        try:
            acc = accuracy_score(labels, preds)
        except:
            acc = 0.0

        try:
            ap = average_precision_score(labels, probs)
        except:
            ap = 0.0

        # EER 계산
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            fnr = 1 - tpr
            diff = np.abs(fpr - fnr)
            min_idx = np.argmin(diff)
            eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
        except:
            eer = 0.5

        return {
            'auc': auc,
            'acc': acc,
            'eer': eer,
            'ap': ap
        }

    def extract_attributes_from_dataset(self, dataset, dataset_name):
        """데이터셋에서 속성 정보 추출"""
        attr_dict = {'gender': [], 'race': []}

        if dataset_name not in self.intersec_label_mappings:
            print(f"Warning: No intersec_label mapping for {dataset_name}")
            return attr_dict

        mapping = self.intersec_label_mappings[dataset_name]

        for intersec_label in dataset.intersec_label_list:
            if intersec_label in mapping:
                gender, race = mapping[intersec_label]
                attr_dict['gender'].append(gender)
                attr_dict['race'].append(race)
            else:
                attr_dict['gender'].append('unknown')
                attr_dict['race'].append('unknown')

        return attr_dict

    def evaluate_model(self, model, dataloader, dataset, dataset_name):
        """모델 평가 (성능 + Fairness)"""
        model.eval()

        all_labels = []
        all_probs = []
        all_preds = []
        all_text_similarities = []

        print(f"\nEvaluating on {dataset_name}...")

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(dataloader):
                if data_dict is None:
                    continue

                # 데이터를 device로 이동
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                try:
                    # Forward pass
                    pred_dict = model(data_dict, inference=False)

                    # 예측 결과 추출
                    probs = pred_dict['prob'].cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    labels = data_dict['label'].cpu().numpy()

                    all_labels.extend(labels)
                    all_probs.extend(probs)
                    all_preds.extend(preds)

                    # Text Similarity 수집 (있는 경우)
                    if 'text_similarity' in pred_dict:
                        all_text_similarities.extend(pred_dict['text_similarity'].cpu().numpy())

                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    continue

                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

        if len(all_labels) == 0:
            print(f"  Warning: No valid predictions for {dataset_name}")
            return None, None

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        # 1. 기본 성능 메트릭
        metrics = self.compute_basic_metrics(all_labels, all_probs, all_preds)
        metrics['num_samples'] = len(all_labels)

        print(f"\n  {dataset_name} Performance:")
        print(f"    AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}, "
              f"EER: {metrics['eer']:.4f}, AP: {metrics['ap']:.4f}")

        # Text Similarity 분석 (있는 경우)
        if len(all_text_similarities) > 0:
            all_text_similarities = np.array(all_text_similarities)
            print(f"\n  Text Similarity Analysis:")
            print(f"    Shape: {all_text_similarities.shape}")
            print(f"    Mean: {all_text_similarities.mean():.4f}")
            print(f"    Std: {all_text_similarities.std():.4f}")

        # 2. Fairness 메트릭
        fairness_results = None
        try:
            print(f"\n  Computing Fairness Metrics...")

            # 속성 정보 추출
            attr_dict = self.extract_attributes_from_dataset(dataset, dataset_name)

            # DataFrame 생성
            df_fairness = pd.DataFrame({
                'prediction': all_probs,
                'label': all_labels,
                'gender': attr_dict['gender'][:len(all_labels)],
                'race': attr_dict['race'][:len(all_labels)]
            })

            # Unknown 필터링
            df_fairness = df_fairness[
                (df_fairness['gender'] != 'unknown') &
                (df_fairness['race'] != 'unknown')
            ]

            if len(df_fairness) > 0:
                print(f"    Valid samples for fairness: {len(df_fairness)}")

                # 샘플 분포 출력
                print(f"\n    Sample distribution:")
                for attr in ['gender', 'race']:
                    counts = df_fairness[attr].value_counts()
                    print(f"      {attr}: {dict(counts)}")

                # Intersection 그룹 분포
                intersection_counts = df_fairness.groupby(['gender', 'race']).size()
                print(f"\n    Intersection groups: {len(intersection_counts)}")

                # Fairness 메트릭 계산
                fairness_results = compute_fairness_with_indices(
                    df_fairness,
                    ['gender', 'race'],
                    target_column='label',
                    prediction_column='prediction'
                )

                # 요약 출력
                self._print_fairness_summary(fairness_results)

        except Exception as e:
            print(f"  Warning: Fairness evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        return metrics, fairness_results

    def _print_fairness_summary(self, fairness_results):
        """Fairness 결과 요약 출력"""
        print(f"\n    Fairness Summary:")

        # Intra-group
        if 'intra_fairness' in fairness_results:
            intra = fairness_results['intra_fairness']
            print(f"      [Intra-group]")
            for key, value in intra.items():
                if 'F_FPR' in key or 'F_OAE' in key:
                    print(f"        {key}: {value:.3f}%")

        # Inter-group
        if 'inter_fairness' in fairness_results:
            inter = fairness_results['inter_fairness']
            print(f"      [Inter-group]")
            for key, value in inter.items():
                print(f"        {key}: {value:.3f}%")

    def test_all_datasets(self, test_datasets=None):
        """모든 테스트 데이터셋 평가"""
        if test_datasets is None:
            test_datasets = ['ff++', 'celebdf', 'dfd', 'dfdc']

        print("\n" + "=" * 60)
        print("Stage 2 TXT Cross-dataset + Fairness Evaluation")
        print("=" * 60)
        print(f"Checkpoint: {os.path.basename(self.checkpoint_path)}")
        print(f"Test datasets: {test_datasets}")

        # 모델 로드
        model = self.load_model()

        start_time = time.time()

        # 각 데이터셋에 대해 평가
        for dataset_name in test_datasets:
            print(f"\n{'='*40}")
            print(f"Testing on {dataset_name}")
            print(f"{'='*40}")

            try:
                dataloader, dataset = self.create_test_dataloader(dataset_name)
                metrics, fairness = self.evaluate_model(model, dataloader, dataset, dataset_name)

                if metrics is not None:
                    self.results[dataset_name] = metrics
                if fairness is not None:
                    self.fairness_results[dataset_name] = fairness

            except Exception as e:
                print(f"Error testing on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - start_time
        print(f"\nTotal evaluation time: {total_time:.2f}s")

        return self.results, self.fairness_results

    def generate_report(self):
        """상세 리포트 생성"""
        print("\n" + "=" * 80)
        print("STAGE 2 TXT CROSS-DATASET + FAIRNESS EVALUATION REPORT")
        print("=" * 80)
        print(f"Checkpoint: {os.path.basename(self.checkpoint_path)}")
        print(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 성능 요약
        print("\n" + "-" * 40)
        print("1. GENERALIZATION PERFORMANCE")
        print("-" * 40)
        print(f"{'Dataset':<15} {'Samples':>8} {'AUC':>8} {'ACC':>8} {'EER':>8} {'AP':>8}")
        print("-" * 60)

        valid_aucs = []
        for dataset, metrics in self.results.items():
            print(f"{dataset:<15} {metrics['num_samples']:>8} "
                  f"{metrics['auc']:>8.4f} {metrics['acc']:>8.4f} "
                  f"{metrics['eer']:>8.4f} {metrics['ap']:>8.4f}")
            valid_aucs.append(metrics['auc'])

        if valid_aucs:
            print("-" * 60)
            print(f"{'Average':<15} {'':<8} {np.mean(valid_aucs):>8.4f}")

        # 2. Fairness 요약
        if self.fairness_results:
            print("\n" + "-" * 40)
            print("2. FAIRNESS SUMMARY")
            print("-" * 40)

            for dataset, fairness in self.fairness_results.items():
                print(f"\n[{dataset}]")

                # Intra-group
                if 'intra_fairness' in fairness:
                    print("  Intra-group Fairness:")
                    intra = fairness['intra_fairness']
                    for key in ['gender_F_FPR', 'race_F_FPR', 'gender_F_OAE', 'race_F_OAE']:
                        if key in intra:
                            print(f"    {key}: {intra[key]:.3f}%")

                # Inter-group
                if 'inter_fairness' in fairness:
                    print("  Inter-group Fairness:")
                    inter = fairness['inter_fairness']
                    for key in ['inter_F_FPR', 'inter_F_OAE', 'inter_F_DP', 'inter_F_MEO']:
                        if key in inter:
                            print(f"    {key}: {inter[key]:.3f}%")

        print("=" * 80)

        # 결과 저장
        self._save_results()

    def _save_results(self):
        """결과를 파일로 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV 저장 (성능 메트릭)
        if self.results:
            csv_data = []
            for dataset, metrics in self.results.items():
                csv_data.append({
                    'Dataset': dataset,
                    'AUC': metrics['auc'],
                    'ACC': metrics['acc'],
                    'EER': metrics['eer'],
                    'AP': metrics['ap'],
                    'Samples': metrics['num_samples']
                })

            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.result_dir, f'performance_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nPerformance results saved to: {csv_path}")

        # JSON 저장 (전체 결과)
        json_data = {
            'checkpoint': self.checkpoint_path,
            'timestamp': datetime.datetime.now().isoformat(),
            'performance': self.results,
            'fairness': self._convert_numpy_to_python(self.fairness_results)
        }

        json_path = os.path.join(self.result_dir, f'full_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"Full results saved to: {json_path}")

    def _convert_numpy_to_python(self, obj):
        """Numpy types를 Python types로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Stage 2 TXT Cross-dataset + Fairness Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to test config file')

    parser.add_argument('--test-datasets', nargs='+',
                        default=['ff++', 'celebdf', 'dfd', 'dfdc'],
                        help='Test datasets')

    parser.add_argument('--result-dir', type=str,
                        default='/workspace/code/CLIP_stage2txt/results',
                        help='Directory to save results')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Test batch size')

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 체크포인트 확인
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # 설정 로드
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 기본 설정
        config = {
            'fairness_root': '/workspace/datasets/fairness',
            'resolution': 256,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'test_batch_size': args.batch_size,
            'result_dir': args.result_dir,
            'clip_download_root': '/data/cuixinjie/weights',
        }

    # 테스터 생성
    tester = Stage2TxtCrossDatasetTester(config, args.checkpoint)

    # 평가 실행
    results, fairness_results = tester.test_all_datasets(args.test_datasets)

    # 리포트 생성
    tester.generate_report()

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
