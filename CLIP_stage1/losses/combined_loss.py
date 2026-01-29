"""
Combined Loss for CLIP Stage1
Race Loss + Gender Loss + Similarity Loss + Fairness Loss 조합

v2: AdversarialFairnessLoss와 함께 사용 가능
    - use_adversarial=True: AdversarialFairnessLoss 사용 (GRL 기반)
    - use_adversarial=False: 기존 CombinedLoss 사용 (호환성 유지)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .race_loss import RaceLoss
from .gender_loss import GenderLoss
from .fairness_loss import FairnessLoss, PairwiseFairnessLoss, MMDFairnessLoss


class CombinedLoss(nn.Module):
    """
    Stage1 학습을 위한 Combined Loss

    Total Loss = λ_race * Loss_race + λ_gender * Loss_gender + λ_fairness * Loss_fairness + λ_pairwise_fairness * Loss_pairwise_fairness

    기본 가중치:
        - λ_race = 1.0
        - λ_gender = 0.1 (낮은 가중치)
        - λ_fairness = 1e-4 (Global fairness: 각 subgroup ↔ 전체 분포)
        - λ_pairwise_fairness = 0.0 (Pairwise fairness: 모든 subgroup 쌍끼리)
    """

    def __init__(self,
                 lambda_race=1.0,
                 lambda_gender=0.1,
                 lambda_fairness=1e-4,
                 lambda_pairwise_fairness=0.0,
                 sinkhorn_blur=1e-4,
                 use_mmd_fallback=True,
                 label_smoothing=0.0):
        """
        Args:
            lambda_race (float): Race loss 가중치
            lambda_gender (float): Gender loss 가중치
            lambda_fairness (float): Fairness loss 가중치 (Global fairness)
            lambda_pairwise_fairness (float): Pairwise fairness loss 가중치
            sinkhorn_blur (float): Sinkhorn blur 파라미터
            use_mmd_fallback (bool): geomloss가 없을 때 MMD 사용 여부
            label_smoothing (float): Label smoothing 값
        """
        super().__init__()

        self.lambda_race = lambda_race
        self.lambda_gender = lambda_gender
        self.lambda_fairness = lambda_fairness
        self.lambda_pairwise_fairness = lambda_pairwise_fairness

        # Loss 함수들 초기화
        self.race_loss = RaceLoss(label_smoothing=label_smoothing)
        self.gender_loss = GenderLoss(label_smoothing=label_smoothing)

        # Fairness loss 초기화
        try:
            from geomloss import SamplesLoss
            self.fairness_loss = FairnessLoss(sinkhorn_blur=sinkhorn_blur)
            print("[CombinedLoss] Using Sinkhorn distance for fairness loss")

            # Pairwise fairness loss 초기화
            if lambda_pairwise_fairness > 0:
                self.pairwise_fairness_loss = PairwiseFairnessLoss(sinkhorn_blur=sinkhorn_blur)
                print(f"[CombinedLoss] Pairwise fairness loss enabled (lambda={lambda_pairwise_fairness})")
            else:
                self.pairwise_fairness_loss = None
        except ImportError:
            if use_mmd_fallback:
                self.fairness_loss = MMDFairnessLoss()
                print("[CombinedLoss] geomloss not found, using MMD fallback")
            else:
                self.fairness_loss = None
                print("[CombinedLoss] Fairness loss disabled (geomloss not installed)")
            self.pairwise_fairness_loss = None

    def forward(self, pred_dict, data_dict):
        """
        Combined loss 계산

        Args:
            pred_dict (dict): 모델 예측 결과
                - 'race_logits': Race prediction logits
                - 'gender_logits': Gender prediction logits
                - 'final_features_norm': L2 정규화된 Final features
            data_dict (dict): 입력 데이터
                - 'race': Race labels
                - 'gender': Gender labels
                - 'subgroup': Subgroup labels

        Returns:
            dict: Loss 딕셔너리
                - 'race': Race loss (weighted)
                - 'gender': Gender loss (weighted)
                - 'fairness': Fairness loss (weighted)
                - 'overall': Total loss
        """
        loss_dict = {}

        # Race loss
        race_logits = pred_dict['race_logits']
        race_labels = data_dict['race']
        loss_race = self.race_loss(race_logits, race_labels)
        loss_dict['race'] = loss_race
        loss_dict['race_weighted'] = self.lambda_race * loss_race

        # Gender loss
        gender_logits = pred_dict['gender_logits']
        gender_labels = data_dict['gender']
        loss_gender = self.gender_loss(gender_logits, gender_labels)
        loss_dict['gender'] = loss_gender
        loss_dict['gender_weighted'] = self.lambda_gender * loss_gender

        # Fairness loss (정규화된 features 사용) - Global fairness
        features = pred_dict['final_features_norm']  # L2 정규화된 features
        subgroups = data_dict['subgroup']

        if self.fairness_loss is not None and self.lambda_fairness > 0:
            loss_fairness, subgroup_losses = self.fairness_loss(features, subgroups)
            loss_dict['fairness'] = loss_fairness
            loss_dict['fairness_weighted'] = self.lambda_fairness * loss_fairness
            loss_dict['subgroup_losses'] = subgroup_losses
        else:
            loss_dict['fairness'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['fairness_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['subgroup_losses'] = {}

        # Pairwise Fairness loss - 모든 subgroup 쌍끼리 비교
        if self.pairwise_fairness_loss is not None and self.lambda_pairwise_fairness > 0:
            loss_pairwise, pair_losses = self.pairwise_fairness_loss(features, subgroups)
            loss_dict['pairwise_fairness'] = loss_pairwise
            loss_dict['pairwise_fairness_weighted'] = self.lambda_pairwise_fairness * loss_pairwise
            loss_dict['pair_losses'] = pair_losses
        else:
            loss_dict['pairwise_fairness'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pairwise_fairness_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pair_losses'] = {}

        # Total loss
        loss_dict['overall'] = (
            loss_dict['race_weighted'] +
            loss_dict['gender_weighted'] +
            loss_dict['fairness_weighted'] +
            loss_dict['pairwise_fairness_weighted']
        )

        return loss_dict

    def get_metrics(self, pred_dict, data_dict):
        """
        Classification metrics 계산

        Args:
            pred_dict (dict): 모델 예측 결과
            data_dict (dict): 입력 데이터

        Returns:
            dict: 메트릭 딕셔너리
        """
        metrics = {}

        # Race accuracy
        race_logits = pred_dict['race_logits']
        race_labels = data_dict['race']
        metrics['race_acc'] = self.race_loss.get_accuracy(race_logits, race_labels)
        metrics['race_per_class'] = self.race_loss.get_per_class_accuracy(race_logits, race_labels)

        # Gender accuracy
        gender_logits = pred_dict['gender_logits']
        gender_labels = data_dict['gender']
        metrics['gender_acc'] = self.gender_loss.get_accuracy(gender_logits, gender_labels)
        metrics['gender_per_class'] = self.gender_loss.get_per_class_accuracy(gender_logits, gender_labels)

        return metrics

    def update_weights(self, lambda_race=None, lambda_gender=None,
                       lambda_fairness=None, lambda_pairwise_fairness=None):
        """
        Loss 가중치 업데이트 (학습 중 동적 조절 가능)

        Args:
            lambda_race (float, optional): 새로운 race loss 가중치
            lambda_gender (float, optional): 새로운 gender loss 가중치
            lambda_fairness (float, optional): 새로운 fairness loss 가중치 (Global)
            lambda_pairwise_fairness (float, optional): 새로운 pairwise fairness loss 가중치
        """
        if lambda_race is not None:
            self.lambda_race = lambda_race
        if lambda_gender is not None:
            self.lambda_gender = lambda_gender
        if lambda_fairness is not None:
            self.lambda_fairness = lambda_fairness
        if lambda_pairwise_fairness is not None:
            self.lambda_pairwise_fairness = lambda_pairwise_fairness

        print(f"[CombinedLoss] Updated weights: "
              f"race={self.lambda_race}, gender={self.lambda_gender}, "
              f"fairness={self.lambda_fairness}, pairwise_fairness={self.lambda_pairwise_fairness}")


def create_stage1_loss(config):
    """
    Config에 따라 Stage 1 loss 생성

    Args:
        config (dict): 설정 딕셔너리

    Returns:
        nn.Module: AdversarialFairnessLoss 또는 CombinedLoss
    """
    use_grl = config.get('use_grl', True)

    if use_grl:
        from .adversarial_fairness_loss import AdversarialFairnessLoss
        loss_fn = AdversarialFairnessLoss(
            lambda_race=config.get('lambda_race', 1.0),
            lambda_gender=config.get('lambda_gender', 0.5),
            lambda_similarity=config.get('lambda_similarity', 2.0),
            lambda_fairness=config.get('lambda_fairness', 0.5),
            lambda_pairwise=config.get('lambda_pairwise', 0.3),
            sinkhorn_blur=config.get('sinkhorn_blur', 1e-4),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        print("[create_stage1_loss] AdversarialFairnessLoss (GRL + Sinkhorn) 생성")
    else:
        loss_fn = CombinedLoss(
            lambda_race=config.get('lambda_race', 1.0),
            lambda_gender=config.get('lambda_gender', 0.1),
            lambda_fairness=config.get('lambda_fairness', 1e-4),
            lambda_pairwise_fairness=config.get('lambda_pairwise_fairness', 0.0),
            sinkhorn_blur=config.get('sinkhorn_blur', 1e-4),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        print("[create_stage1_loss] CombinedLoss (기존 방식) 생성")

    return loss_fn
