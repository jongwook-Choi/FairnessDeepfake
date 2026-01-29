"""
Adversarial Fairness Loss (GRL + Sinkhorn 하이브리드)

GRL 기반 adversarial debiasing과 Sinkhorn 분포 정렬을 결합한 통합 loss

L_stage1 = lambda_race * CE(race_clf(GRL(debiased)), race_label)
         + lambda_gender * CE(gender_clf(GRL(debiased)), gender_label)
         + lambda_sim * (1 - cosine_similarity(clip_feat, debiased_feat))
         + lambda_fairness * Sinkhorn(subgroup_feats, global_feats)
         + lambda_pairwise * PairwiseSinkhorn(subgroup_pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .race_loss import RaceLoss
from .gender_loss import GenderLoss
from .fairness_loss import FairnessLoss, PairwiseFairnessLoss


class AdversarialFairnessLoss(nn.Module):
    """
    GRL + Sinkhorn 하이브리드 Stage1 Loss

    구성 요소:
    1. Race CE Loss (GRL 통과 -> adversarial)
    2. Gender CE Loss (GRL 통과 -> adversarial)
    3. Cosine Similarity Loss (유용 정보 보존)
    4. Sinkhorn Global Fairness Loss (subgroup -> 전체 분포 정렬)
    5. Sinkhorn Pairwise Fairness Loss (subgroup 쌍 정렬)
    """

    def __init__(self,
                 lambda_race=1.0,
                 lambda_gender=0.5,
                 lambda_similarity=2.0,
                 lambda_fairness=0.5,
                 lambda_pairwise=0.3,
                 sinkhorn_blur=1e-4,
                 label_smoothing=0.0):
        """
        Args:
            lambda_race (float): Race adversarial loss 가중치
            lambda_gender (float): Gender adversarial loss 가중치
            lambda_similarity (float): Cosine similarity 보존 loss 가중치
            lambda_fairness (float): Global Sinkhorn fairness loss 가중치
            lambda_pairwise (float): Pairwise Sinkhorn fairness loss 가중치
            sinkhorn_blur (float): Sinkhorn blur 파라미터
            label_smoothing (float): Label smoothing 값
        """
        super().__init__()

        self.lambda_race = lambda_race
        self.lambda_gender = lambda_gender
        self.lambda_similarity = lambda_similarity
        self.lambda_fairness = lambda_fairness
        self.lambda_pairwise = lambda_pairwise

        # Classification losses (GRL이 gradient를 반전시키므로, CE loss 자체는 정상)
        self.race_loss = RaceLoss(label_smoothing=label_smoothing)
        self.gender_loss = GenderLoss(label_smoothing=label_smoothing)

        # Fairness losses (Sinkhorn distance)
        self._init_fairness_losses(sinkhorn_blur)

    def _init_fairness_losses(self, sinkhorn_blur):
        """Fairness loss 초기화"""
        try:
            from geomloss import SamplesLoss
            self.fairness_loss = FairnessLoss(sinkhorn_blur=sinkhorn_blur)
            print("[AdversarialFairnessLoss] Sinkhorn distance 활성화")

            if self.lambda_pairwise > 0:
                self.pairwise_fairness_loss = PairwiseFairnessLoss(
                    sinkhorn_blur=sinkhorn_blur
                )
                print(f"[AdversarialFairnessLoss] Pairwise fairness 활성화 (lambda={self.lambda_pairwise})")
            else:
                self.pairwise_fairness_loss = None
        except ImportError:
            raise ImportError(
                "geomloss is required for Sinkhorn distance. "
                "Please run: pip install geomloss"
            )

    def forward(self, pred_dict, data_dict):
        """
        Combined loss 계산

        Args:
            pred_dict (dict): 모델 예측 결과
                - 'race_logits': Race prediction logits (GRL 통과 후)
                - 'gender_logits': Gender prediction logits (GRL 통과 후)
                - 'clip_features': 원본 CLIP features
                - 'final_features_norm': L2 정규화된 debiased features
            data_dict (dict): 입력 데이터
                - 'race': Race labels
                - 'gender': Gender labels
                - 'subgroup': Subgroup labels

        Returns:
            dict: Loss 딕셔너리
        """
        loss_dict = {}

        # 1. Race adversarial loss (GRL이 gradient를 반전시킴)
        race_logits = pred_dict['race_logits']
        race_labels = data_dict['race']
        loss_race = self.race_loss(race_logits, race_labels)
        loss_dict['race'] = loss_race
        loss_dict['race_weighted'] = self.lambda_race * loss_race

        # 2. Gender adversarial loss (GRL이 gradient를 반전시킴)
        gender_logits = pred_dict['gender_logits']
        gender_labels = data_dict['gender']
        loss_gender = self.gender_loss(gender_logits, gender_labels)
        loss_dict['gender'] = loss_gender
        loss_dict['gender_weighted'] = self.lambda_gender * loss_gender

        # 3. Cosine similarity 보존 loss
        clip_features = pred_dict['clip_features']
        debiased_features = pred_dict['final_features_norm']

        # clip_features도 정규화
        clip_norm = F.normalize(clip_features, dim=-1)
        debiased_norm = F.normalize(debiased_features, dim=-1)

        # cosine similarity: 1 - cos(clip, debiased)
        cosine_sim = F.cosine_similarity(clip_norm, debiased_norm, dim=-1).mean()
        loss_similarity = 1.0 - cosine_sim
        loss_dict['similarity'] = loss_similarity
        loss_dict['similarity_weighted'] = self.lambda_similarity * loss_similarity
        loss_dict['cosine_sim'] = cosine_sim.item()

        # 4. Global Sinkhorn fairness loss
        features = pred_dict['final_features_norm']
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

        # 5. Pairwise Sinkhorn fairness loss
        if self.pairwise_fairness_loss is not None and self.lambda_pairwise > 0:
            loss_pairwise, pair_losses = self.pairwise_fairness_loss(features, subgroups)
            loss_dict['pairwise_fairness'] = loss_pairwise
            loss_dict['pairwise_fairness_weighted'] = self.lambda_pairwise * loss_pairwise
            loss_dict['pair_losses'] = pair_losses
        else:
            loss_dict['pairwise_fairness'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pairwise_fairness_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pair_losses'] = {}

        # Total loss
        loss_dict['overall'] = (
            loss_dict['race_weighted'] +
            loss_dict['gender_weighted'] +
            loss_dict['similarity_weighted'] +
            loss_dict['fairness_weighted'] +
            loss_dict['pairwise_fairness_weighted']
        )

        return loss_dict

    def get_metrics(self, pred_dict, data_dict):
        """Classification metrics 계산"""
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

        # Cosine similarity (이미 loss_dict에서 계산, 여기선 재계산)
        clip_features = pred_dict['clip_features']
        debiased_features = pred_dict['final_features_norm']
        clip_norm = F.normalize(clip_features, dim=-1)
        debiased_norm = F.normalize(debiased_features, dim=-1)
        metrics['cosine_sim'] = F.cosine_similarity(
            clip_norm, debiased_norm, dim=-1
        ).mean().item()

        return metrics

    def update_weights(self, **kwargs):
        """Loss 가중치 동적 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        print(f"[AdversarialFairnessLoss] Updated weights: "
              f"race={self.lambda_race}, gender={self.lambda_gender}, "
              f"sim={self.lambda_similarity}, "
              f"fairness={self.lambda_fairness}, pairwise={self.lambda_pairwise}")
