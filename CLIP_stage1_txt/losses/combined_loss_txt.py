"""
Combined Loss with Text for CLIP Stage1

기존 Combined Loss (Race + Gender + Fairness) + Text Loss (Alignment + Consistency)

Total Loss = λ_race × L_race
           + λ_gender × L_gender
           + λ_fairness × L_fairness (Sinkhorn)
           + λ_text_align × L_text_align
           + λ_text_consist × L_text_consist
"""

import torch
import torch.nn as nn
from .race_loss import RaceLoss
from .gender_loss import GenderLoss
from .fairness_loss import FairnessLoss, PairwiseFairnessLoss, MMDFairnessLoss
from .text_alignment_loss import TextVisualAlignmentLoss, TextConsistencyLoss


class CombinedLossWithText(nn.Module):
    """
    Stage1 학습을 위한 Combined Loss with Text

    기본 가중치:
        - λ_race = 1.0
        - λ_gender = 0.1
        - λ_fairness = 0.5
        - λ_text_align = 0.1
        - λ_text_consist = 0.01
    """

    def __init__(self,
                 lambda_race=1.0,
                 lambda_gender=0.1,
                 lambda_fairness=0.5,
                 lambda_pairwise_fairness=0.0,
                 lambda_text_align=0.1,
                 lambda_text_consist=0.01,
                 sinkhorn_blur=1e-4,
                 use_mmd_fallback=True,
                 label_smoothing=0.0,
                 consistency_loss_type='pairwise'):
        """
        Args:
            lambda_race (float): Race loss 가중치
            lambda_gender (float): Gender loss 가중치
            lambda_fairness (float): Fairness loss (Sinkhorn) 가중치
            lambda_pairwise_fairness (float): Pairwise Fairness loss 가중치
            lambda_text_align (float): Text-Visual Alignment loss 가중치
            lambda_text_consist (float): Text Consistency loss 가중치
            sinkhorn_blur (float): Sinkhorn blur 파라미터
            use_mmd_fallback (bool): geomloss가 없을 때 MMD 사용 여부
            label_smoothing (float): Label smoothing 값
            consistency_loss_type (str): 'pairwise' or 'variance'
        """
        super().__init__()

        self.lambda_race = lambda_race
        self.lambda_gender = lambda_gender
        self.lambda_fairness = lambda_fairness
        self.lambda_pairwise_fairness = lambda_pairwise_fairness
        self.lambda_text_align = lambda_text_align
        self.lambda_text_consist = lambda_text_consist

        # 기존 Loss 함수들 초기화
        self.race_loss = RaceLoss(label_smoothing=label_smoothing)
        self.gender_loss = GenderLoss(label_smoothing=label_smoothing)

        # Fairness loss 초기화
        try:
            from geomloss import SamplesLoss
            self.fairness_loss = FairnessLoss(sinkhorn_blur=sinkhorn_blur)
            print("[CombinedLossWithText] Using Sinkhorn distance for fairness loss")
        except ImportError:
            if use_mmd_fallback:
                self.fairness_loss = MMDFairnessLoss()
                print("[CombinedLossWithText] geomloss not found, using MMD fallback")
            else:
                self.fairness_loss = None
                print("[CombinedLossWithText] Fairness loss disabled (geomloss not installed)")

        # Pairwise Fairness loss 초기화
        if lambda_pairwise_fairness > 0:
            self.pairwise_fairness_loss = PairwiseFairnessLoss(sinkhorn_blur=sinkhorn_blur)
            print(f"[CombinedLossWithText] Pairwise fairness loss enabled (lambda={lambda_pairwise_fairness})")
        else:
            self.pairwise_fairness_loss = None

        # 신규 Text Loss 함수들 초기화
        self.text_align_loss = TextVisualAlignmentLoss()
        self.text_consist_loss = TextConsistencyLoss(loss_type=consistency_loss_type)

        print(f"[CombinedLossWithText] Initialized with:")
        print(f"  λ_race={lambda_race}, λ_gender={lambda_gender}, λ_fairness={lambda_fairness}")
        print(f"  λ_text_align={lambda_text_align}, λ_text_consist={lambda_text_consist}")

    def forward(self, pred_dict, data_dict):
        """
        Combined loss 계산

        Args:
            pred_dict (dict): 모델 예측 결과
                - 'race_logits': Race prediction logits
                - 'gender_logits': Gender prediction logits
                - 'final_features_norm': L2 정규화된 Final features
                - 'text_anchors': Text anchors [num_subgroups, feature_dim]
            data_dict (dict): 입력 데이터
                - 'race': Race labels
                - 'gender': Gender labels
                - 'subgroup': Subgroup labels

        Returns:
            dict: Loss 딕셔너리
                - 'race': Race loss
                - 'gender': Gender loss
                - 'fairness': Fairness loss
                - 'text_align': Text-Visual Alignment loss
                - 'text_consist': Text Consistency loss
                - 'overall': Total loss
        """
        loss_dict = {}

        # ========== 기존 Loss 계산 ==========

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

        # Fairness loss (정규화된 features 사용)
        if self.fairness_loss is not None and self.lambda_fairness > 0:
            features = pred_dict['final_features_norm']
            subgroups = data_dict['subgroup']
            loss_fairness, subgroup_losses = self.fairness_loss(features, subgroups)
            loss_dict['fairness'] = loss_fairness
            loss_dict['fairness_weighted'] = self.lambda_fairness * loss_fairness
            loss_dict['subgroup_losses'] = subgroup_losses
        else:
            loss_dict['fairness'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['fairness_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['subgroup_losses'] = {}

        # Pairwise Fairness loss
        if self.pairwise_fairness_loss is not None and self.lambda_pairwise_fairness > 0:
            features = pred_dict['final_features_norm']
            subgroups = data_dict['subgroup']
            loss_pairwise, pair_losses = self.pairwise_fairness_loss(features, subgroups)
            loss_dict['pairwise_fairness'] = loss_pairwise
            loss_dict['pairwise_fairness_weighted'] = self.lambda_pairwise_fairness * loss_pairwise
            loss_dict['pair_losses'] = pair_losses
        else:
            loss_dict['pairwise_fairness'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pairwise_fairness_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['pair_losses'] = {}

        # ========== 신규 Text Loss 계산 ==========

        # Text-Visual Alignment Loss
        if self.lambda_text_align > 0 and 'text_anchors' in pred_dict:
            visual_features = pred_dict['final_features_norm']
            subgroups = data_dict['subgroup']
            text_anchors = pred_dict['text_anchors']

            loss_text_align, align_info = self.text_align_loss(
                visual_features, subgroups, text_anchors
            )
            loss_dict['text_align'] = loss_text_align
            loss_dict['text_align_weighted'] = self.lambda_text_align * loss_text_align
            loss_dict['text_align_info'] = align_info
        else:
            loss_dict['text_align'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['text_align_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['text_align_info'] = {}

        # Text Consistency Loss
        if self.lambda_text_consist > 0 and 'text_anchors' in pred_dict:
            text_anchors = pred_dict['text_anchors']

            loss_text_consist, consist_info = self.text_consist_loss(text_anchors)
            loss_dict['text_consist'] = loss_text_consist
            loss_dict['text_consist_weighted'] = self.lambda_text_consist * loss_text_consist
            loss_dict['text_consist_info'] = consist_info
        else:
            loss_dict['text_consist'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['text_consist_weighted'] = torch.tensor(0.0, device=race_logits.device)
            loss_dict['text_consist_info'] = {}

        # ========== Total Loss ==========
        loss_dict['overall'] = (
            loss_dict['race_weighted'] +
            loss_dict['gender_weighted'] +
            loss_dict['fairness_weighted'] +
            loss_dict['pairwise_fairness_weighted'] +
            loss_dict['text_align_weighted'] +
            loss_dict['text_consist_weighted']
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

        # Text-Visual Alignment (추가 메트릭)
        if 'text_anchors' in pred_dict:
            visual_features = pred_dict['final_features_norm']
            subgroups = data_dict['subgroup']
            text_anchors = pred_dict['text_anchors']

            _, align_info = self.text_align_loss(visual_features, subgroups, text_anchors)
            metrics['text_visual_sim'] = align_info['mean_cosine_sim']

        return metrics

    def update_weights(self,
                       lambda_race=None,
                       lambda_gender=None,
                       lambda_fairness=None,
                       lambda_pairwise_fairness=None,
                       lambda_text_align=None,
                       lambda_text_consist=None):
        """
        Loss 가중치 업데이트 (학습 중 동적 조절 가능)

        Args:
            lambda_race (float, optional): 새로운 race loss 가중치
            lambda_gender (float, optional): 새로운 gender loss 가중치
            lambda_fairness (float, optional): 새로운 fairness loss 가중치
            lambda_pairwise_fairness (float, optional): 새로운 pairwise fairness loss 가중치
            lambda_text_align (float, optional): 새로운 text alignment loss 가중치
            lambda_text_consist (float, optional): 새로운 text consistency loss 가중치
        """
        if lambda_race is not None:
            self.lambda_race = lambda_race
        if lambda_gender is not None:
            self.lambda_gender = lambda_gender
        if lambda_fairness is not None:
            self.lambda_fairness = lambda_fairness
        if lambda_pairwise_fairness is not None:
            self.lambda_pairwise_fairness = lambda_pairwise_fairness
        if lambda_text_align is not None:
            self.lambda_text_align = lambda_text_align
        if lambda_text_consist is not None:
            self.lambda_text_consist = lambda_text_consist

        print(f"[CombinedLossWithText] Updated weights: "
              f"race={self.lambda_race}, gender={self.lambda_gender}, "
              f"fairness={self.lambda_fairness}, pairwise_fairness={self.lambda_pairwise_fairness}, "
              f"text_align={self.lambda_text_align}, text_consist={self.lambda_text_consist}")
