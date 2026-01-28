"""
Deepfake Detection Loss Functions
CrossEntropy Loss 및 Focal Loss 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Deepfake Detection용 Loss 클래스

    CrossEntropy Loss를 기본으로 사용하며,
    선택적으로 Focal Loss, Label Smoothing 등을 적용 가능
    """

    def __init__(self,
                 loss_type='cross_entropy',
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 label_smoothing=0.0,
                 class_weights=None):
        """
        Args:
            loss_type (str): Loss 타입 ('cross_entropy', 'focal', 'bce')
            focal_alpha (float): Focal Loss의 alpha 파라미터
            focal_gamma (float): Focal Loss의 gamma 파라미터
            label_smoothing (float): Label smoothing 계수
            class_weights (list): 클래스별 가중치 [real_weight, fake_weight]
        """
        super().__init__()

        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        # Loss function 설정
        if loss_type == 'cross_entropy':
            if label_smoothing > 0:
                self.loss_fn = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    label_smoothing=label_smoothing
                )
            else:
                self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=self.class_weights
            )

        elif loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([class_weights[1]/class_weights[0]]) if class_weights else None
            )

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, logits, labels):
        """
        Loss 계산

        Args:
            logits (torch.Tensor): 모델 출력 logits [batch_size, num_classes]
            labels (torch.Tensor): Ground truth labels [batch_size]

        Returns:
            torch.Tensor: Loss 값
        """
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)

        if self.loss_type == 'bce':
            # BCE는 binary logits 필요
            return self.loss_fn(logits[:, 1], labels.float())

        return self.loss_fn(logits, labels)


class FocalLoss(nn.Module):
    """
    Focal Loss for Imbalanced Classification

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Helps focus on hard examples and handles class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        """
        Args:
            alpha (float): Balancing factor for positive/negative examples
            gamma (float): Focusing parameter (higher = more focus on hard examples)
            class_weights (torch.Tensor): Optional class weights
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass

        Args:
            inputs (torch.Tensor): Model predictions [batch_size, num_classes]
            targets (torch.Tensor): Ground truth labels [batch_size]

        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_weight = alpha_t * focal_weight

        # Class weights
        if self.class_weights is not None:
            class_weights = self.class_weights.to(inputs.device)
            weight_t = class_weights[targets]
            focal_weight = weight_t * focal_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedDetectionLoss(nn.Module):
    """
    여러 Loss를 조합한 Combined Loss

    Classification Loss + 추가 Regularization Loss 조합 가능
    """

    def __init__(self,
                 cls_loss_type='cross_entropy',
                 cls_weight=1.0,
                 feature_reg_weight=0.0,
                 **kwargs):
        """
        Args:
            cls_loss_type (str): Classification loss 타입
            cls_weight (float): Classification loss 가중치
            feature_reg_weight (float): Feature regularization 가중치
        """
        super().__init__()

        self.cls_weight = cls_weight
        self.feature_reg_weight = feature_reg_weight

        self.cls_loss = DetectionLoss(loss_type=cls_loss_type, **kwargs)

    def forward(self, pred_dict, data_dict):
        """
        Combined Loss 계산

        Args:
            pred_dict (dict): 모델 예측 결과
            data_dict (dict): 입력 데이터

        Returns:
            dict: Loss 딕셔너리
        """
        losses = {}

        # Classification Loss
        cls_logits = pred_dict['cls']
        labels = data_dict['label']
        cls_loss = self.cls_loss(cls_logits, labels)
        losses['cls'] = cls_loss

        # Feature Regularization (optional)
        if self.feature_reg_weight > 0 and 'additive_features' in pred_dict:
            additive_features = pred_dict['additive_features']
            # L2 regularization on additive features
            feature_reg = torch.mean(additive_features ** 2)
            losses['feature_reg'] = feature_reg
        else:
            feature_reg = 0.0

        # Total Loss
        total_loss = self.cls_weight * cls_loss + self.feature_reg_weight * feature_reg
        losses['overall'] = total_loss

        return losses


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing CrossEntropy Loss

    Regularization technique that prevents overconfident predictions
    """

    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        """
        Args:
            smoothing (float): Smoothing factor (0 = no smoothing)
            weight (torch.Tensor): Class weights
            reduction (str): Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass

        Args:
            inputs (torch.Tensor): Model logits [batch_size, num_classes]
            targets (torch.Tensor): Ground truth labels [batch_size]

        Returns:
            torch.Tensor: Label smoothed loss
        """
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)

        # Smooth labels
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        # Loss calculation
        loss = -(targets_smooth * log_preds).sum(dim=-1)

        # Class weights
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            weight_t = weight[targets]
            loss = loss * weight_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    # 테스트 코드
    print("Detection Loss Test")

    # 더미 데이터
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))

    # CrossEntropy Loss
    ce_loss = DetectionLoss(loss_type='cross_entropy')
    ce_value = ce_loss(logits, labels)
    print(f"CrossEntropy Loss: {ce_value.item():.4f}")

    # Focal Loss
    focal_loss = DetectionLoss(loss_type='focal', focal_alpha=0.25, focal_gamma=2.0)
    focal_value = focal_loss(logits, labels)
    print(f"Focal Loss: {focal_value.item():.4f}")

    # Label Smoothing
    smooth_loss = DetectionLoss(loss_type='cross_entropy', label_smoothing=0.1)
    smooth_value = smooth_loss(logits, labels)
    print(f"Label Smoothing Loss: {smooth_value.item():.4f}")

    # Combined Loss
    combined_loss = CombinedDetectionLoss(cls_loss_type='focal', feature_reg_weight=0.01)
    pred_dict = {'cls': logits, 'additive_features': torch.randn(8, 768)}
    data_dict = {'label': labels}
    combined_value = combined_loss(pred_dict, data_dict)
    print(f"Combined Loss: {combined_value['overall'].item():.4f}")
