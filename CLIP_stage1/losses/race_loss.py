"""
Race Classification Loss
4-class CrossEntropy Loss for race prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaceLoss(nn.Module):
    """
    Race Classification Loss

    4-class CrossEntropy Loss (PG-FDD 기준 순서):
        - Class 0: Asian
        - Class 1: White
        - Class 2: Black
        - Class 3: Other
    """

    def __init__(self, weight=None, label_smoothing=0.0):
        """
        Args:
            weight (torch.Tensor, optional): 클래스별 가중치 (불균형 데이터 대응)
            label_smoothing (float): Label smoothing 값
        """
        super().__init__()

        self.weight = weight
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing
        )

    def forward(self, race_logits, race_labels):
        """
        Args:
            race_logits (torch.Tensor): Race prediction logits [batch_size, 4]
            race_labels (torch.Tensor): Race ground truth labels [batch_size]

        Returns:
            torch.Tensor: Race classification loss (scalar)
        """
        return self.criterion(race_logits, race_labels)

    def get_accuracy(self, race_logits, race_labels):
        """
        Race classification accuracy 계산

        Args:
            race_logits (torch.Tensor): Race prediction logits [batch_size, 4]
            race_labels (torch.Tensor): Race ground truth labels [batch_size]

        Returns:
            float: Accuracy
        """
        preds = torch.argmax(race_logits, dim=1)
        correct = (preds == race_labels).float().sum()
        total = race_labels.size(0)
        return (correct / total).item()

    def get_per_class_accuracy(self, race_logits, race_labels):
        """
        클래스별 accuracy 계산

        Args:
            race_logits (torch.Tensor): Race prediction logits [batch_size, 4]
            race_labels (torch.Tensor): Race ground truth labels [batch_size]

        Returns:
            dict: 클래스별 accuracy
        """
        race_names = ['Asian', 'White', 'Black', 'Other']
        preds = torch.argmax(race_logits, dim=1)

        per_class_acc = {}
        for i, name in enumerate(race_names):
            mask = (race_labels == i)
            if mask.sum() > 0:
                correct = (preds[mask] == race_labels[mask]).float().sum()
                per_class_acc[name] = (correct / mask.sum()).item()
            else:
                per_class_acc[name] = 0.0

        return per_class_acc
