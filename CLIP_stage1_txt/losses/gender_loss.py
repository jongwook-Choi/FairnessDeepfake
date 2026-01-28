"""
Gender Classification Loss
2-class CrossEntropy Loss for gender prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenderLoss(nn.Module):
    """
    Gender Classification Loss

    2-class CrossEntropy Loss:
        - Class 0: Male
        - Class 1: Female
    """

    def __init__(self, weight=None, label_smoothing=0.0):
        """
        Args:
            weight (torch.Tensor, optional): 클래스별 가중치
            label_smoothing (float): Label smoothing 값
        """
        super().__init__()

        self.weight = weight
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing
        )

    def forward(self, gender_logits, gender_labels):
        """
        Args:
            gender_logits (torch.Tensor): Gender prediction logits [batch_size, 2]
            gender_labels (torch.Tensor): Gender ground truth labels [batch_size]

        Returns:
            torch.Tensor: Gender classification loss (scalar)
        """
        return self.criterion(gender_logits, gender_labels)

    def get_accuracy(self, gender_logits, gender_labels):
        """
        Gender classification accuracy 계산

        Args:
            gender_logits (torch.Tensor): Gender prediction logits [batch_size, 2]
            gender_labels (torch.Tensor): Gender ground truth labels [batch_size]

        Returns:
            float: Accuracy
        """
        preds = torch.argmax(gender_logits, dim=1)
        correct = (preds == gender_labels).float().sum()
        total = gender_labels.size(0)
        return (correct / total).item()

    def get_per_class_accuracy(self, gender_logits, gender_labels):
        """
        클래스별 accuracy 계산

        Args:
            gender_logits (torch.Tensor): Gender prediction logits [batch_size, 2]
            gender_labels (torch.Tensor): Gender ground truth labels [batch_size]

        Returns:
            dict: 클래스별 accuracy
        """
        gender_names = ['Male', 'Female']
        preds = torch.argmax(gender_logits, dim=1)

        per_class_acc = {}
        for i, name in enumerate(gender_names):
            mask = (gender_labels == i)
            if mask.sum() > 0:
                correct = (preds[mask] == gender_labels[mask]).float().sum()
                per_class_acc[name] = (correct / mask.sum()).item()
            else:
                per_class_acc[name] = 0.0

        return per_class_acc
