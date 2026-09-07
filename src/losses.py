import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMultiClass(nn.Module):
    """Implementa a Focal Loss multi-classe balanceada conforme os slides 74 a 79 da aula."""
    def __init__(self, weight=None, gamma=0.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        focal_weight = (1.0 - p) ** self.gamma
        loss = -focal_weight * log_p * targets_one_hot

        if self.weight is not None:
            w = self.weight.to(logits.device).view(1, num_classes, 1, 1)
            loss = loss * w

        if self.reduction == 'mean':
            return loss.sum() / (targets.numel() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
