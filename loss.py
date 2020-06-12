import torch
import torch.nn.functional as F


def adversarial_loss(pred, ans):
    return F.binary_cross_entropy_with_logits(pred, ans)


def mse_loss(pred, ans):
    return F.mse_loss(pred, ans)
