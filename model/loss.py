import torch.nn as nn
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target, weight=None):
    return F.binary_cross_entropy(input=output, target=target,
                                  weight=weight)