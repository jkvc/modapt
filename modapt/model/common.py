MULTICLASS_STRATEGY = ["multinomial", "ovr"]

import torch
import torch.nn.functional as F
from modapt.utils import AUTO_DEVICE


def calc_multiclass_loss(logits, labels, multiclass_strategy):
    if multiclass_strategy == "multinomial":
        loss = F.cross_entropy(logits, labels, reduction="none")
    elif multiclass_strategy == "ovr":
        # convert label to one-hot
        nsample, nclasses = logits.shape
        labels = torch.eye(nclasses).to(AUTO_DEVICE)[labels].to(torch.float)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = loss.mean(dim=-1)
    else:
        raise NotImplementedError()
    return loss, labels
