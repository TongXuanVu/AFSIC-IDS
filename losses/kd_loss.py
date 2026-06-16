import torch
import torch.nn.functional as F

def compute_kd_loss(new_logits, old_logits, T=2.0):
    """
    Knowledge Distillation Loss based on KL divergence.
    """
    p = F.log_softmax(new_logits / T, dim=1)
    q = F.softmax(old_logits / T, dim=1)
    loss_kd = F.kl_div(p, q, reduction="batchmean") * (T ** 2)
    return loss_kd
