import torch
import torch.nn.functional as F

def compute_fsp_loss(features, targets, proto_matrix, T_fsp=0.5):
    """
    Few-Shot Sparse Pairwise Loss
    Calculates positive and negative pair distances against global prototypes.
    """
    z_norm = F.normalize(features, p=2, dim=1)
    sims = torch.mm(z_norm, proto_matrix.t())
    batch_size = z_norm.size(0)
    
    # Positive pairs
    s_pos = sims[torch.arange(batch_size), targets]
    
    # Negative pairs (exclude the true target)
    mask = torch.ones_like(sims).scatter_(1, targets.unsqueeze(1), 0.0)
    s_neg, _ = torch.max(sims * mask - (1.0 - mask) * 1e9, dim=1)
    
    loss_fsp = torch.mean(torch.log(1.0 + torch.exp((s_neg - s_pos) / T_fsp)))
    return loss_fsp
