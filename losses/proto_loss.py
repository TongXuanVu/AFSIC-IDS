import torch
import torch.nn.functional as F

def compute_proto_loss(features, targets, proto_matrix):
    """
    Prototype Alignment Loss
    Calculates MSE between normalized features and corresponding global prototypes.
    """
    z_norm = F.normalize(features, p=2, dim=1)
    loss_proto = F.mse_loss(z_norm, proto_matrix[targets], reduction="mean")
    return loss_proto
