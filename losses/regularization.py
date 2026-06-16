import torch

def compute_sparse_regularization(network, device):
    """
    L1 norm on adapter and gate parameters to promote sparsity.
    """
    loss_rs = torch.tensor(0.0).to(device)
    if hasattr(network, 'plasticity_adapter'):
        for p in network.plasticity_adapter.parameters():
            if p.requires_grad:
                loss_rs += torch.sum(torch.abs(p))
    if hasattr(network, 'gate'):
        for p in network.gate.parameters():
            if p.requires_grad:
                loss_rs += torch.sum(torch.abs(p))
    return loss_rs

def compute_fedprox_regularization(network, global_model_params_round_start, device):
    """
    FedProx-style L2 regularization to prevent client drift.
    """
    loss_prox = torch.tensor(0.0).to(device)
    for name, p in network.named_parameters():
        if p.requires_grad and name in global_model_params_round_start:
            p_old = global_model_params_round_start[name]
            loss_prox += torch.sum((p - p_old) ** 2)
    return loss_prox
