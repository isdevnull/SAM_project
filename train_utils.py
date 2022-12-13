import torch
from torch.nn.modules.batchnorm import _BatchNorm


def top_n_accuracy(logits, ref_labels, n: int = 1):
    accuracy = 0
    N = len(ref_labels)

    # get top indices
    _, top_indices = torch.topk(logits, dim=-1, k=n)

    for i in range(N):
        if ref_labels[i].item() in set(top_indices[i].tolist()):
            accuracy += 1

    return accuracy / N


def disable_running_stats(model):
    """
    Disables running stats for BatchNorm
    Taken from https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py
    """

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    """
    Enables running stats for BatchNorm
    Taken from https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py
    """

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_weight_norm(model):
    total_weight_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_weight_norm += p.detach().data.norm(2).item() ** 2
    total_weight_norm = total_weight_norm**0.5
    return total_weight_norm


def get_grad_norm(model):
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm**0.5
    return total_grad_norm
