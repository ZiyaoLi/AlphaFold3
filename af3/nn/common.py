import torch
from .old.common import Linear

def one_hot(x: torch.Tensor, num_classes: int, dtype=torch.float):
    ret = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    ret.scatter_(-1, x.long().unsqueeze(-1), 1)
    return ret

def one_hot_continuous(x: torch.Tensor, bins: torch.Tensor, dtype=torch.float):
    assert len(bins.shape) == 1, f"bins must be 1D. got {bins.shape}"
    x_id = (x[..., None] > bins).sum(-1)
    return one_hot(x_id, len(bins)+1, dtype)
