import torch
from torch import nn
from torch.nn import functional as F
from .old.common import Linear, SimpleModuleList

# alg4 one hot
def one_hot(x: torch.Tensor, num_classes: int, dtype=torch.float):
    ret = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    ret.scatter_(-1, x.long().unsqueeze(-1), 1)
    return ret

# alg4 one hot
def one_hot_continuous(x: torch.Tensor, bins: torch.Tensor, dtype=torch.float):
    assert len(bins.shape) == 1, f"bins must be 1D. got {bins.shape}"
    x_id = (x[..., None] > bins).sum(-1)
    return one_hot(x_id, len(bins)+1, dtype)

# alg 11 transition, different to af2
class Transition(nn.Module):
    def __init__(self, dim, n):
        super(Transition, self).__init__()
        hid_dim = dim * n

        self.layer_norm = nn.LayerNorm(dim)
        self.wa = Linear(dim, hid_dim, init="relu")
        self.act = F.silu   # aka swish
        self.wb = Linear(dim, hid_dim)
        self.wo = Linear(hid_dim, dim, init="final")

    def forward(self, x):
        x = self.layer_norm(x)
        return self.wo(self.act(self.wa(x)) * self.wb(x))