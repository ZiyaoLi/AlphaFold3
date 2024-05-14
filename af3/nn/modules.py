# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
from typing import *
from .old.common import Linear


# alg 8 MSA Module
class MSAModuleBlock(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_opm_hid: int = 32,
        d_paw_hid: int = 8,     # inconsistent in alg8 (8) and alg10 (32), using alg8.
        num_paw_heads: int = 8,
        n_transition: int = 4,
        dropout_msa: float = 0.15,
        dropout_pair: float = 0.25,
    ) -> None:
        super().__init__()
        self.opm = OuterProductMean(d_msa, d_pair, d_opm_hid)
        self.paw = PairAverageWeighting(d_msa, d_pair, d_paw_hid, num_paw_heads)

    def forward(self):
        pass

# alg 9 outer prod mean
from .old.common import OuterProductMean

# alg 10
class PairAverageWeighting(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hid: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_hid = d_hid
        d_hid_all = num_heads * d_hid
        self.m_layer_norm = nn.LayerNorm(d_msa)
        self.m_v_linear = Linear(d_msa, d_hid_all, bias=False)
        self.z_layer_norm = nn.LayerNorm(d_pair)
        self.z_b_linear = Linear(d_pair, num_heads)
        self.m_g_linear = Linear(d_msa, d_hid_all, bias=False)
        self.o_linear = Linear(d_hid_all, d_msa, bias=False, init="final")

    def forward(self, m, z):
        m = self.m_layer_norm(m)
        v = self.m_v_linear(m).view(*m.shape[:-1], self.d_hid, self.num_heads)      # *sidh
        b = self.z_b_linear(self.z_layer_norm(z))   
        g = torch.sigmoid(self.m_g_linear(m)).view(*m.shape[:-1], self.d_hid, self.num_heads)   # *sidh
        w = torch.softmax(b, dim=-2)    # *ijh
        o = g * torch.einsum("...ijh,...sjdh->...sidh", w, v)       # *sidh
        return self.o_linear(o.view(*o.shape[:-2], -1))

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

# alg 12 & 13 triangular updates
from functools import partialmethod

class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.outgoing = outgoing

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_in = nn.LayerNorm(d_pair)
        self.layer_norm_out = nn.LayerNorm(d_hid)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        z_ = self.layer_norm_in(z)
        ab = self.linear_ab_p(z_) * torch.sigmoid(self.linear_ab_g(z_))
        if mask is not None:
            ab *= mask[..., None]
        a, b = torch.chunk(ab, 2, dim=-1)
        del z_, ab

        if self.outgoing:
            x = torch.einsum("...ikd,...jkd->...ijd", a, b)
        else:
            x = torch.einsum("...kid,...kjd->...ijd", a, b)
        del a, b
        g = torch.sigmoid(self.linear_g(z))
        z = g * self.linear_z(self.layer_norm_out(x))

        return z

class TriangleMultiplicationOutgoing(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=False)


# alg 14 & 15 triangular attention
from .old.attentions import (
    TriangleAttentionStarting,
    TriangleAttentionEnding
)

# alg26 AdaLN

class AdaptiveLayerNorm(nn.Module):
    # TODO understand what is going on. impl. as-is.
    def __init__(self, d_a, d_s) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_a, elementwise_affine=False)
        self.norm_s = nn.LayerNorm(d_s)
        self.norm_s.bias.requires_grad_(False)
        self.w_s = Linear(d_s, d_a, init="gating")
        self.b_s = Linear(d_s, d_a, bias=False, init="final")

    def forward(self, a, s):
        a = self.norm_a(a)
        s = self.norm_s(s)
        a = torch.sigmoid(self.w_s(s)) * a + self.b_s(s)
