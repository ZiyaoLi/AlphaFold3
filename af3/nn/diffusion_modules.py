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

# alg21 diffusion conditioning
class DiffusionConditioning(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        t,
        features: Dict[str, torch.Tensor],
        s_inputs,
        s,
        z,
        sigma_data,
    ):
        pass


# alg22 fourier emb
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, learnable: bool = True) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim), requires_grad=learnable)
        self.b = nn.Parameter(torch.empty(dim), requires_grad=learnable)
        nn.init.trunc_normal_(self.w)
        nn.init.trunc_normal_(self.b)

    def forward(self, t):
        return torch.cos(2 * torch.pi * t[..., None] * self.w + self.b)


# alg24 DiffusionAttention with pair bias and mask
class AttentionPairBias(nn.Module):
    def __init__(
        self,
        d_token: int,
        d_pair: int,
        d_s: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.d_hid = d_token / num_heads
        self.num_heads = num_heads
        self.d_hid_all = d_hid_all = num_heads * self.d_hid
        self.ada_ln = AdaptiveLayerNorm(d_token, d_s)
        self.norm_a = nn.LayerNorm(d_token)
        self.norm_z = nn.LayerNorm(d_pair)
        self.wq = Linear(d_token, d_hid_all)
        self.wk = Linear(d_token, d_hid_all, bias=False)
        self.wv = Linear(d_token, d_hid_all, bias=False)
        self.wb = Linear(d_pair, num_heads)
        self.wg = Linear(d_token, d_hid_all, bias=False)
        self.wo = Linear(d_hid_all, d_token, bias=False)
        self.ws = Linear(d_s, d_token, init="adaln_zero")
        self.denominator = self.d_hid ** -.5

    def forward(
        self,
        a,
        s,
        z,
        beta,
    ):
        if s is not None:
            a = self.ada_ln(a, s)
        else:
            a = self.norm_a(a)
        
        q = self.wq(a).view(*a.shape[:-1], self.num_heads, self.d_hid) * self.denominator  # *ihc
        k = self.wk(a).view(*a.shape[:-1], self.num_heads, self.d_hid)  # *ihc
        b = self.wb(z) + beta  # *ijh
        att = (q[..., :, None, :, :] * k[..., None, :, :, :]).sum(-1) + b   # *ijh
        att = att.softmax(dim=-2)
        del q, k, b

        v = self.wv(a).view(*a.shape[:-1], self.num_heads, self.d_hid)  # *jhc
        g = self.wg(a).sigmoid()    # *ihc
        o = g * (att[..., None] * v[..., None, :, :, :]).sum(-3)    # *ihc
        del v, g
        o = self.wo(o.view(*o.shape[:-2], self.d_hid_all))

        if s is not None:
            o = o * self.ws(s).sigmoid()
        return o


# alg25 conditional transition
class ConditionalTransitionBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        d_s: int,
        n: int = 2
    ) -> None:
        super().__init__()
        self.ada_ln = AdaptiveLayerNorm(d_token, d_s)
        self.w1 = Linear(d_token, n * d_token, bias=False)
        self.w2 = Linear(d_token, n * d_token, bias=False)
        self.w3 = Linear(n * d_token, d_token, bias=False)
        self.wg = Linear(d_s, d_token, init="adaln_zero")

    def forward(self, a, s):
        a = self.ada_ln(a, s)
        b = F.silu(self.w1(a)) * self.w2(a)
        o = torch.sigmoid(self.wg(s)) * self.w3(b)
        return o


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
        return a

