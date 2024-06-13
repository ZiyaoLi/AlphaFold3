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
from .common import (
    Linear,
    Transition,
    one_hot,
)


# alg3 relpos
# it's seemingly better to do this on CPU.
# the parameters are removed from this module. 
# this current module calculates one-hot embeddings only.
class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max=32, s_max=2) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max

    @property
    def output_dim(self):
        # (2r+2) + (2r+2) + 1 + (2s+2)
        return 4*self.r_max + 2*self.s_max + 7

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        dtype = torch.float,
    ):
        asym_id = features["asym_id"]
        sym_id = features["sym_id"]
        entity_id = features["entity_id"]
        token_idx = features["token_index"]
        res_idx = features["residue_index"]

        def _pair_same(x):
            return x[..., :, None] == x[..., None, :]

        def _pair_diff(x):
            return x[..., :, None] - x[..., None, :]

        b_same_chain = _pair_same(asym_id)
        b_same_res = _pair_same(res_idx) & b_same_chain     # same res must be same chain
        b_same_entity = _pair_same(entity_id)

        d_res = torch.where(
            b_same_chain,
            torch.clip(_pair_diff(res_idx) + self.r_max, min=0, max=2*self.r_max),
            2*self.r_max+1,
        )
        rel_pos = one_hot(d_res, 2*self.r_max+2)

        d_token = torch.where(
            b_same_res,
            torch.clip(_pair_diff(token_idx) + self.r_max, min=0, max=2*self.r_max),
            2*self.r_max+1,
        )
        rel_token = one_hot(d_token, 2*self.r_max+2)

        d_chain = torch.where(
            ~b_same_chain,
            torch.clip(_pair_diff(sym_id) + self.s_max, min=0, max=2*self.s_max),
            2*self.s_max+1
        )
        rel_chain = one_hot(d_chain, 2*self.s_max+2)

        ret = torch.cat(
            [rel_pos, rel_token, b_same_entity.float()[..., None], rel_chain], dim=-1
        ).to(dtype)

        # assert ret.shape[-1] == self.output_dim
        return ret






# alg 8 MSA Module
class MSAModuleBlock(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_opm_hid: int = 32,
        d_paw_hid: int = 8,     # inconsistent in alg8 (8) and alg10 (32), using alg8.
        num_paw_heads: int = 8,
        d_tri_mul_hid: int = 128,       # alg 12 & 13
        d_tri_att_hid: int = 32,        # alg 14 & 15
        num_tri_att_heads: int = 4,
        n_transition: int = 4,
        dropout_msa: float = 0.15,
        dropout_pair: float = 0.25,
    ) -> None:
        super().__init__()
        self.opm = OuterProductMean(d_msa, d_pair, d_opm_hid)
        self.paw = PairAverageWeighting(d_msa, d_pair, d_paw_hid, num_paw_heads)
        self.tri_out = TriangleMultiplicationOutgoing(d_pair, d_tri_mul_hid)
        self.tri_in = TriangleMultiplicationIncoming(d_pair, d_tri_mul_hid)
        self.tri_att_start = TriangleAttentionStarting(d_pair, d_tri_att_hid)
        self.tri_att_end = TriangleAttentionEnding(d_pair, d_tri_att_hid, num_tri_att_heads)
        self.transition = Transition(d_pair, n_transition)
        self.dropout_msa = dropout_msa
        self.dropout_pair = dropout_pair

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

