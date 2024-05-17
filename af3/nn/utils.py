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
import numpy as np
import scipy
import contextlib
from typing import *
from scipy.spatial.transform import Rotation as R


def constant_string_hash(text:str):
    hash=0
    for ch in text:
        hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
    return hash


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64
    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
    if key is not None:
        seed = int(hash((seed, constant_string_hash(key))) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def uniform_random_rotation(
    size: int,
    dtype=torch.float,
    device="cpu",
    seed: Optional[int] = None
) -> torch.Tensor:      # shape [n, 3, 3]
    with numpy_seed(seed, key="uniform_random_rotation"):
        rot = R.random(size)
    rotmats = R.as_matrix(rot)
    return torch.tensor(rotmats, dtype=dtype, device=device, requires_grad=False)


def gaussian_random_translation(
    size: int,
    scale: float = 1.,
    dtype=torch.float,
    device="cpu",
    seed: Optional[int] = None
) -> torch.Tensor:
    with numpy_seed(seed, key="gaussian_random_translation"):
        trans = np.random.randn(size, 3) * scale
    return torch.tensor(trans, dtype=dtype, device=device, requires_grad=False)
