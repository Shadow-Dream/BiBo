import copy
from typing import Optional

import torch.nn as nn


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_clone(module):
    return copy.deepcopy(module)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_activation_fn(act_fn=None):
    if act_fn is None:
        return nn.Identity()
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
