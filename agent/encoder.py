from typing import NamedTuple

import torch
from torch.nn import Linear

class Encoder(NamedTuple):
    device: torch.device
    item_static_encoder: Linear
    node_encoder: Linear
    item_dynamic_encoder: Linear
    node_dynamic_encoder: Linear
    depot_init_embed: torch.Tensor
    inital_input: torch.Tensor
