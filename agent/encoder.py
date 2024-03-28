from typing import NamedTuple

from agent.graph_encoder import GraphAttentionEncoder

import torch
from torch.nn import Linear

class Encoder(NamedTuple):
    device: torch.device
    gae: GraphAttentionEncoder
    item_init_embedder: Linear
    depot_init_embed: torch.Tensor
    node_init_embedder: Linear
    project_fixed_context: Linear
    project_embeddings: Linear