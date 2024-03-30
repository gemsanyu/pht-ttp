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

def save_encoder(encoder:Encoder):
    torch.jit.save(encoder.gae,"gae")
    torch.jit.save(encoder.item_init_embedder,"item_init_embedder")
    torch.jit.save(encoder.node_init_embedder,"node_init_embedder")
    torch.jit.save(encoder.project_fixed_context,"project_fixed_context")
    torch.jit.save(encoder.project_embeddings,"project_embeddings")
    torch.save(encoder.depot_init_embed, "depot_init_embed")

def load_encoder(device_str, with_gae=True)->Encoder:
    device = torch.device(device_str)
    gae = None
    if with_gae:
        gae = torch.jit.load("gae", map_location=device)
    item_init_embedder = torch.jit.load("item_init_embedder", map_location=device)
    depot_init_embed = torch.load("depot_init_embed", map_location=device)
    node_init_embedder = torch.jit.load("node_init_embedder", map_location=device)
    project_fixed_context = torch.jit.load("project_fixed_context", map_location=device)
    project_embeddings = torch.jit.load("project_embeddings", map_location=device)
    encoder = Encoder(device,
                      gae,
                      item_init_embedder,
                      depot_init_embed,
                      node_init_embedder,
                      project_fixed_context,
                      project_embeddings)
    return encoder