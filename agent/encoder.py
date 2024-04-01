import pathlib
from typing import NamedTuple

from agent.graph_encoder import GraphAttentionEncoder

import torch
from torch.nn import Linear

class Encoder(NamedTuple):
    device: torch.device
    gae: GraphAttentionEncoder
    item_init_embedder: Linear
    depot_init_embed: torch.Tensor
    node_init_embed: torch.Tensor
    project_fixed_context: Linear
    project_embeddings: Linear

def save_encoder(encoder:Encoder, title, weight_idx, total_weight):
    traced_dir = pathlib.Path()/"traced_agent"
    prefix = title+"_"+str(weight_idx)+"_"+str(total_weight)+"_"
    gae_path = (traced_dir/(prefix+"gae")).absolute()
    item_init_embedder_path     = (traced_dir/(prefix+"item_init_embedder")).absolute()
    node_init_embed_path     = (traced_dir/(prefix+"node_init_embedder")).absolute()
    project_fixed_context_path  = (traced_dir/(prefix+"project_fixed_context")).absolute()
    project_embeddings_path     = (traced_dir/(prefix+"project_embeddings")).absolute()
    depot_init_embed_path       = (traced_dir/(prefix+"depot_init_embed")).absolute()
    torch.jit.save(encoder.gae,gae_path)
    torch.jit.save(encoder.item_init_embedder,      item_init_embedder_path)
    torch.save(encoder.node_init_embed,      node_init_embed_path)
    torch.jit.save(encoder.project_fixed_context,   project_fixed_context_path)
    torch.jit.save(encoder.project_embeddings,      project_embeddings_path)
    torch.save(encoder.depot_init_embed,            depot_init_embed_path)

def load_encoder(device, title, weight_idx, total_weight)->Encoder:
    # device = torch.device(device_str)
    traced_dir = pathlib.Path()/"traced_agent"
    prefix = title+"_"+str(weight_idx)+"_"+str(total_weight)+"_"
    gae_path = (traced_dir/(prefix+"gae")).absolute()
    item_init_embedder_path     = (traced_dir/(prefix+"item_init_embedder")).absolute()
    node_init_embed_path     = (traced_dir/(prefix+"node_init_embedder")).absolute()
    project_fixed_context_path  = (traced_dir/(prefix+"project_fixed_context")).absolute()
    project_embeddings_path     = (traced_dir/(prefix+"project_embeddings")).absolute()
    depot_init_embed_path       = (traced_dir/(prefix+"depot_init_embed")).absolute()
    
    gae = torch.jit.load(gae_path, map_location=device)
    item_init_embedder    = torch.jit.load(item_init_embedder_path, map_location=device)
    depot_init_embed      = torch.load(depot_init_embed_path, map_location=device)
    node_init_embed    = torch.load(node_init_embed_path, map_location=device)
    project_fixed_context = torch.jit.load(project_fixed_context_path, map_location=device)
    project_embeddings    = torch.jit.load(project_embeddings_path, map_location=device)
    encoder = Encoder(device,
                      gae,
                      item_init_embedder,
                      depot_init_embed,
                      node_init_embed,
                      project_fixed_context,
                      project_embeddings)
    return encoder