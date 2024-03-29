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

def load_encoder(device_str)->Encoder:
    device = torch.device(device_str)
    item_static_encoder = torch.jit.load("item_static_encoder", map_location=device)
    node_encoder = torch.jit.load("node_encoder", map_location=device)
    item_dynamic_encoder = torch.jit.load("item_dynamic_encoder", map_location=device)
    node_dynamic_encoder = torch.jit.load("node_dynamic_encoder", map_location=device)
    depot_init_embed = torch.load("depot_init_embed", map_location=device)
    inital_input = torch.load("inital_input", map_location=device)
    encoder = Encoder(device,
                      item_static_encoder,
                      node_encoder,
                      item_dynamic_encoder,
                      node_dynamic_encoder,
                      depot_init_embed,
                      inital_input)
    return encoder

def save_encoder(encoder:Encoder):
    torch.jit.save(encoder.item_static_encoder,"item_static_encoder")
    torch.jit.save(encoder.node_encoder,"node_encoder")
    torch.jit.save(encoder.item_dynamic_encoder,"item_dynamic_encoder")    
    torch.jit.save(encoder.node_dynamic_encoder,"node_dynamic_encoder")
    torch.save(encoder.depot_init_embed,"depot_init_embed")    
    torch.save(encoder.inital_input,"inital_input")
    
