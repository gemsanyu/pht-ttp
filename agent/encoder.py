import pathlib
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

def load_encoder(device, title, weight_idx, total_weight)->Encoder:
    traced_dir = pathlib.Path()/"traced_agent"
    traced_dir.mkdir(parents=True, exist_ok=True)
    prefix = title+"_"+str(weight_idx)+"_"+str(total_weight)+"_"
    item_static_encoder_path = (traced_dir/(prefix+"item_static_encoder")).absolute()
    node_encoder_path = (traced_dir/(prefix+"node_encoder")).absolute()
    item_dynamic_encoder_path = (traced_dir/(prefix+"item_dynamic_encoder")).absolute()
    node_dynamic_encoder_path = (traced_dir/(prefix+"node_dynamic_encoder")).absolute()
    depot_init_embed_path = (traced_dir/(prefix+"depot_init_embed")).absolute()
    inital_input_path = (traced_dir/(prefix+"inital_input")).absolute()
    
    item_static_encoder = torch.jit.load(item_static_encoder_path, map_location=device)
    node_encoder = torch.jit.load(node_encoder_path, map_location=device)
    item_dynamic_encoder = torch.jit.load(item_dynamic_encoder_path, map_location=device)
    node_dynamic_encoder = torch.jit.load(node_dynamic_encoder_path, map_location=device)
    depot_init_embed = torch.load(depot_init_embed_path, map_location=device)
    inital_input = torch.load(inital_input_path, map_location=device)
    encoder = Encoder(device,
                      item_static_encoder,
                      node_encoder,
                      item_dynamic_encoder,
                      node_dynamic_encoder,
                      depot_init_embed,
                      inital_input)
    return encoder

def save_encoder(encoder:Encoder, title, weight_idx, total_weight):
    traced_dir = pathlib.Path()/"traced_agent"
    traced_dir.mkdir(parents=True, exist_ok=True)
    prefix = title+"_"+str(weight_idx)+"_"+str(total_weight)+"_"
    item_static_encoder_path = (traced_dir/(prefix+"item_static_encoder")).absolute()
    node_encoder_path = (traced_dir/(prefix+"node_encoder")).absolute()
    item_dynamic_encoder_path = (traced_dir/(prefix+"item_dynamic_encoder")).absolute()
    node_dynamic_encoder_path = (traced_dir/(prefix+"node_dynamic_encoder")).absolute()
    depot_init_embed_path = (traced_dir/(prefix+"depot_init_embed")).absolute()
    inital_input_path = (traced_dir/(prefix+"inital_input")).absolute()
    torch.jit.save(encoder.item_static_encoder,item_static_encoder_path)
    torch.jit.save(encoder.node_encoder,node_encoder_path)
    torch.jit.save(encoder.item_dynamic_encoder,item_dynamic_encoder_path)    
    torch.jit.save(encoder.node_dynamic_encoder,node_dynamic_encoder_path)
    torch.save(encoder.depot_init_embed,depot_init_embed_path)    
    torch.save(encoder.inital_input,inital_input_path)
    
