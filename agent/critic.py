from typing import List

import torch as T

from agent.embedding import Embedder

CPU_DEVICE = T.device("cpu")

class Critic(T.jit.ScriptModule):
# class Critic(T.nn.Module):
    def __init__(self, 
                 dynamic_feature_size: int=3,
                 static_feature_size: int=7,
                 embedding_size: int = 64,
                 device: T.device = CPU_DEVICE) -> None:
        super().__init__()
        self.static_encoder = Embedder(static_feature_size,embedding_size,device)
        self.dynamic_encoder = Embedder(dynamic_feature_size,embedding_size,device)
        self.value_layer = T.nn.Sequential(
                                Embedder(2*embedding_size,20,device),
                                T.nn.ReLU(),
                                Embedder(20,20,device),
                                T.nn.ReLU(),
                                Embedder(20,2,device),
                                T.nn.ReLU()               
                            )
        self.embedding_size = embedding_size
        self.device = device
        self.to(device)

    @T.jit.script_method
    def forward(self, raw_static_feature: T.Tensor, raw_dynamic_feature:T.Tensor):
        batch_size, num_items, _ = raw_static_feature.shape
        static_features = self.static_encoder(raw_static_feature)
        dynamic_features = self.dynamic_encoder(raw_dynamic_feature).unsqueeze(1)
        dynamic_features = dynamic_features.expand_as(static_features)
        features = T.cat((static_features, dynamic_features), dim=-1)
        features = features.view(batch_size, num_items, 2*self.embedding_size)
        value = self.value_layer(features).sum(1)
        return value