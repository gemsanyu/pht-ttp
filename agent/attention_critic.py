from typing import List

import torch as T
import torch.nn as nn

CPU_DEVICE = T.device("cpu")

class Critic(T.jit.ScriptModule):
    def __init__(self, 
                 dynamic_feature_size: int,
                 static_feature_size: int,
                 embedding_size: int = 64,
                 device: T.device = CPU_DEVICE,
                 n_process_blocks:int=3
                ) -> None:
        super().__init__()
        self.n_process_blocks = n_process_blocks
        # self.static_embedder = 
        # self.process_blocks = 
        self.embedding_size = embedding_size
        self.value_predictor = T.nn.Sequential(T.nn.Linear(self.embedding_size, 20),
                                                T.nn.ReLU(),
                                                T.nn.Linear(20, 20),
                                                T.nn.ReLU(),
                                                T.nn.Linear(20, 1))    
        self.device = device
        self.to(device)

    @T.jit.script_method
    def forward(self, raw_static_feature, raw_dynamic_feature):
        # batch_size, num_vec, num_cust, _ = raw_static_feature.shape

        # _, graph_embedding = self.attention_embedder(raw_static_feature, raw_dynamic_feature)
        # value = self.value_predictor(graph_embedding)
        return 0