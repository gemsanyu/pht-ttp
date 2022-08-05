from typing import List, Optional

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from agent.embedding import Embedder
from agent.attention import Attention

CPU_DEVICE = T.device("cpu")

class DecoderEmbedder(nn.Module):
    def __init__(
            self,
            dynamic_feature_size: int,
            static_feature_size: int,
            embedding_size: int = 64,
            device = CPU_DEVICE
        ) -> None:
        '''
        ### Embedder class.
        -----
        It uses MLP method.

        Parameter:
            input_size: size for input in int
            hidden_layer_sizes: size for layers in hidden layer
        '''
        super(DecoderEmbedder, self).__init__()
        self.dynamic_encoder = Embedder(dynamic_feature_size, embedding_size, use_relu=True)
        self.static_encoder = Embedder(static_feature_size, embedding_size, use_relu=True)
        self.embedding_size = embedding_size
        self.attention = Attention(self.embedding_size, device)
        self.out_encoder = Embedder(4*self.embedding_size,self.embedding_size, use_relu=True)
        self.device = device
        self.to(device)
        

    def forward(
            self,
            raw_static_features: T.Tensor,
            raw_dynamic_features: T.Tensor,
            last_pointer_hidden_state: Optional[T.Tensor] = None
            ):
        '''
        ### Calculate embedding for decoder.
        -----

        Parameter:
        

        Return: an embedding for decoder input
        '''
        batch_size, num_vehicle, num_cust, _ = raw_static_features.shape
        static_features = self.static_encoder(raw_static_features)
        static_features = static_features.view(batch_size, num_vehicle*num_cust, self.embedding_size)

        dynamic_features = self.dynamic_encoder(raw_dynamic_features)
        dynamic_features = dynamic_features.view(batch_size, num_vehicle*num_cust, self.embedding_size)
        features = T.cat((static_features, dynamic_features), dim=2)
        query = last_pointer_hidden_state
        if query is None:
            query = T.zeros((batch_size, 1, self.embedding_size), dtype=T.float32, device=self.device)

        # decoder_input = features.mean(dim=1, keepdim=True)
        embedded_features, logits = self.attention(query=query, features=features) #1*n
        attentions = F.softmax(logits, dim=2)
        contexts = T.bmm(attentions, features) #b*1*n,b*n*2h = 1*2h
        contexts = contexts.expand_as(features) #b*n*2h

        feature_contexts = T.cat((features, contexts), dim=2) #b*n*4h
        embeddings = self.out_encoder(feature_contexts)
        graph_embedding = embeddings.mean(dim=1)
        
        return embeddings, graph_embedding
