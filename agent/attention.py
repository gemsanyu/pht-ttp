import math
from typing import Dict, Optional, Tuple

import torch as T
import torch.nn as nn
from torch.nn import functional as F

from agent.embedding import Embedder

class Attention(T.jit.ScriptModule):
    def __init__(self, num_neurons: int, device: T.device, use_tanh:bool=False, tanh_clip:Optional[int]=10):
        """
        ### Calculates attention over the input nodes given the current state.
        -----
        Assuming that the GRU decoder's out dim is also num_neurons
        3*neurons because the hidden state is concatenated with the feature.
        Assume that features are alread concatenation of static and dynamic_feature

        Parameter:
            num_neurons: total neurons
            device: device to be used by torch
        """
        super(Attention, self).__init__()

        self.device = device
        self.num_neurons = num_neurons

        # W processes features from static decoder elements
        v = T.zeros(size=(1, 1, num_neurons), dtype=T.float32, requires_grad=True)
        self.v = nn.Parameter(v)
        stdv = 1./math.sqrt(num_neurons)
        self.v.data.uniform_(-stdv , stdv)
        self.features_embedder = Embedder(2*num_neurons, num_neurons, device=device)
        self.query_embedder = Embedder(num_neurons, num_neurons, device=device)     
        self.tanh_clip = tanh_clip
        self.use_tanh = use_tanh
        self.to(device)

    @T.jit.script_method
    def forward(
            self,
            features: T.Tensor,
            query: T.Tensor,
            param_dict: Optional[Dict[str, T.Tensor]]=None,
        ) -> Tuple[T.Tensor, T.Tensor]:
        '''
        ### Calculate attentions' score.
        -----

        Parameter:
            features: features of the environment
            pointer_hidden_state: hidden state of the previous pointer

        Return: attentions' score with shape ([batch_size, num_items])
        '''
        batch_size, _, _ = features.shape
        if param_dict is None:
            projected_features = self.features_embedder(features)
            projected_query = self.query_embedder(query)
            v = self.v.expand(batch_size, 1, self.num_neurons)
        else:
            fe_weight, fe_bias, qe_weight, qe_bias = param_dict["fe_weight"],param_dict["fe_bias"],param_dict["qe_weight"],param_dict["qe_bias"]
            projected_features = F.linear(features, fe_weight, fe_bias)
            projected_query = F.linear(query, qe_weight, qe_bias)
            v = param_dict["v"].expand(batch_size, 1, self.num_neurons)
        hidden =(projected_features+projected_query).tanh()
        hidden = hidden.permute(0,2,1)
        u = T.bmm(v,hidden)
        if self.use_tanh:
            logits = self.tanh_clip*u.tanh()
        else:
            logits = u
        return projected_features, logits
