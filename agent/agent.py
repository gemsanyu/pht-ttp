from typing import Dict, Tuple, Optional

from torch.nn.functional import softmax
import torch as T
import torch.nn as nn

from agent.embedding import Embedder
from agent.pointer import Pointer
from agent.attention_embedder import DecoderEmbedder

"""
Vehicle Features:
    Static: depotx, depoty,
    Dynamic: currentx, currenty, remaining_duration, remaining_load

Customer Features:
    Static: x, y, demand_size, service_duration, time_window_start, time_window_end,
            ###jarak kust ke depot terdekat, jarak kust ke depot terjauh, rerata jarak ke depot?
    Dynamic: jarak sekarang ke kendaraan yg sedang diconsider,
            ###jarak kust ke depot kendaraan yg diconsider

Encoder input: last customers static features
"""
CPU_DEVICE = T.device("cpu")

class Agent(T.jit.ScriptModule):
# class Agent(nn.Module):
    def __init__(
            self,
            device: CPU_DEVICE,
            num_static_features: int = 4,
            num_dynamic_features: int = 3,
            static_encoder_size: int = 64,
            dynamic_encoder_size: int = 64,
            decoder_encoder_size: int = 64,
            pointer_num_layers: int = 2,
            pointer_num_neurons: int = 64,
            dropout: float = 0.2,
            n_glimpses: int=1,
            is_high: bool = False
        ) -> None:
        '''
        ### Agent of the architecture.
        -----

        Parameter:
            cust_num_static_features: total static features for customers
            cust_num_dynamic_features: total dynamic features for customers
            vehicle_num_static_features: total static features for vehicle
            vehicle_num_dynamic_features: total dynamic features for vehicle
            static_encoder_sizes: layer size for static encoder
            dynamic_encoder_sizes: layer size for dynamic encoder
            pointer_num_layers: total layer for pointer
            pointer_num_neurons: pointer layer size
        '''
        super(Agent, self).__init__()
        self.device = device
        self.n_glimpses = n_glimpses
        self.embedding_size = static_encoder_size
        self.num_static_features = num_static_features
        self.num_dynamic_features = num_dynamic_features

        self.static_encoder = Embedder(self.num_static_features, static_encoder_size, device=self.device)
        self.dynamic_encoder = Embedder(self.num_dynamic_features, dynamic_encoder_size, device=self.device)
        self.total_num_features = self.num_static_features + self.num_dynamic_features
        self.decoder_input_encoder = Embedder(static_encoder_size, decoder_encoder_size, device=device)
        # self.decoder_input_encoder = DecoderEmbedder(self.num_dynamic_features, self.num_static_features, self.embedding_size, device=device)
        self.pointer = Pointer(pointer_num_neurons, pointer_num_layers, device=self.device, dropout=dropout, n_glimpses=n_glimpses)
        initial_input = T.randn(size=(1,1,static_encoder_size), dtype=T.float32, device=self.device)
        self.inital_input = nn.parameter.Parameter(initial_input)
        self.softmax = nn.Softmax(dim=2)
        self.to(self.device)

    @T.jit.script_method   
    # @T.jit.ignore
    def forward(self, 
                last_pointer_hidden_states: T.Tensor, 
                static_embeddings: T.Tensor, 
                dynamic_embeddings: T.Tensor,
                eligibility_mask: T.Tensor,
                previous_embeddings: T.Tensor,
                param_dict: Optional[Dict[str, T.Tensor]]=None) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        '''
        ### get probs and selection

        Parameter:
            pointer_hidden_state, raw_features, and eligibility_mask

        Return: logprobs, selected_vecs, and selected_custs
        '''
        batch_size, num_items, _ = static_embeddings.shape
        eligibility_mask = eligibility_mask.view(batch_size, 1, -1)
        decoder_input = self.decoder_input_encoder(previous_embeddings)
        # dynamic_embeddings = dynamic_embeddings.unsqueeze(1)
        # dynamic_embeddings = dynamic_embeddings.repeat_interleave(num_items, dim=1)
        features = T.cat((static_embeddings, dynamic_embeddings), dim=-1)
        # features = features.view(batch_size, num_vec*num_cust, 2*self.embedding_size)

        logits, next_pointer_hidden_state = self.pointer(features, decoder_input, last_pointer_hidden_states, eligibility_mask, param_dict)
        probs = self.softmax(logits)
        return next_pointer_hidden_state, logits, probs

    @T.jit.ignore
    def select(self, probs: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        '''
        ### Select next operation to be executed.
        -----
        operation is pair of vec x cust
        Parameter:
            probs: probabilities of each operation

        Return: index of operations, log of probabilities
        '''
        if self.training:
            dist = T.distributions.Categorical(probs)
            selected_idx = dist.sample()
            logprob = dist.log_prob(selected_idx)
            entropy = dist.entropy()
            entropy = entropy.squeeze(1)        
        else:
            prob, selected_idx = T.max(probs, dim=2)
            logprob = T.log(prob)
            # no_probs = probs == 0
            # probs[no_probs] = 1
        # entropy = (-probs*T.log(probs)).sum(dim=2)
        entropy = 0
        selected_idx = selected_idx.squeeze(1)
        logprob = logprob.squeeze(1)
        return selected_idx, logprob, entropy
