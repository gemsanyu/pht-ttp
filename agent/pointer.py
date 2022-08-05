from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from agent.attention import Attention


class Pointer(nn.Module):
    def __init__(
            self,
            num_neurons: int,
            num_layers: int,
            device: T.device,
            dropout: float = 0.2,
            n_glimpses: int = 1
        ) -> None:
        """
        ### Pointer class.
        -----
        Pointer to output probability
        and also the last hidden feature from the RNN (GRU).
        Assuming the input to the GRU is the HIDDEN FEATURE (ENCODED) state of
        the previously selected node. In this case ENCODE(coords):
        3*num_neurons = 1 from context, 1 from static features, 1 from dynamic features

        Parameter:
            num_neurons: total neurons
            num_layers: total layers
            device: device to be used by torch
            dropout: dropout
        """
        super(Pointer, self).__init__()

        self.device = device
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.n_glimpses = n_glimpses


        if num_layers == 1:
            dropout = 0

        self.gru = nn.GRU(
                num_neurons,
                num_neurons,
                num_layers,
                batch_first=True,
                dropout=dropout
            )
        self.glimpse = Attention(num_neurons, device=device)
        self.attention_layer = Attention(num_neurons, device=device, use_tanh=True, tanh_clip=10)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

        self.to(device)

    def forward(self,
                features: T.Tensor,
                decoder_input: T.Tensor,
                last_pointer_hidden_state: T.Tensor,
                mask: T.Tensor,
            ) -> Tuple[T.Tensor, T.Tensor]:
        '''
        ### Calculate pointer.
        -----

        Parameter:
            features: features of problem
            decoder_input: input for decoder
            last_pointer_hidden_state: last hidden state of pointer

        Return: - energy with shape ([1, num_nodes])
                - pointer's hidden state with shape ([num_layers, 1, num_neurons])
        '''
        # Calculate RNN
        rnn_out, pointer_hidden_state = self.gru(
                decoder_input,
                last_pointer_hidden_state
            )
        rnn_out = self.drop_rnn(rnn_out)
        # Now, we will use the output of the RNN to compute the attention.
        # The attentions now will be used to get the context feature
        # (weighted sum of features).

        # compute glimpse first, but what is glimpse really?
        # its like there are multiple layers of attention
        q = rnn_out
        for i in range(self.n_glimpses):
            embedded_features, glimpse_logits = self.glimpse(query=q, features=features) #1*n
            # mask the logit
            masked_glimpse_logits = glimpse_logits + mask.float().log()
            glimpse_att =  F.softmax(masked_glimpse_logits, dim=2)
            q = glimpse_att@embedded_features   
        _, logits = self.attention_layer(query=q, features=features)
        masked_logits = logits + mask.float().log()
        return masked_logits, pointer_hidden_state
