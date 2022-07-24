from dis import dis
import math
from typing import Optional
import torch

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = torch.device("cpu")

DYNAMIC_FEATURE_SIZE = 3

class Agent(torch.jit.ScriptModule):
# class Agent(torch.nn.Module):
    def __init__(self,
                 n_heads: int,
                 n_gae_layers: int,
                 input_dim: int,
                 embed_dim: int,
                 gae_ff_hidden: int,
                 tanh_clip: float,
                 device=CPU_DEVICE):
        super(Agent, self).__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads
        self.norm_factor = math.sqrt(embed_dim)
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=input_dim,
                                         feed_forward_hidden=gae_ff_hidden)
        
        # embedder for glimpse and logits
        self.project_embeddings = torch.nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.project_fixed_context = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        current_state_dim = embed_dim + DYNAMIC_FEATURE_SIZE
        self.project_current_state = torch.nn.Linear(current_state_dim, embed_dim, bias=False)
        self.project_out = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.to(self.device)

    # num_step = 1
    @torch.jit.script_method    
    def forward(self, 
                item_embeddings: torch.Tensor,
                graph_embeddings: torch.Tensor,
                prev_item_embeddings: torch.Tensor,
                dynamic_features: torch.Tensor,
                glimpse_V: torch.Tensor,
                glimpse_K: torch.Tensor,
                logit_K: torch.Tensor,
                eligibility_mask: torch.Tensor,
                ):
        
        batch_size = item_embeddings.shape[0]
        current_state = torch.cat((prev_item_embeddings, dynamic_features), dim=-1)
        projected_current_state = self.project_current_state(current_state)
        query = graph_embeddings + projected_current_state
        glimpse_Q = query.view(batch_size, self.n_heads, 1, self.key_size)
        glimpse_Q = glimpse_Q.permute(1,0,2,3)
        compatibility = glimpse_Q@glimpse_K.permute(0,1,3,2) # glimpse_K => n_heads, batch_size, num_items, embed_dim
        mask = eligibility_mask.unsqueeze(0).unsqueeze(2) # batch_size, num_items -> 1, bs, 1, ni : broadcastable
        compatibility = compatibility + mask.float().log()
        attention = torch.softmax(compatibility, dim=-1)
        heads = attention@glimpse_V
        # supaya n_heads jadi dim nomor -2
        concated_heads = heads.permute(1,2,0,3).contiguous()
        concated_heads = concated_heads.view(batch_size, 1, self.embed_dim)
        final_Q = self.project_out(concated_heads)
        logits = final_Q@logit_K.permute(0,2,1) #batch_size, num_items, embed_dim
        logits = torch.tanh(logits) * self.tanh_clip * self.norm_factor
        logits = logits.squeeze(1) + eligibility_mask.float().log()
        # sudah dapat logits, ini untuk probability seleksinya
        # hehe logits is unnormalized probs/weights
        probs = torch.softmax(logits, dim=-1)
        # if selected_idx is None:
            # only select and return item for unfinished instance 
        selected_idx, logp, entropy = self.select(probs)
        # else:
        #     logp = torch.gather(probs, dim=1, index=selected_idx)
        return selected_idx, logp, entropy

    @torch.jit.ignore
    def select(self, probs):
        '''
        ### Select next to be executed.
        -----
        Parameter:
            probs: probabilities of each operation

        Return: index of operations, log of probabilities
        '''
        if self.training:
            dist = torch.distributions.Categorical(probs)
            op = dist.sample()
            logprob = dist.log_prob(op)
            entropy = dist.entropy()
        else:
            prob, op = torch.max(probs, dim=1)
            logprob = torch.log(prob)
            entropy = -torch.sum(prob*logprob)
        return op, logprob, entropy
