import math
from typing import Dict, Optional, Tuple

from torch.nn import Linear, Parameter
import torch.nn.functional as F
import torch

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = torch.device("cpu")

# @torch.jit.script
def get_glimpses(projected_item_state:torch.Tensor, projected_node_state:torch.Tensor):
    projected_item_node_state = torch.cat([projected_item_state, projected_node_state], dim=1)
    glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = projected_item_node_state.chunk(3, dim=-1)
    return glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic

# @torch.jit.script
def compute_lk_and_ch(glimpse_V_static: torch.Tensor,
                      glimpse_V_dynamic: torch.Tensor,
                      glimpse_K_static: torch.Tensor,
                      glimpse_K_dynamic: torch.Tensor,
                      logit_K_static: torch.Tensor,
                      logit_K_dynamic: torch.Tensor,
                      fixed_context: torch.Tensor,
                      projected_current_state: torch.Tensor,
                      eligibility_mask: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    n_heads, batch_size, _, key_size = glimpse_V_static.shape
    embed_dim = key_size*n_heads
    glimpse_V = glimpse_V_static + glimpse_V_dynamic
    glimpse_K = glimpse_K_static + glimpse_K_dynamic
    logit_K = logit_K_static + logit_K_dynamic
    query = fixed_context + projected_current_state
    glimpse_Q = query.view(batch_size, n_heads, 1, key_size)
    glimpse_Q = glimpse_Q.permute(1,0,2,3)
    compatibility = glimpse_Q@glimpse_K.permute(0,1,3,2) / math.sqrt(glimpse_Q.size(-1)) # glimpse_K => n_heads, batch_size, num_items, embed_dim
    mask = eligibility_mask.unsqueeze(0).unsqueeze(2) # batch_size, num_items -> 1, bs, 1, ni : broadcastable
    compatibility = compatibility + mask.float().log()
    attention = torch.softmax(compatibility, dim=-1)
    heads = attention@glimpse_V
    # supaya n_heads jadi dim nomor -2
    concated_heads = heads.permute(1,2,0,3).contiguous()
    concated_heads = concated_heads.view(batch_size, 1, embed_dim)
    return logit_K, concated_heads

# @torch.jit.script
def get_probs(final_Q:torch.Tensor, logit_K:torch.Tensor, eligibility_mask:torch.Tensor):
    logits = final_Q@logit_K.permute(0,2,1) / math.sqrt(final_Q.size(-1)) #batch_size, num_items, embed_dim
    logits = torch.tanh(logits) * 10
    logits = logits.squeeze(1) + eligibility_mask.float().log()
    probs = torch.softmax(logits, dim=-1)
    return probs

# class Agent(torch.jit.ScriptModule):
class Agent(torch.nn.Module):
    def __init__(self,
                 num_static_features: int,
                 num_dynamic_features: int,
                 n_heads: int,
                 n_gae_layers: int,
                 embed_dim: int,
                 gae_ff_hidden: int,
                 tanh_clip: float,
                 device=CPU_DEVICE):
        super(Agent, self).__init__()
        self.num_static_features = num_static_features
        self.num_node_dynamic_features = num_dynamic_features-2
        self.num_global_dynamic_features = 2
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=None,
                                         feed_forward_hidden=gae_ff_hidden)
        
        # embedder for glimpse and logits
        self.item_init_embedder = Linear(3, embed_dim)
        self.depot_init_embed = Parameter(torch.Tensor(size=(1,1,embed_dim)))
        self.depot_init_embed.data.uniform_(-1, 1)
        self.node_init_embed = Parameter(torch.Tensor(size=(1,1,embed_dim)))
        self.node_init_embed.data.uniform_(-1, 1)
        self.project_embeddings = Linear(embed_dim, 3*embed_dim, bias=False)
        self.project_fixed_context = Linear(embed_dim, embed_dim, bias=False)
        current_state_dim = embed_dim + self.num_global_dynamic_features
        self.project_current_state = Linear(current_state_dim, embed_dim, bias=False)
        self.project_node_state = Linear(self.num_node_dynamic_features, 3*embed_dim, bias=False)
        self.project_out = Linear(embed_dim, embed_dim, bias=False)
        self.compute_lk_and_ch = torch.jit.script(compute_lk_and_ch)
        self.get_glimpses = torch.jit.script(get_glimpses)
        self.get_probs = torch.jit.script(get_probs)
        self.to(self.device)

    # num_step = 1
    # @torch.jit.script_method    
    def forward(self, 
                item_embeddings: torch.Tensor,
                fixed_context: torch.Tensor,
                prev_item_embeddings: torch.Tensor,
                node_dynamic_features: torch.Tensor,
                global_dynamic_features: torch.Tensor,
                glimpse_V_static: torch.Tensor,
                glimpse_K_static: torch.Tensor,
                logit_K_static: torch.Tensor,
                eligibility_mask: torch.Tensor,
                param_dict: Dict[str, torch.Tensor]=None):
        batch_size = item_embeddings.shape[0]
        current_state = torch.cat((prev_item_embeddings, global_dynamic_features), dim=-1)
        projected_current_state = self.project_current_state(current_state)
        glimpse_V_dynamic, glimpse_K_dynamic, logit_K_dynamic = self.project_node_state(node_dynamic_features).chunk(3, dim=-1)
        glimpse_V_dynamic = self._make_heads(glimpse_V_dynamic)
        glimpse_K_dynamic = self._make_heads(glimpse_K_dynamic)
        logit_K, concated_heads = self.compute_lk_and_ch(glimpse_V_static,
                                                    glimpse_V_dynamic,
                                                    glimpse_K_static,
                                                    glimpse_K_dynamic,
                                                    logit_K_static,
                                                    logit_K_dynamic,
                                                    fixed_context,
                                                    projected_current_state,
                                                    eligibility_mask)
        final_Q = self.project_out(concated_heads)
        probs = self.get_probs(final_Q, logit_K, eligibility_mask)
        selected_idx, logp, entropy = self.select(probs)
        return selected_idx, logp, entropy

    # @torch.jit.script_method
    def _make_heads(self, x: torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(2).view(x.size(0), x.size(1), self.n_heads, self.key_size)
        x = x.permute(2,0,1,3)
        return x
    

    # @torch.jit.ignore
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
