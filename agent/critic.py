import torch
from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = torch.device("cpu")

class Critic(torch.jit.ScriptModule):
# class Critic(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        device=CPU_DEVICE
    ):
        super(Critic, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = GraphAttentionEncoder(
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.to(self.device)

    @torch.jit.script_method
    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        _, graph_embeddings = self.encoder(inputs)
        return self.value_head(graph_embeddings).squeeze(1)
