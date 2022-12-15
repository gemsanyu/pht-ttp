import torch
from torch.nn import ModuleList
from torch_geometric.nn import GATConv, Linear, BatchNorm

class GraphEncoder(torch.nn.Module):
    def __init__(self, 
                 n_heads=8,
                 embed_dim=128,
                 n_layers=3) -> None:
        super(GraphEncoder, self).__init__()
        self.init_embed = Linear(-1, embed_dim, bias=False)
        layers = [GATConv(embed_dim, embed_dim//n_heads, n_heads) for _ in range(n_layers)]
        norms = [BatchNorm(embed_dim) for _ in range(n_layers-1)]

        self.layers = ModuleList(layers)
        self.norms = ModuleList(norms)

    def forward(self, x, edge_index):
        x = self.init_embed(x)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x)
        x = self.layers[-1](x, edge_index)
        return x