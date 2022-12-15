import torch
from torch.nn import ModuleList
from torch_geometric.nn import GATConv, Linear, BatchNorm, Sequential

# class GraphEncoder(torch.jit.ScriptModule):
class GraphEncoder(torch.nn.Module):
    def __init__(self, 
                 n_heads=8,
                 embed_dim=128) -> None:
        super(GraphEncoder, self).__init__()
        self.init_embed = Linear(-1, embed_dim, bias=False)
        # self.layers = Sequential('x, edge_index', [
        #     (GaeLayer(n_heads, embed_dim), 'x, edge_index -> x')
        #     for _ in range(n_layers)
        # ])
        self.layer1 = GATConv(embed_dim, embed_dim//n_heads, n_heads).jittable()
        self.norm1 = BatchNorm(embed_dim)
        self.layer2 = GATConv(embed_dim, embed_dim//n_heads, n_heads).jittable()
        self.norm2 = BatchNorm(embed_dim)
        self.layer3 = GATConv(embed_dim, embed_dim//n_heads, n_heads).jittable()
        self.norm3 = BatchNorm(embed_dim)

    def forward(self, x, edge_index):
        x = self.init_embed(x)
        x = x + self.layer1(x, edge_index)
        x = self.norm1(x)
        x = x + self.layer2(x, edge_index)
        x = self.norm2(x)
        x = x + self.layer3(x, edge_index)
        x = self.norm3(x)
        return x