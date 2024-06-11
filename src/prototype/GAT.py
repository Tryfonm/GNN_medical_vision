import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def 
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)
        self.classifier = nn.Linear(out_dim * num_heads, in_dim)

    def forward(self, g, features):
        h = self.layer1(g, features).flatten(1)
        h = F.elu(h)
        h = self.layer2(g, h).mean(1)
        output = self.classifier(h)
        return F.log_softmax(output, dim=1)


def get_synthetic_dataset(dim_0, dim_1, dim_2, feature_size, show=False):
    num_nodes = dim_0 * dim_1 * dim_2
    p_edge = 0.1

    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.rand() < p_edge:
                src.append(i)
                dst.append(j)

    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)

    node_features = torch.randn(num_nodes, feature_size)
    g.ndata["feat"] = node_features

    if show:
        nx_g = g.to_networkx()
        pos = nx.spring_layout(nx_g)
        nx.draw(
            nx_g,
            pos,
            with_labels=True,
            node_size=300,
            node_color="skyblue",
            font_size=8,
        )
        plt.title("Graph Visualization")
        plt.show()

    return g


if __name__ == "__main__":
    dim_0 = 4  # 240
    dim_1 = 4  # 240
    dim_2 = 4  # 154
    feature_size = 8  # number of in_channels from encoder
    in_dim = dim_0 * dim_1 * dim_2

    g = get_synthetic_dataset(dim_0=4, dim_1=4, dim_2=4, feature_size=8)

    model = GAT(g, in_dim=feature_size, hidden_dim=32, out_dim=32, num_heads=1)
    output = model(g, g.ndata["feat"])

    assert output.shape == torch.Size([in_dim, feature_size])
