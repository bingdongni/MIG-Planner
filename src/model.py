import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class MaskedGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=4):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_proj = nn.Linear(edge_dim, out_channels)
        self.att = nn.Parameter(torch.Tensor(1, out_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # 融入边特征并进行物理掩码 (Masked Softmax)
        e_feat = self.edge_proj(edge_attr)
        alpha = (x_i + x_j + e_feat) * self.att

        oneway_mask = edge_attr[:, 1] == 1
        alpha = alpha.sum(dim=-1)
        alpha[oneway_mask] -= 1e9

        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)


class MIGPlannerV3(nn.Module):
    def __init__(self, n_feats=2, hidden=128):
        super().__init__()
        self.conv1 = MaskedGATLayer(n_feats, hidden)
        self.conv2 = MaskedGATLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        return self.head(x)