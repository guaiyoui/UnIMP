import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
from utils import get_activation
from torch_scatter import scatter_mean
from models.set_transformer import AllSetTrans

class V2E_layer_set(nn.Module):
    
    def __init__(self, edge_in_channels, edge_out_channels, node_in_channels, activation):
        super(V2E_layer, self).__init__()

        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.node_in_channels = node_in_channels

        self.v2e_lin = nn.Linear(node_in_channels, edge_out_channels)
        self.update_lin = nn.Linear(edge_in_channels + edge_out_channels, edge_out_channels)

        self.v2e_activation = get_activation(activation)
        self.update_activation = get_activation(activation)

        self.v2e_trans = AllSetTrans(in_channels=edge_out_channels, head_num=8, out_channels=edge_out_channels)
        self.fuse = nn.Linear(edge_in_channels + edge_out_channels, edge_out_channels)

    def forward(self, hyperedge, hyper_node, ve_affiliation):
        
        edge_index = [[i, ve_affiliation[0][i]] for i in range(int(hyper_node.shape[0]))]
        edge_index = torch.Tensor(edge_index).T.long().to(hyper_node.device)

        # print(ve_affiliation)
        hyper_node = self.v2e_activation(self.v2e_lin(hyper_node))
        hyperedge_tem = F.relu(self.v2e_trans(hyper_node, edge_index))
        
        # Update hyperedge
        out = self.update_activation(self.update_lin(torch.cat([hyperedge_tem, hyperedge], dim=-1)))

        return out


class V2E_layer(nn.Module):
    
    def __init__(self, edge_in_channels, edge_out_channels, node_in_channels, activation):
        super(V2E_layer, self).__init__()

        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.node_in_channels = node_in_channels

        self.v2e_lin = nn.Linear(node_in_channels, edge_out_channels)
        self.update_lin = nn.Linear(edge_in_channels + edge_out_channels, edge_out_channels)

        self.v2e_activation = get_activation(activation)
        self.update_activation = get_activation(activation)

    def forward(self, hyperedge, hyper_node, ve_affiliation):

        num_hyperedges = hyperedge.size(0)
        
        # Hypernode to hyperedge
        node_info = self.v2e_activation(self.v2e_lin(hyper_node))
        out = scatter_mean(node_info, ve_affiliation[0], dim=0, dim_size=num_hyperedges)
        
        # Update hyperedge
        out = self.update_activation(self.update_lin(torch.cat([out, hyperedge], dim=-1)))
        
        out = F.normalize(out, p=2, dim=-1)

        return out
