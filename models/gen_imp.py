import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from utils import get_activation
from torch_geometric.nn import GCNConv
from models.v2e_layer import V2E_layer
from models.e2v_layer import E2V_layer
import time

def get_gen_imp(hyperedge_dim, hyper_node_dim, args):
    
    # build model
    model = Gen_IMP(hyperedge_dim, hyper_node_dim,
                        args.hyperedge_dim_hidden, args.hyper_node_dim_hidden,
                        args.gnn_layer_num, args.dropout, args.gnn_activation)
    return model

class Gen_IMP(torch.nn.Module):
    def __init__(self, 
                hyperedge_dim, hyper_node_dim,
                hyperedge_dim_hidden, hyper_node_dim_hidden,
                gnn_layer_num, dropout, activation
                ):
        super(Gen_IMP, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.gnn_layer_num = gnn_layer_num
        
        self.v2e_layers = self.build_v2e(hyperedge_dim, hyper_node_dim,
                                    hyperedge_dim_hidden, hyper_node_dim_hidden,
                                    activation)

        self.e2v_layers = self.build_e2v(hyperedge_dim_hidden, hyper_node_dim, 
                                    hyper_node_dim_hidden, activation)
        
        self.proj_layer = nn.Linear(hyper_node_dim_hidden, 1)

    def build_v2e(self, hyperedge_dim, hyper_node_dim,
                     hyperedge_dim_hidden, hyper_node_dim_hidden,activation):
        v2e_layers = nn.ModuleList()
        layer = V2E_layer(hyperedge_dim, hyperedge_dim_hidden, hyper_node_dim, activation)
        v2e_layers.append(layer)
        for l in range(1, self.gnn_layer_num):
            layer = V2E_layer(hyperedge_dim_hidden, hyperedge_dim_hidden, hyper_node_dim_hidden, activation)
            v2e_layers.append(layer)
        return v2e_layers

    def build_e2v(self, hyperedge_dim_hidden, hyper_node_dim, hyper_node_dim_hidden, activation):
        e2v_layers = nn.ModuleList()
        layer = E2V_layer(hyperedge_dim_hidden, hyper_node_dim, hyper_node_dim_hidden, activation)
        e2v_layers.append(layer)
        for l in range(1, self.gnn_layer_num-1):
            layer = E2V_layer(hyperedge_dim_hidden, hyper_node_dim_hidden, hyper_node_dim_hidden, activation)
            e2v_layers.append(layer)
        layer = E2V_layer(hyperedge_dim_hidden, hyper_node_dim_hidden, 1, activation)
        e2v_layers.append(layer)
        return e2v_layers

    def forward(self, hyperedge, hyper_node, ve_affiliation):
        
        for l in range(self.gnn_layer_num):
            # start_time = time.time()
            hyperedge = self.v2e_layers[l](hyperedge, hyper_node, ve_affiliation)
            # print(f"the time of v2e using {time.time()-start_time}s")
            # start_time = time.time()
            hyper_node = self.e2v_layers[l](hyperedge, hyper_node, ve_affiliation)
            # print(f"the time of e2v using {time.time()-start_time}s")

        # return hyperedge, self.proj_layer(hyper_node)
        return hyperedge, hyper_node