import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGatedGCNModelVRP(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModelVRP, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config.voc_nodes_in
        self.voc_nodes_out = config.num_nodes
        self.voc_edges_in = config.voc_edges_in
        self.voc_edges_out = config.voc_edges_out
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.mlp_layers = config.mlp_layers
        self.aggregation = config.aggregation
        self.num_segments_checkpoint = getattr(config, 'num_segments_checkpoint', 0)
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim // 2, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
        # Node and edge embedding
        x_vals = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        x_tags = self.nodes_embedding(x_nodes)
        x = torch.cat((x_vals, x_tags), -1)

        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), -1)

        if self.num_segments_checkpoint != 0:
            layer_functions = [lambda args: layer(*args) for layer in self.gcn_layers]
            x, e = torch.utils.checkpoint.checkpoint_sequential(layer_functions, self.num_segments_checkpoint, (x, e))
        else:
            for layer in self.gcn_layers:
                x, e = layer(x, e)

        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out

        y = F.log_softmax(y_pred_edges, dim=3)
        y_perm = y.permute(0, 3, 1, 2).contiguous()

        if y_edges is not None:
            edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)
            loss = nn.NLLLoss(edge_cw)(y_perm, y_edges)
        else:
            loss = None

        return y_pred_edges, loss