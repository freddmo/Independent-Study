import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGatedGCNModelVRP(nn.Module):
    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModelVRP, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

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
        self.num_segments_checkpoint = getattr(config, "num_segments_checkpoint", 0)


        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim // 2, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)

        gcn_layers = [ResidualGatedGCNLayer(self.hidden_dim, self.aggregation) for _ in range(self.num_layers)]
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
        x_vals = self.nodes_coord_embedding(x_nodes_coord)
        x_tags = self.nodes_embedding(x_nodes)
        x = torch.cat((x_vals, x_tags), -1)

        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))
        e_tags = self.edges_embedding(x_edges)
        e = torch.cat((e_vals, e_tags), -1)

        if self.num_segments_checkpoint != 0:
            layer_functions = [lambda args: layer(*args) for layer in self.gcn_layers]
            x, e = torch.utils.checkpoint.checkpoint_sequential(layer_functions, self.num_segments_checkpoint, (x, e))
        else:
            for layer in self.gcn_layers:
                x, e = layer(x, e)

        y_pred_edges = self.mlp_edges(e)

        y = F.log_softmax(y_pred_edges, dim=3)
        y_perm = y.permute(0, 3, 1, 2).contiguous()

        if y_edges is not None:
            edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)
            loss = nn.NLLLoss(edge_cw)(y_perm, y_edges)
        else:
            loss = None

        return y_pred_edges, loss
