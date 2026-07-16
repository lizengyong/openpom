import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv
from dgllife.model.gnn import MPNNGNN
from dgl.utils import expand_as_pair


class CustomMPNNGNN(MPNNGNN):
    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        super(CustomMPNNGNN,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        self.gnn_layer = NNConv(in_feats=node_out_feats,
                                out_feats=node_out_feats,
                                edge_func=edge_network,
                                aggregator_type=message_aggregator_type,
                                residual=residual)
        self._message_aggregator_type = message_aggregator_type

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        is_npu = node_feats.device.type == 'npu'
        npu_device = node_feats.device if is_npu else None
        if is_npu:
            g = g.int()
            if not hasattr(self, '_npu_converted'):
                gru = self.gru.cpu().float()
                w_ih = gru.weight_ih_l0.data.to(npu_device)
                w_hh = gru.weight_hh_l0.data.to(npu_device)
                b_ih = gru.bias_ih_l0.data.to(npu_device)
                b_hh = gru.bias_hh_l0.data.to(npu_device)
                (self.W_ir, self.W_iz,
                 self.W_in) = w_ih.chunk(3, dim=0)
                (self.W_hr, self.W_hz,
                 self.W_hn) = w_hh.chunk(3, dim=0)
                (self.b_ir, self.b_iz,
                 self.b_in) = b_ih.chunk(3, dim=0)
                (self.b_hr, self.b_hz,
                 self.b_hn) = b_hh.chunk(3, dim=0)
                self.gnn_layer.edge_func = self.gnn_layer.edge_func.to(
                    npu_device).float()
                self.gnn_layer.res_fc = self.gnn_layer.res_fc.cpu()
                self.gnn_layer.bias.data = self.gnn_layer.bias.data.to(
                    npu_device)
                self._npu_converted = True
            edge_w = self.gnn_layer.edge_func(edge_feats)
            edge_w = edge_w.view(-1, self.gnn_layer._in_src_feats,
                                 self.gnn_layer._out_feats)
            src, dst = g.edges()

        for step in range(self.num_step_message_passing):
            if is_npu:
                feat_src, feat_dst = expand_as_pair(node_feats, g)
                msg = torch.bmm(feat_src[src].unsqueeze(1),
                                edge_w).squeeze(1)
                out = torch.zeros(node_feats.shape[0],
                                  self.gnn_layer._out_feats,
                                  device=npu_device)
                node_feats = out.scatter_add(
                    0, dst.unsqueeze(-1).expand(
                        -1, self.gnn_layer._out_feats), msg)
                if self.gnn_layer.res_fc is not None:
                    node_feats = node_feats + feat_dst
                if self.gnn_layer.bias is not None:
                    node_feats = node_feats + self.gnn_layer.bias
            else:
                node_feats = self.gnn_layer(g, node_feats, edge_feats)
            node_feats = F.relu(node_feats)
            if is_npu:
                h_prev = hidden_feats.squeeze(0)
                r = torch.sigmoid(node_feats @ self.W_ir.T + self.b_ir +
                                  h_prev @ self.W_hr.T + self.b_hr)
                z = torch.sigmoid(node_feats @ self.W_iz.T + self.b_iz +
                                  h_prev @ self.W_hz.T + self.b_hz)
                n = torch.tanh(node_feats @ self.W_in.T + self.b_in +
                               r * (h_prev @ self.W_hn.T + self.b_hn))
                node_feats = (1 - z) * n + z * h_prev
                hidden_feats = node_feats.unsqueeze(0)
            else:
                node_feats, hidden_feats = self.gru(
                    node_feats.unsqueeze(0), hidden_feats)
                node_feats = node_feats.squeeze(0)

        return node_feats

