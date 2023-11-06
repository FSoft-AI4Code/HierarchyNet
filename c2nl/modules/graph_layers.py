import torch
import torch.nn as nn
import math
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from modules.position_ffn import PositionwiseFeedForward
class DglHGTConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 *,
                 dropout: int = 0.2,
                 use_norm : bool = False):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            k_dict, q_dict, v_dict = {}, {}, {}
            # Iterate over node-types:
            for node_type, node_id in node_dict.items():
                k_dict[node_type] = self.k_linears[node_id](h[node_type])
                q_dict[node_type] = self.q_linears[node_id](h[node_type])
                v_dict[node_type] = self.v_linears[node_id](h[node_type])
            funcs = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[srctype, etype, dsttype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
            G.multi_update_all(funcs, cross_reducer = 'sum')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = F.gelu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                trans_out = self.a_linears[n_id](t)
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class DglHGTGRUConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 time_steps,
                 *,
                 dropout: int = 0.2,
                 use_norm: bool = False):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()
        self.time_steps = time_steps
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        self.gru_layer = nn.GRUCell(out_dim, out_dim)
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        # self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        for _ in range(self.time_steps):
            with G.local_scope():
                node_dict, edge_dict = self.node_dict, self.edge_dict
                k_dict, q_dict, v_dict = {}, {}, {}
                # Iterate over node-types:
                for node_type, node_id in node_dict.items():
                    k_dict[node_type] = self.k_linears[node_id](h[node_type])
                    q_dict[node_type] = self.q_linears[node_id](h[node_type])
                    v_dict[node_type] = self.v_linears[node_id](h[node_type])
                funcs = {}
                for srctype, etype, dsttype in G.canonical_etypes:
                    sub_graph = G[srctype, etype, dsttype]

                    k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                    v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                    q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                    e_id = self.edge_dict[srctype, etype, dsttype]

                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata[f'v_{e_id}'] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                    sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                    funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
                G.multi_update_all(funcs, cross_reducer = 'sum')
                new_h = {}
                for ntype in G.ntypes:
                    '''
                        Step 3: Target-specific Aggregation
                        x = norm( W[node_type] * gelu( Agg(x) ) + x )
                    '''
                    n_id = node_dict[ntype]
                    # alpha = torch.sigmoid(self.skip[n_id])
                    t = F.gelu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                    trans_out = self.a_linears[n_id](t)
                    # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                    trans_out = self.gru_layer(trans_out, h[ntype])
                    if self.use_norm:
                        new_h[ntype] = self.norms[n_id](trans_out)
                    else:
                        new_h[ntype] = trans_out
                h = new_h
        return h

class DglHGTFFDConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 d_ff,
                 node_dict,
                 edge_dict,
                 n_heads,
                 *,
                 dropout: int = 0.2):
        """
        Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
        """
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        self.ffd_layers = nn.ModuleList()
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            self.ffd_layers.append(PositionwiseFeedForward(out_dim, d_ff))
            self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        # self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            k_dict, q_dict, v_dict = {}, {}, {}
            # Iterate over node-types:
            for node_type, node_id in node_dict.items():
                k_dict[node_type] = self.k_linears[node_id](h[node_type])
                q_dict[node_type] = self.q_linears[node_id](h[node_type])
                v_dict[node_type] = self.v_linears[node_id](h[node_type])
            funcs = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k = k_dict[srctype].view(-1, self.n_heads, self.d_k)
                v = v_dict[srctype].view(-1, self.n_heads, self.d_k)
                q = q_dict[dsttype].view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[srctype, etype, dsttype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata[f't_{e_id}'] = attn_score.unsqueeze(-1)
                funcs[srctype, etype, dsttype] = fn.u_mul_e(f'v_{e_id}', f't_{e_id}', 'm'), fn.sum('m', 't')
            G.multi_update_all(funcs, cross_reducer = 'sum')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                # alpha = torch.sigmoid(self.skip[n_id])
                t = torch.relu(G.nodes[ntype].data['t'].view(-1, self.out_dim))
                trans_out = self.a_linears[n_id](t)
                x = trans_out + h[node_type]
                x = self.drop(x)
                x = self.norms[n_id](x)
                x = self.ffd_layers[n_id](x)
                new_h[node_type] = x
        return new_h

class HGTLayer(nn.Module):
    def __init__(self, config, metadata, func):
        super().__init__()
        self.in_channels = config.channels.in_dim
        self.num_graph_heads = config.num_heads.gnn
        self.num_graph_steps = config.num_steps.gnn

        self.linear_trans = nn.ModuleDict()
        #TODO:
        self.use_norm = False #config.get('normalization', None)

        self.ffd_channels = config.num_heads.d_ff
        # for node_type in metadata[0]:
        #     self.linear_trans[node_type] = nn.Linear(self.in_channels, self.in_channels)

        if func == 'linear':
            self.convs = nn.ModuleList([
                DglHGTConv(self.in_channels, self.in_channels, metadata[0], metadata[1], self.num_graph_heads, use_norm = self.use_norm)
                for _ in range(self.num_graph_steps)
            ])
        elif func == 'gru':
            self.convs = nn.ModuleList([
                DglHGTGRUConv(self.in_channels, self.in_channels, metadata[0], metadata[1], self.num_graph_heads, timestep, use_norm = self.use_norm)
                for timestep in config['time_steps']
            ])
        elif func == 'ffd':
            self.convs = nn.ModuleList([
                DglHGTFFDConvBlock(self.in_channels, self.in_channels, self.ffd_channels, metadata[0], metadata[1], self.num_graph_heads)
                for _ in range(self.num_graph_steps)
            ])
    def forward(self, x_dict, graphs = None):
        # for node_type, x in x_dict.items():
        #     x_dict[node_type] = F.relu_(self.linear_trans[node_type](x))
        for conv in self.convs:
            x_dict = conv(graphs, x_dict)
        return x_dict