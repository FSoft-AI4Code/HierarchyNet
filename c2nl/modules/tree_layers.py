import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import gather_nd

class TBCNNFFDLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.channels.in_dim
        self.out_channels = config.channels.out_dim
        self.num_layers = config.num_layers.tree_nn
        self.dropout_rate = config.dropout_rate
        self.aggr = config.tree_aggr
        
        assert self.in_channels % 2 == 0, "the number of in channels must be even"
        self.w_t = nn.Parameter(torch.randn(self.in_channels, self.out_channels)) 
        self.w_l = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.w_r = nn.Parameter(torch.randn(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels, ))
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(self.dropout_rate)
    
        self.norm_layer = nn.LayerNorm(self.out_channels)

        if self.aggr == 'attention':
            self.attn_linear = nn.Linear(self.out_channels, 1)
            self.out_linear = nn.Linear(self.out_channels, self.out_channels)

        self.init_params()
    def init_params(self):
        nn.init.xavier_normal_(self.w_t)
        nn.init.xavier_normal_(self.w_l)
        nn.init.xavier_normal_(self.w_r)
        if self.aggr == 'attention':
            nn.init.xavier_normal_(self.attn_linear.weight)
    def forward(self, parent_node_embedding, children_index, batch_tree_mask):
        all_num_stmts, num_nodes, num_children = children_index.shape
        parent_node_embedding = parent_node_embedding.reshape(all_num_stmts, num_nodes, -1)
        children_index = children_index.reshape(all_num_stmts, num_nodes, -1)
        batch_tree_mask = batch_tree_mask.reshape(all_num_stmts, -1)
        children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        # print('children_embedding', parent_node_embedding.shape, children_index.shape, children_embedding.shape)
        node_embedding = self.conv_op(parent_node_embedding, children_embedding, children_index)
        all_stmt_embeds = self.aggregate(node_embedding, batch_tree_mask)
        return all_stmt_embeds
    def aggregate(self, node_embedding, batch_tree_mask):
        if self.aggr == 'attention':
            node_logits = self.attn_linear(node_embedding)
            #TODO: mask
            batch_tree_mask.masked_fill_(batch_tree_mask.bool(), -1.0e8)
            node_logits = node_logits + batch_tree_mask.unsqueeze(-1)
            node_scores = F.softmax(node_logits, dim = 1)
            node_embedding = self.out_linear(node_embedding)
            tree_embeddings = (node_scores * node_embedding).sum(dim = 1)
        elif self.aggr == 'max-pooling':
            tree_embeddings = torch.max(node_embedding, dim = 1)[0]
        return tree_embeddings

    def conv_op(self, parent_node_embedding, children_embedding, children_index):
        x = parent_node_embedding
        parent_node_embedding = self.conv_step(parent_node_embedding, children_embedding, children_index, w_t = self.w_t, w_l = self.w_l, w_r = self.w_r, bias = self.bias)
        parent_node_embedding = self.norm_layer(parent_node_embedding + x)
        parent_node_embedding = self.relu(parent_node_embedding)
        parent_node_embedding = self.dropout(parent_node_embedding)
        # children_embedding = self.get_children_embedding_from_parent(parent_node_embedding, children_index)
        return parent_node_embedding
    def conv_step(self, parent_node_embedding, children_embedding, children_index, w_t, w_l, w_r, bias):
        parent_node_embedding = parent_node_embedding.unsqueeze(2)
        tree_embedding = parent_node_embedding if np.prod(children_embedding.shape) == 0 else torch.cat((parent_node_embedding, children_embedding), dim = 2)
        eta_t = self.eta_t(children_index)
        eta_r = self.eta_r(children_index, eta_t)
        eta_l = self.eta_l(children_index, eta_t, eta_r)
        eta = torch.stack((eta_t, eta_l, eta_r), dim = -1)
        weights = torch.stack((w_t, w_l, w_r), dim = 0)
        result = torch.matmul(tree_embedding.permute(0, 1, 3, 2), eta)
        result = torch.tensordot(result, weights, dims = ([3, 2], [0, 1]))
        return result + bias

    def get_children_embedding_from_parent(self, parent_node_embedding, children_index):
        batch_size, num_nodes, num_children = children_index.shape
        channels = parent_node_embedding.shape[-1]
        zero_vecs = torch.zeros(batch_size, 1, channels).to(parent_node_embedding.device)
        lookup_table = torch.cat((zero_vecs, parent_node_embedding[:, 1:]), dim = 1)
        children_index = children_index.unsqueeze(-1)
        batch_index = torch.arange(batch_size).view(batch_size, 1, 1, 1).to(parent_node_embedding.device)
        batch_index = torch.tile(batch_index, (1, num_nodes, num_children, 1))
        children_index = torch.cat((batch_index, children_index), dim = -1)
        return gather_nd(lookup_table, children_index)

    def get_parent_type_embedding(self, parent_node_type_index):
        return self.embedding_layer.get_type_embedding(parent_node_type_index)
    def get_parent_token_embedding(self, parent_node_token_ids):
        return self.embedding_layer.get_token_embedding(parent_node_token_ids).sum(dim = 2)
    def get_children_tokens_embedding(self, children_node_token_ids):
        return self.embedding_layer.get_token_embedding(children_node_token_ids).sum(dim = 3)
    

    def eta_t(self, children):
        batch_size, num_nodes, num_children = children.shape
        return torch.tile(torch.unsqueeze(
            torch.cat((torch.ones(num_nodes, 1), torch.zeros(num_nodes, num_children)), dim = 1),
            dim = 0
        ), (batch_size, 1, 1)).to(children.device)
    def eta_r(self, children, eta_t):
        batch_size, num_nodes, num_children = children.shape
        if num_children == 0:
            return torch.zeros(batch_size, num_nodes, 1).to(children.device)
        num_siblings = torch.tile(torch.count_nonzero(children, dim = 2).unsqueeze(-1), (1, 1, num_children + 1))
        mask = torch.cat((torch.zeros(batch_size, num_nodes, 1).to(children.device), torch.minimum(children, torch.ones_like(children).to(children.device))), dim = 2).to(children.device)
        child_indices = torch.tile(torch.arange(-1, num_children).unsqueeze(0).unsqueeze(0), (batch_size, num_nodes, 1)).to(children.device) * mask
        singles = torch.cat((
            torch.zeros(batch_size, num_nodes, 1),
            torch.full((batch_size, num_nodes, 1), 0.5),
            torch.zeros(batch_size, num_nodes, num_children - 1)
        ), dim = 2).to(children.device)
        return torch.where(num_siblings == 1, singles, (1 - eta_t) * child_indices / (num_siblings - 1))
    def eta_l(self, children, eta_t, eta_r):
        batch_size, num_nodes, _ = children.shape
        mask = torch.cat((torch.zeros(batch_size, num_nodes, 1, device = children.device), torch.minimum(children, torch.ones_like(children, device = children.device))), dim = 2)
        return mask * (1 - eta_t) * (1 - eta_r)