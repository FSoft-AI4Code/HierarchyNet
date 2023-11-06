from torch.utils.data import Dataset
import jsonlines
import torch
import dgl
import os
import numpy as np
import random
import pickle
import json
from functools import reduce
from tqdm import tqdm
import multiprocessing
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from data.convert_java import convert_into_java_graph
from utils.pad import _pad_batch_2D, _pad_batch_3D, _pad_batch_4D
from inputters import BOS_WORD, EOS_WORD, EMPTY_WORD, CLS_WORD, BOS, UNK
class CustomDataset(Dataset):
    def __init__(self, data_path, parser, *, node_tokenizer, code_tokenizer, text_tokenizer):
        self.data_path = data_path
        self.parser = parser
        self.node_tokenizer = node_tokenizer
        self.data = []
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        with jsonlines.open(data_path) as f:
            for line in f:
                self.data.append(line)
    def __len__(self):
        return 100#len(self.data)
    def get_code_token_ids(self, flattened_nodes):
        code_token_ids = []
        for i, node in enumerate(flattened_nodes):
            if len(node['node_token']) > 0:
                code_token_ids.append(i)
        return code_token_ids
    def __getitem__(self, index):
        item = self.data[index]
        raw_code = item['code']
        _, flattened_nodes, ast_node_ids, ast_dfs_ids, stmts, stmt_indices, G = convert_into_java_graph(self.parser.parse(raw_code.encode()), raw_code, lambda x: self.code_tokenizer.encode(x).tokens)
        # print('[INFO] ast_node_ids', ast_node_ids, ast_dfs_ids)
        flattened_nodes = [{'node_type': BOS_WORD, 'node_token': BOS_WORD}] + flattened_nodes + [{'node_type': EOS_WORD, 'node_token': EOS_WORD}]
        node_type_ids, node_token_ids = self.get_node_ids(flattened_nodes)
        code_token_ids = self.get_code_token_ids(flattened_nodes)
        graph = self.convert_nx_into_dgl(G)
        code_tree_ids = self.get_query_ids_for_token(ast_node_ids, ast_dfs_ids, stmts, stmt_indices, code_token_ids)
        tgt_tokens = [BOS_WORD, CLS_WORD] + item['target_tokens'] + [EOS_WORD]
        tgt_token_ids = self.node_tokenizer.tokens_to_ids(tgt_tokens)
        
        tgt_len = len(tgt_token_ids)
        num_graph_nodes = G.number_of_nodes()
        return node_type_ids, node_token_ids, ast_node_ids, ast_dfs_ids, stmts, stmt_indices, graph, num_graph_nodes, code_token_ids, code_tree_ids, tgt_token_ids, tgt_len
    def get_node_ids(self, nodes):
        node_type_ids = []
        node_token_ids = []
        for node in nodes:
            node_type = node['node_type']
            node_token = node['node_token'] or EMPTY_WORD
            node_type_ids.append(self.node_tokenizer.get_node_type_id(node_type))
            node_token_ids.append(self.node_tokenizer.get_node_token_ids(node_token))
        return node_type_ids, node_token_ids
    def convert_nx_into_dgl(self, nx_graph):
        edge_type = ['ast_edge', 'next_stmt_edge', 'control_flow_edge', 'data_flow_edge']
        graph_data = {}
        for edge_type in edge_type:
            graph_data['node', edge_type, 'node'] = []
        for u, v, edge_type in nx_graph.edges(data = 'edge_type'):
            graph_data[('node', edge_type, 'node')].append(torch.tensor([u, v]))
        return dgl.heterograph(graph_data)
    def get_query_ids_for_token(self, ast_node_ids, ast_dfs_ids, stmts, stmt_indices, code_token_ids):
        all_stmt_dfs_ids = list(map(lambda x: x['all_dfs_ids'], stmts))

        stmt_dfs_ids2node_ids = {}
        for i, stmt_dfs_ids in enumerate(all_stmt_dfs_ids):
            for dfs_ids in stmt_dfs_ids:
                stmt_dfs_ids2node_ids[dfs_ids] = stmt_indices[i]
        dfs_mapping_node_ids = [None] * len(code_token_ids)
        for i in range(len(code_token_ids)):
            code_token_id = code_token_ids[i]
            if code_token_id == 0: 
                dfs_mapping_node_ids[0] = 0
                continue
            if code_token_id in ast_dfs_ids:
                dfs_mapping_node_ids[i] = ast_node_ids[ast_dfs_ids.index(code_token_id)] + 1
            elif code_token_id in stmt_dfs_ids2node_ids:
                dfs_mapping_node_ids[i] = stmt_dfs_ids2node_ids[code_token_id] + 1
        dfs_mapping_node_ids[-1] = 0
        return dfs_mapping_node_ids

class CustomFileDataset(Dataset):
    def __init__(self, data_path_lst, *, node_tokenizer, code_tokenizer, text_tokenizer, src2tgt = None, limit_tgt_len = 30):
        self.idx_names = os.listdir(data_path_lst)
        self.data_path_lst = data_path_lst
        self.node_tokenizer = node_tokenizer
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.src2tgt = src2tgt
        self.limit_tgt_len = limit_tgt_len
        random.shuffle(self.idx_names)
    def __len__(self):
        return len(self.idx_names)
    def get_code_token_ids(self, flattened_nodes):
        code_token_ids = []
        for i, node in enumerate(flattened_nodes):
            if len(node['node_token']) > 0:
                code_token_ids.append(i)
        return code_token_ids
    def __getitem__(self, index):
        item = self.idx_names[index]
        base_path = f'{self.data_path_lst}/{item}'
        flattened_nodes = []
        with jsonlines.open(f'{base_path}/flattened_nodes.jsonl') as f:
            for line in f:
                flattened_nodes.append(line)
        with open(f'{base_path}/ast_node_ids.txt') as f:
            ast_node_ids = list(map(int, f.read().split()))
        with open(f'{base_path}/ast_dfs_ids.txt') as f:
            ast_dfs_ids = list(map(int, f.read().split()))
        stmts = []
        with jsonlines.open(f'{base_path}/stmts.jsonl') as f:
            for line in f:
                stmts.append(line)
        with open(f'{base_path}/stmt_indices.txt') as f:
            stmt_indices = list(map(int, f.read().split()))
        with open(f'{base_path}/G.pickle', 'rb') as f:
            G = pickle.load(f)
        with open(f'{base_path}/target.json') as f:
            target = json.load(f)
        flattened_nodes = [{'node_type': BOS_WORD, 'node_token': BOS_WORD}] + flattened_nodes + [{'node_type': EOS_WORD, 'node_token': EOS_WORD}]
        node_type_ids, node_token_ids = self.get_node_ids(flattened_nodes)
        code_len = len(node_type_ids)
        node_type_ids = node_type_ids 
        node_token_ids = node_token_ids
        code_token_ids = self.get_code_token_ids(flattened_nodes)
        if self.src2tgt is not None:
            src_node_id, src_token_ids = self.get_src_ids_copy(node_token_ids, code_token_ids)
        else:
            src_node_id, src_token_ids = [], []
        code_token_len = len(code_token_ids)
        graph = self.convert_nx_into_dgl(G)
        code_tree_ids = self.get_query_ids_for_token(ast_node_ids, ast_dfs_ids, stmts, stmt_indices, code_token_ids)

        tgt_tokens = self.text_tokenizer.encode(target['target'].lower()).tokens
        if len(tgt_tokens) > self.limit_tgt_len:
            tgt_tokens = tgt_tokens[:self.limit_tgt_len]
            while not tgt_tokens[-1].endswith('</w>'):
                tgt_tokens = tgt_tokens[:-1]
        tgt_tokens = [BOS_WORD] + tgt_tokens + [EOS_WORD]
        tgt_token_ids = list(map(self.text_tokenizer.token_to_id, tgt_tokens))
        tgt_len = len(tgt_token_ids)
        
        num_graph_nodes = G.number_of_nodes()
        return node_type_ids, node_token_ids, ast_node_ids, ast_dfs_ids, stmts, stmt_indices, graph, num_graph_nodes, code_token_ids, code_tree_ids, tgt_token_ids, tgt_len, code_len, code_token_len, src_node_id, src_token_ids
    def get_node_ids(self, nodes):
        node_type_ids = []
        node_token_ids = []
        for node in nodes:
            node_type = node['node_type']
            if node_type == "string_literal":
                node_token = '"<str>"'
            else:
                node_token = node['node_token'] or EMPTY_WORD
            node_type_ids.append(self.node_tokenizer.get_node_type_id(node_type))
            node_token_ids.append(self.node_tokenizer.get_node_token_ids(node_token.replace('</w>', '')))
        return node_type_ids, node_token_ids
    def get_src_ids_copy(self, node_token_ids, code_token_ids):
        node_ids, src_token_ids = [], []
        for i, code_token_id in enumerate(code_token_ids):
            token_ids = node_token_ids[code_token_id]
            if len(token_ids) != 1: continue
            if token_ids[0] in [BOS]: continue
            tgt_token_id = self.src2tgt.get(token_ids[0], UNK)
            if tgt_token_id == UNK: continue
            node_ids.append(i)
            src_token_ids.append(tgt_token_id)
        return node_ids, src_token_ids
    def convert_nx_into_dgl(self, nx_graph):
        edge_type = ['ast_edge', 'next_stmt_edge', 'control_flow_edge', 'data_flow_edge']
        graph_data = {}
        for edge_type in edge_type:
            graph_data['node', edge_type, 'node'] = []
        for u, v, edge_type in nx_graph.edges(data = 'edge_type'):
            graph_data[('node', edge_type, 'node')].append(torch.tensor([u, v]))
        return dgl.heterograph(graph_data, num_nodes_dict = {'node': nx_graph.number_of_nodes()})
    def get_query_ids_for_token(self, ast_node_ids, ast_dfs_ids, stmts, stmt_indices, code_token_ids):
        all_stmt_dfs_ids = list(map(lambda x: x['all_dfs_ids'], stmts))

        stmt_dfs_ids2node_ids = {}
        for i, stmt_dfs_ids in enumerate(all_stmt_dfs_ids):
            for dfs_ids in stmt_dfs_ids:
                stmt_dfs_ids2node_ids[dfs_ids] = stmt_indices[i]
        dfs_mapping_node_ids = [0] * len(code_token_ids)
        for i in range(len(code_token_ids)):
            code_token_id = code_token_ids[i]
            if code_token_id == 0: 
                dfs_mapping_node_ids[0] = 0
                continue
            if code_token_id in ast_dfs_ids:
                dfs_mapping_node_ids[i] = ast_node_ids[ast_dfs_ids.index(code_token_id)] + 1
            elif code_token_id in stmt_dfs_ids2node_ids:
                dfs_mapping_node_ids[i] = stmt_dfs_ids2node_ids[code_token_id] + 1
        dfs_mapping_node_ids[-1] = 0
        return dfs_mapping_node_ids


def custom_collate(batch):
    node_type_ids, node_token_ids, ast_node_ids, ast_dfs_ids, tree_stmts, stmt_indices, graphs, num_graph_nodes, code_token_ids, code_tree_ids, tgt_token_ids, tgt_len, code_len, code_token_len, src_node_ids, src_token_ids = list(zip(*batch))
    # code_len = list(map(len, node_token_ids))
    node_type_ids = _pad_batch_2D(node_type_ids)
    node_token_ids = _pad_batch_3D(node_token_ids)
    # batch_ast_node_ids = _pad_batch_2D(ast_node_ids)
    batch_ast_node_ids = ast_node_ids#list(map(torch.tensor, ast_node_ids))
    batch_ast_dfs_ids = list(map(torch.tensor, ast_dfs_ids))
    code_token_ids = _pad_batch_2D(code_token_ids)
    # print('code_tree_ids', code_tree_ids)
    code_tree_ids = list(map(torch.tensor, code_tree_ids))
    
    # tree_stmts = reduce(list.__add__, tree_stmts)
    batch_tree_node_ids = []
    tree_children_index = []
    batch_tree_ids = reduce(list.__add__, stmt_indices)
    batch_tree_size = []
    batch_tree_mask = []
    max_code_len = node_type_ids.shape[1] #max(code_len)
    for i, ex_stmts in enumerate(tree_stmts):
        # batch_tree_node_ids.append([])
        # tree_children_index.append([])
        # batch_tree_mask.append([])
        for ex in ex_stmts:
            # batch_tree_node_ids[-1].append(ex['all_dfs_ids'])
            # tree_children_index[-1].append(ex['children_index'])
            # batch_tree_mask[-1].append([0] * len(ex['all_dfs_ids']))
            batch_tree_node_ids.append((np.array(ex['all_dfs_ids']) + i * max_code_len).tolist())
            tree_children_index.append(ex['children_index'])
            batch_tree_mask.append([0] * len(ex['all_dfs_ids']))
        batch_tree_size.append(len(ex_stmts))

    batch_src_node_ids = list(src_node_ids)
    batch_src_token_len = list(map(len, src_node_ids))
    batch_src_token_ids = _pad_batch_2D(src_token_ids)
    # batch_tree_node_ids = _pad_batch_2D([ex['all_dfs_ids'] for ex in tree_stmts])
    # tree_children_index = [ex['children_index'] for ex in tree_stmts]
    # print('tree_children_index', tree_children_index)
    # batch_tree_node_ids = _pad_batch_3D(batch_tree_node_ids)
    # batch_tree_node_ids = list(map(torch.tensor, batch_tree_node_ids))

    batch_tree_mask = _pad_batch_2D(batch_tree_mask, value = 1) # true if this position is padded
    # print('batch_tree_mask batch_tree_mask', batch_tree_mask.shape)
    # print('batch_tree_node_ids', len(batch_tree_node_ids))
    
    # batch_tree_node_ids = list(map(lambda tup: (np.array(tup[0]) + tup[1] * max_code_len).tolist(), zip(batch_tree_node_ids, range(len(code_len)))))
    # print('batch_tree_node_ids', batch_tree_node_ids)
   
    batch_tree_node_ids = _pad_batch_2D(batch_tree_node_ids)
    tree_children_index = _pad_batch_3D(tree_children_index)

    # tree_children_index = list(map(torch.tensor, tree_children_index))
    # print(tree_stmts)
    # tree_stmts = reduce(list.__add__, tree_stmts)
    # batch_trees = make_batch(tree_stmts)
    tgt_token_ids = _pad_batch_2D(tgt_token_ids)
            # masked_exp_ids.append(1)

    return {
        'batch_size': len(node_type_ids),
        'node_type_ids': torch.tensor(node_type_ids), 
        'node_token_ids': torch.tensor(node_token_ids), 
        'code_len': torch.tensor(code_len),
        'code_token_len': torch.tensor(code_token_len),
        'batch_src_node_ids': batch_src_node_ids,
        'batch_src_token_ids': torch.tensor(batch_src_token_ids) if len(batch_src_token_ids[0]) > 0 else [],
        'batch_src_token_len': torch.tensor(batch_src_token_len) if len(batch_src_token_ids[0]) > 0 else [],
        'batch_ast_node_ids': batch_ast_node_ids,
        'batch_ast_dfs_ids': batch_ast_dfs_ids,
        'batch_tree_ids': batch_tree_ids,
        'batch_tree_node_ids': torch.tensor(batch_tree_node_ids),
        'batch_tree_mask': torch.tensor(batch_tree_mask),
        'batch_tree_size': batch_tree_size,
        'tree_children_index': torch.tensor(tree_children_index),
        'code_token_ids': torch.tensor(code_token_ids),
        'code_tree_ids': code_tree_ids,
        'token_tgt_seq': torch.tensor(tgt_token_ids),
        'tgt_len': torch.tensor(tgt_len),
        'src_map': 0,
        'alignment': 0,
        # 'batch_trees': batch_trees, 
        # 'stmt_indices': stmt_indices, 
        'graphs': dgl.batch(graphs),
        'num_graph_nodes': num_graph_nodes
    }

import math
from typing import TypeVar, Optional, Iterator
import torch.distributed as dist
from functools import reduce
T_co = TypeVar('T_co', covariant=True)
class DistSortedBatchSampler(DistributedSampler):
    def __init__(self, lengths, batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.

        self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        np.random.seed(self.seed + self.epoch)
        nums = np.random.random(self.total_size)
        lengths = np.array(
            [(-l[0], -l[1], -l[2], n) for l, n in zip(self.lengths, nums)],
            dtype=[('l1', np.int_), ('l2', np.int_), ('l3', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'l3', 'rand')).tolist()
        
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[-padding_size:]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[-padding_size:]
        assert len(indices) == self.total_size

        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        # subsample
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            np.random.seed(self.seed + self.epoch)
            np.random.shuffle(batches)
        local_indices = batches[self.rank:len(batches):self.num_replicas]
        local_indices = reduce(list.__add__, local_indices, [])
        return iter(local_indices)


    def __len__(self) -> int:
        return self.num_samples
    def set_epoch(self, epoch):
        self.epoch = epoch


