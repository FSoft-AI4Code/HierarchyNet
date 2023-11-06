import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from functools import reduce
from prettytable import PrettyTable

from modules.tree_layers import TBCNNFFDLayer
from modules.graph_layers import HGTLayer
from modules.embedder import Embedder
from modules.transformer_modules import Encoder, Decoder, SerialDecoder
from modules.multi_head_attn import MultiHeadedAttention
from modules.proj import TwoNonLinearProj
from modules.global_attention import GlobalAttention
from modules.copy_generator import CopyLayer
from utils.ops import gather_nd
from utils.misc import sequence_mask
from inputters import PAD, BOS
from models.t5_anycode import CustomT5
from utils.misc import sequence_mask


class EncoderGating(nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.linear = nn.Linear(dim_size, dim_size)
        self.two_embeds_proj = TwoNonLinearProj(dim_size, dim_size)
    def forward(self, code_token_embeds, code_tree_embeds, graph_embeds):
        scores = torch.sigmoid(self.linear(graph_embeds))
        code_embeds = code_tree_embeds
        code_embeds = self.two_embeds_proj(code_embeds)
        return code_token_embeds * scores + (1 - scores) * code_embeds

class DecoderGating(nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        # self.decoder_context_linear = nn.Linear(dim_size, 1)
        # self.decoder_states_linear = nn.Linear(dim_size, 1)
        # self.graph_linear = nn.Linear(dim_size, 1)
        self.dim_size = dim_size
        self.bilinear = nn.Bilinear(dim_size, 2 * dim_size, 1)
        self.masked_node_linear = nn.Linear(dim_size, 1)
        self.two_embeds_proj = TwoNonLinearProj(dim_size * 2, dim_size)
    def forward(self, decoder_token_embeds, decoder_context_embeds, graph_embeds):
        # print('decoder_context_embeds', decoder_context_embeds.shape, decoder_context_embeds.shape,graph_embeds.shape, masked_node_embeds.shape )
        control_embeds = torch.cat((decoder_context_embeds, graph_embeds), dim = -1)
        scores = torch.einsum('nik, okl, njl -> nio', decoder_token_embeds, self.bilinear.weight, control_embeds) + self.bilinear.bias
        scores = scores / (self.dim_size ** 0.5)
        # scores =  self.decoder_states_linear(decoder_token_embeds) + self.decoder_context_linear(decoder_context_embeds) + self.graph_linear(graph_embeds) + self.masked_node_linear(masked_node_embeds)
        scores = torch.sigmoid(scores) 
       
        fused_masked_node_embeds = torch.cat((decoder_token_embeds, graph_embeds.tile(1, decoder_token_embeds.size(1), 1)), dim = -1)
        fused_masked_node_embeds = self.two_embeds_proj(fused_masked_node_embeds)
        return decoder_token_embeds * scores + (1 - scores) * fused_masked_node_embeds

class CodeSumModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config)
        self.encoder = Encoder(config, config.channels.in_dim)
        self.decoder = SerialDecoder(config, config.channels.out_dim)
        self.layer_wise_attn = config.layer_wise_attn

        self.tree_layer = TBCNNFFDLayer(config)
        metadata = [('node', 'ast_edge', 'node'), #000
                            ('node', 'control_flow_edge', 'node'), #011
                            ('node', 'next_stmt_edge','node'),
                            ('node', 'data_flow_edge', 'node')] 

        metadata = (
                {'node': 0},
                dict(zip(metadata, range(len(metadata))))     
            ) 
        self.ast_node_proj = TwoNonLinearProj(config.channels.out_dim, config.channels.out_dim)
        self.code_query_proj = TwoNonLinearProj(config.channels.out_dim * 2, config.channels.out_dim)
        self.graph_layer = HGTLayer(config, metadata, func = 'ffd')
        self.graph_aggr = dgl.nn.GlobalAttentionPooling(nn.Linear(config.channels.out_dim, 1))
        self.cross_attention_code_graph = MultiHeadedAttention(8, 512, 64, 64, coverage = False)
        # self.cross_attention_norm = nn.LayerNorm(config.channels.out_dim)
        # self.drop = nn.Dropout(0.1)
        self.fused_gating_encoder = EncoderGating(config.channels.out_dim)
        self.decoder_gating = DecoderGating(config.channels.out_dim)
        self.token_generator = nn.Linear(self.decoder.input_size, config.vocab_size.token)
       
        self._copy = config.copy_attn.apply
        if self._copy:
            self.global_attn_layer = GlobalAttention(config.channels.out_dim, False, config.copy_attn.context)
            self.copy_layer = CopyLayer(config.channels.out_dim, self.token_generator)
            self.criterion = nn.NLLLoss(reduction = 'none')
        else:
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = nn.CrossEntropyLoss(reduction = 'none')
            # self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
    def get_tree_node_embedddings(self, all_node_embeddings, batch_node_ids):
        batch_size, num_nodes, num_children = batch_node_ids.shape
        channels = all_node_embeddings.shape[-1]
        zero_vecs = torch.zeros(batch_size, 1, channels).to(all_node_embeddings.device)
        lookup_table = torch.cat((zero_vecs, all_node_embeddings[:, 1:]), dim = 1)
        batch_node_ids = batch_node_ids.unsqueeze(-1)
        batch_index = torch.arange(batch_size).view(batch_size, 1, 1, 1).to(all_node_embeddings.device)
        batch_index = torch.tile(batch_index, (1, num_nodes, num_children, 1))
        batch_node_ids = torch.cat((batch_index, batch_node_ids), dim = -1)
        return gather_nd(lookup_table, batch_node_ids)
    def get_code_token_embedddings(self, all_node_embeddings, code_token_ids):
        code_token_embeddings = []
        for node_embeds, token_ids in zip(all_node_embeddings, code_token_ids):
            code_token_embeddings.append(node_embeds[token_ids])
        return code_token_embeddings
    def get_node_graph_embeds_for_token(self, tree_embeddings, code_tree_ids):
        code_tree_embeds = []
        for tree_embeds, tree_ids in zip(tree_embeddings, code_tree_ids):
            tree_embeds = F.pad(tree_embeds, (0, 0, 1, 0))
            code_tree_embeds.append(tree_embeds[tree_ids])
        return code_tree_embeds

    def get_ast_node_embedddings(self, all_node_embeddings, batch_ast_dfs_ids):
        ast_node_embeddings = []
        for i in range(all_node_embeddings.shape[0]):
            if len(batch_ast_dfs_ids[i]) == 0: continue
            ast_node_embeddings.append(all_node_embeddings[i, batch_ast_dfs_ids[i]])
        return ast_node_embeddings
        ơm
    def encode(self,
                node_token_ids,
                node_type_ids,
                code_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size, 
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids,
                batch_src_node_ids):
        batch_size = code_len.size(0)
        # embed and encode the source sequence
        code_rep = self.embedder(node_token_ids,
                                 node_type_ids,
                                 mode='encoder')
        # print('code_rep', code_rep.min(), code_rep.max())
        memory_bank, _ = self.encoder(code_rep, code_len)  # B x seq_len x h
        # print('code_rep ENCODER', code_rep.min(), code_rep.max())
        tree_node_embeddings = self.get_tree_node_embedddings(memory_bank.reshape(1, -1, memory_bank.size(-1)), batch_tree_node_ids.unsqueeze(0)).squeeze(0)
        # print('tree_node_embeddings', tree_node_embeddings.min(), tree_node_embeddings.max())
        ast_node_embeddings = self.get_ast_node_embedddings(memory_bank, batch_ast_dfs_ids)

        
        batch_ast_node_ids = reduce(list.__add__, batch_ast_node_ids)
        if len(batch_ast_node_ids):
            ast_node_embeddings = torch.cat(ast_node_embeddings, dim= 0)
            ast_node_embeddings = self.ast_node_proj(ast_node_embeddings)

        tree_node_embeddings = self.tree_layer(tree_node_embeddings, tree_children_index, batch_tree_mask)
        batch_graph_node_ids = batch_ast_node_ids + batch_tree_ids
        batch_graph_node_ids = np.argsort(batch_graph_node_ids)
        if len(batch_ast_node_ids):
            graph_node_embeds = torch.cat((ast_node_embeddings, tree_node_embeddings), dim = 0)
        else:
            graph_node_embeds = tree_node_embeddings
        graph_node_embeds = graph_node_embeds[batch_graph_node_ids]

        split_graph_node_embeds = torch.split(graph_node_embeds, num_graph_nodes)
        code_token_embeds = self.get_code_token_embedddings(memory_bank, code_token_ids)
        code_tree_embeds = self.get_node_graph_embeds_for_token(split_graph_node_embeds, code_tree_ids)
        code_token_embeds = nn.utils.rnn.pad_sequence(code_token_embeds, batch_first = True)
        code_tree_embeds = nn.utils.rnn.pad_sequence(code_tree_embeds, batch_first = True)
        code_query_embeds = torch.cat((code_token_embeds, code_tree_embeds), dim = -1)
        code_query_embeds = self.code_query_proj(code_query_embeds)
       
        graph_node_embeds = {'node': graph_node_embeds}
        graph_node_embeds = self.graph_layer(graph_node_embeds, graphs)['node']
        graph_embeds = self.graph_aggr(graphs, graph_node_embeds).unsqueeze(1)
        # print('graph_embeds', graph_embeds.min(), graph_embeds.max())
        graphs.nodes['node'].data['ft'] = graph_node_embeds
        graph_node_embeds = []
        graph_num_nodes = []
        
        for i, graph in enumerate(dgl.unbatch(graphs)):
            graph_node_embeds.append(graph.nodes['node'].data['ft'])
            # graph_num_nodes.append(len(graph_node_embeds[-1]))
            graph_num_nodes.append(len(graph_node_embeds[-1]) + 1)
            # graph_embeds.append(self.graph_aggr(graph, out_graph_node_embeddings[-1]))

        graph_node_embeds = nn.utils.rnn.pad_sequence(graph_node_embeds, batch_first = True)
        graph_node_embeds = torch.cat((graph_embeds, graph_node_embeds), dim = 1)
        graph_num_nodes = torch.tensor(graph_num_nodes).type_as(graph_node_embeds)
        graph_node_mask = ~sequence_mask(graph_num_nodes, max_len = max(graph_num_nodes)).unsqueeze(1)
        graph_node_mask = graph_node_mask.type_as(graph_node_embeds)
        attn_code_embeds = self.cross_attention_code_graph(key = graph_node_embeds, value = graph_node_embeds, query = code_query_embeds, mask = graph_node_mask)[0]
        final_encoder_states = self.fused_gating_encoder(code_token_embeds, attn_code_embeds, graph_embeds)
        # final_encoder_states = code_token_embeds

        if self._copy:
            src_token_states = self.get_code_token_embedddings(final_encoder_states, batch_src_node_ids)
            src_token_states = nn.utils.rnn.pad_sequence(src_token_states, batch_first = True)
        else:
            src_token_states = None
        return final_encoder_states, graph_embeds, graph_node_embeds, graph_num_nodes, src_token_states
    def _run_forward_ml(self,
                        node_token_ids,
                        node_type_ids,
                        code_len,
                        code_token_len, 
                        batch_ast_node_ids, 
                        batch_ast_dfs_ids,
                        batch_tree_ids,
                        batch_tree_node_ids, 
                        batch_tree_mask,
                        batch_tree_size, 
                        tree_children_index,
                        graphs, 
                        num_graph_nodes,
                        code_token_ids,
                        code_tree_ids,
                        token_tgt_seq,
                        tgt_len, 
                        batch_src_node_ids,
                        batch_src_token_ids,
                        batch_src_token_len,
                        src_map,
                        alignment,
                        **kwargs):
 
        final_encoder_states, graph_embeds, graph_node_embeds, graph_num_nodes, src_token_states = self.encode(node_token_ids,
                                node_type_ids,
                                code_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size, 
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids,
                                batch_src_node_ids)
        # embed and encode the target sequence
        # print('final_encoder_states', final_encoder_states.shape, code_len, code_token_len)
        tgt_token_emb = self.embedder(token_tgt_seq,
                                 mode='decoder')
        tgt_pad_mask = ~sequence_mask(tgt_len, max_len = tgt_token_emb.size(1))
        enc_outputs = final_encoder_states
        layer_wise_dec_out, _ = self.decoder(enc_outputs,
                                                 code_token_len,
                                                 graph_node_embeds, graph_num_nodes,
                                                 tgt_pad_mask,
                                                 tgt_token_emb)
        token_decoder_outputs = layer_wise_dec_out[-1]

        # token_decoder_outputs = self.decoder_gating(token_decoder_outputs, token_decoder_output_context, graph_embeds)
        token_target = token_tgt_seq[:, 1:].contiguous()
        

        if self._copy:
            _, copy_attn, _ = self.global_attn_layer(token_decoder_outputs, src_token_states,
                                                    memory_lengths = batch_src_token_len,
                                                    softmax_weights = True)
            token_scores = self.copy_layer(token_decoder_outputs, copy_attn, batch_src_token_ids)
            token_scores = torch.log(token_scores[:, :-1, :].contiguous() + 1e-6)  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(token_scores.view(-1, token_scores.size(2)),
                                     token_target.view(-1))
        else:
            token_decoder_outputs = token_decoder_outputs
            token_scores = self.token_generator(token_decoder_outputs)  # `batch x tgt_len x vocab_size`
            token_scores = token_scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(token_scores.view(-1, token_scores.size(2)),
                                     token_target.view(-1))
        ml_loss = ml_loss.view(*token_scores.size()[:-1])
        ml_loss = ml_loss.mul(token_target.ne(PAD).float()).sum(dim = 1)
        loss = {}
        loss['ml_loss'] = ml_loss.mean()
        loss['loss_per_token'] = ml_loss.div((tgt_len - 1).float()).mean()
        return loss

    def forward(self,
                node_token_ids,
                node_type_ids,
                code_len,
                code_token_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size,
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids,
                token_tgt_seq,
                tgt_len, 
                batch_src_node_ids,
                batch_src_token_ids,
                batch_src_token_len,
                src_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(node_token_ids,
                                        node_type_ids,
                                        code_len,
                                        code_token_len,
                                        batch_ast_node_ids, 
                                        batch_ast_dfs_ids,
                                        batch_tree_ids,
                                        batch_tree_node_ids, 
                                        batch_tree_mask,
                                        batch_tree_size,
                                        tree_children_index,
                                        graphs, 
                                        num_graph_nodes,
                                        code_token_ids,
                                        code_tree_ids,
                                        token_tgt_seq,
                                        tgt_len, 
                                        batch_src_node_ids,
                                        batch_src_token_ids,
                                        batch_src_token_len,
                                        src_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(node_token_ids,
                                node_type_ids,
                                code_len,
                                code_token_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size,
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids,
                            src_map,
                            alignment,
                               **kwargs)

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        tgt_chars = None

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len)

        attns = {"coverage": None}
        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']

        # +1 for <EOS> token

        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self._copy:
                raise NotImplementedError
            else:
                prediction = self.token_generator(decoder_outputs.squeeze(1))
                prediction = F.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))
            if self._copy:
                mask = tgt.gt(len(params['tgt_dict']) - 1)
                copy_info.append(mask.float().squeeze(1))

            # words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            # tgt_chars = None


            # words = [params['tgt_dict'][w] for w in words]
            # words = torch.Tensor(words).type_as(tgt)
            tgt_words = tgt#tgt.unsqueeze(1)
        return dec_preds, attentions, copy_info, dec_log_probs

    def decode(self,
                node_token_ids,
                node_type_ids,
                code_len,
                code_token_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size,
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids,
                src_map,
                alignment,
               **kwargs):

        # batch_size = code_len.size(0)
        # # embed and encode the source sequence
        # code_rep = self.embedder(node_token_ids,
        #                          node_type_ids,
        #                          mode='encoder')
        # memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h

        # tree_node_embeddings = self.get_tree_node_embedddings(memory_bank.reshape(1, -1, memory_bank.size(-1)), batch_tree_node_ids.unsqueeze(0)).squeeze(0)
        # ast_node_embeddings = self.get_ast_node_embedddings(memory_bank, batch_ast_dfs_ids)

        
        # batch_ast_node_ids = reduce(list.__add__, batch_ast_node_ids)
        # if len(batch_ast_node_ids):
        #     ast_node_embeddings = torch.cat(ast_node_embeddings, dim= 0)
        #     ast_node_embeddings = self.ast_node_proj(ast_node_embeddings)

        # tree_node_embeddings = self.tree_layer(tree_node_embeddings, tree_children_index, batch_tree_mask)
        
        # batch_graph_node_ids = batch_ast_node_ids + batch_tree_ids
        # batch_graph_node_ids = np.argsort(batch_graph_node_ids)
        # if len(batch_ast_node_ids):
        #     graph_node_embeds = torch.cat((ast_node_embeddings, tree_node_embeddings), dim = 0)
        # else:
        #     graph_node_embeds = tree_node_embeddings
        # graph_node_embeds = graph_node_embeds[batch_graph_node_ids]

        # split_graph_node_embeds = torch.split(graph_node_embeds, num_graph_nodes)
        # code_token_embeds = self.get_code_token_embedddings(memory_bank, code_token_ids)
        # code_tree_embeds = self.get_node_graph_embeds_for_token(split_graph_node_embeds, code_tree_ids)
        # code_token_embeds = nn.utils.rnn.pad_sequence(code_token_embeds, batch_first = True)
        # code_tree_embeds = nn.utils.rnn.pad_sequence(code_tree_embeds, batch_first = True)
        # code_query_embeds = torch.cat((code_token_embeds, code_tree_embeds), dim = -1)
        # code_query_embeds = self.code_query_proj(code_query_embeds)
       
        # graph_node_embeds = {'node': graph_node_embeds}
        # graph_node_embeds = self.graph_layer(graph_node_embeds, graphs)['node']
        # graph_embeds = self.graph_aggr(graphs, graph_node_embeds).unsqueeze(1)
        # graphs.nodes['node'].data['ft'] = graph_node_embeds
        # graph_node_embeds = []
        # graph_num_nodes = []
        
        # masked_node_embeddings = []
        # for i, graph in enumerate(dgl.unbatch(graphs)):
        #     graph_node_embeds.append(graph.nodes['node'].data['ft'])
        #     graph_num_nodes.append(len(graph_node_embeds[-1]))
        #     masked_node_embeddings.append(graph_node_embeds[-1][masked_exp_ids[i]]) 
        #     # graph_embeds.append(self.graph_aggr(graph, out_graph_node_embeddings[-1]))
        # masked_node_embeddings = torch.stack(masked_node_embeddings, dim = 0).unsqueeze(1)

        # graph_node_embeds = nn.utils.rnn.pad_sequence(graph_node_embeds, batch_first = True)
        
        # graph_node_mask = ~sequence_mask(torch.tensor(graph_num_nodes), max_len = max(graph_num_nodes)).unsqueeze(1)
        # graph_node_mask = graph_node_mask.type_as(graph_node_embeds)
        # attn_code_embeds = self.cross_attention_code_graph(key = graph_node_embeds, value = graph_node_embeds, query = code_query_embeds, mask = graph_node_mask)[0]
        # final_encoder_states = self.fused_gating_encoder(code_token_embeds, attn_code_embeds, graph_embeds)

        final_encoder_states, graph_embeds, masked_node_embeddings = self.encode(node_token_ids,
                                node_type_ids,
                                code_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size, 
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids)
        memory_bank = final_encoder_states
        params = dict()
        params['memory_bank'] = final_encoder_states
        # params['layer_wise_outputs'] = layer_wise_outputs
        params['src_len'] = code_token_len
        # params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        # params['src_mask'] = kwargs['code_mask_rep']
        # params['fill'] = kwargs['fill']
        # params['blank'] = kwargs['blank']
        # params['src_dict'] = kwargs['src_dict']
        # params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = 42#kwargs['max_len']
        # params['src_words'] = code_word_rep

        dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return dec_preds, copy_info, memory_bank, attentions
        # return {
        #     'predictions': dec_preds,
        #     'copy_info': copy_info,
        #     'memory_bank': memory_bank,
        #     'attentions': attentions
        # }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

from transformers import T5Config
class T5CodeSumModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config)
        self.t5_model = CustomT5(T5Config())
        self.tree_layer = TBCNNFFDLayer(config)
        metadata = [('node', 'ast_edge', 'node'), #000
                            ('node', 'control_flow_edge', 'node'), #011
                            ('node', 'next_stmt_edge','node'),
                            ('node', 'data_flow_edge', 'node')] 

        metadata = (
                {'node': 0},
                dict(zip(metadata, range(len(metadata))))     
            ) 
        self.ast_node_proj = TwoNonLinearProj(config.channels.out_dim, config.channels.out_dim)
        self.code_query_proj = TwoNonLinearProj(config.channels.out_dim * 2, config.channels.out_dim)
        self.graph_layer = HGTLayer(config, metadata, func = 'ffd')
        self.graph_aggr = dgl.nn.GlobalAttentionPooling(nn.Linear(config.channels.out_dim, 1))
        self.cross_attention_code_graph = MultiHeadedAttention(8, 512, 64, 64, coverage = True)
        # self.cross_attention_norm = nn.LayerNorm(config.channels.out_dim)
        # self.drop = nn.Dropout(0.1)
        self.fused_gating_encoder = EncoderGating(config.channels.out_dim)
        self.decoder_gating = DecoderGating(config.channels.out_dim)
        self.token_generator = nn.Linear(config.channels.out_dim, config.vocab_size.token)
       
        # if config.share_decoder_embeddings:
        #     if self.embedder.use_tgt_word:
        #         assert config.emsize == self.decoder.input_size
        #         self.generator.weight = self.embedder.tgt_word_embeddings.word_lut.weight

        self._copy = config.copy_attn
        if self._copy:
            raise NotImplementedError
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
    def get_tree_node_embedddings(self, all_node_embeddings, batch_node_ids):
        batch_size, num_nodes, num_children = batch_node_ids.shape
        channels = all_node_embeddings.shape[-1]
        zero_vecs = torch.zeros(batch_size, 1, channels).to(all_node_embeddings.device)
        lookup_table = torch.cat((zero_vecs, all_node_embeddings[:, 1:]), dim = 1)
        batch_node_ids = batch_node_ids.unsqueeze(-1)
        batch_index = torch.arange(batch_size).view(batch_size, 1, 1, 1).to(all_node_embeddings.device)
        batch_index = torch.tile(batch_index, (1, num_nodes, num_children, 1))
        batch_node_ids = torch.cat((batch_index, batch_node_ids), dim = -1)
        return gather_nd(lookup_table, batch_node_ids)
    def get_code_token_embedddings(self, all_node_embeddings, code_token_ids):
        # batch_size, num_tokens = code_token_ids.shape
        # channels = all_node_embeddings.shape[-1]
        # zero_vecs = torch.zeros(batch_size, 1, channels).to(all_node_embeddings.device)
        # lookup_table = torch.cat((zero_vecs, all_node_embeddings), dim = 1)
        # code_token_ids = code_token_ids.unsqueeze(-1)
        # batch_index = torch.arange(batch_size).view(batch_size, 1, 1).to(all_node_embeddings.device)
        # batch_index = torch.tile(batch_index, (1, num_tokens, 1))
        # code_token_ids = torch.cat((batch_index, code_token_ids + 1), dim = -1)
        # return gather_nd(lookup_table, code_token_ids)
        code_token_embeddings = []
        for node_embeds, token_ids in zip(all_node_embeddings, code_token_ids):
            code_token_embeddings.append(node_embeds[token_ids])
        return code_token_embeddings
    def get_node_graph_embeds_for_token(self, tree_embeddings, code_tree_ids):
        code_tree_embeds = []
        for tree_embeds, tree_ids in zip(tree_embeddings, code_tree_ids):
            tree_embeds = F.pad(tree_embeds, (0, 0, 1, 0))
            code_tree_embeds.append(tree_embeds[tree_ids])
        return code_tree_embeds

    def get_ast_node_embedddings(self, all_node_embeddings, batch_ast_dfs_ids):
        ast_node_embeddings = []
        for i in range(all_node_embeddings.shape[0]):
            if len(batch_ast_dfs_ids[i]) == 0: continue
            ast_node_embeddings.append(all_node_embeddings[i, batch_ast_dfs_ids[i]])
        return ast_node_embeddings
        
    def encode(self,
                node_token_ids,
                node_type_ids,
                code_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size, 
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids):
        batch_size = code_len.size(0)
        # embed and encode the source sequence
        code_rep = self.embedder(node_token_ids,
                                 node_type_ids,
                                 mode='encoder')
        # source_mask = sequence_mask(code_len, code_rep.shape[1])
        source_mask = node_type_ids != 0
        memory_bank = self.t5_model.encode(code_rep, source_mask)
        return memory_bank, source_mask
        # print('code_rep', code_rep.min(), code_rep.max())
        memory_bank, _ = self.encoder(code_rep, code_len)  # B x seq_len x h
        # print('code_rep ENCODER', code_rep.min(), code_rep.max())
        tree_node_embeddings = self.get_tree_node_embedddings(memory_bank.reshape(1, -1, memory_bank.size(-1)), batch_tree_node_ids.unsqueeze(0)).squeeze(0)
        # print('tree_node_embeddings', tree_node_embeddings.min(), tree_node_embeddings.max())
        ast_node_embeddings = self.get_ast_node_embedddings(memory_bank, batch_ast_dfs_ids)

        
        batch_ast_node_ids = reduce(list.__add__, batch_ast_node_ids)
        if len(batch_ast_node_ids):
            ast_node_embeddings = torch.cat(ast_node_embeddings, dim= 0)
            # print('before ast_node_embeddings', ast_node_embeddings.min(), ast_node_embeddings.max())
            ast_node_embeddings = self.ast_node_proj(ast_node_embeddings)
            # print('after ast_node_embeddings', ast_node_embeddings.min(), ast_node_embeddings.max())

        tree_node_embeddings = self.tree_layer(tree_node_embeddings, tree_children_index, batch_tree_mask)
        # print('AFTER tree_node_embeddings', tree_node_embeddings.min(), tree_node_embeddings.max())
        batch_graph_node_ids = batch_ast_node_ids + batch_tree_ids
        batch_graph_node_ids = np.argsort(batch_graph_node_ids)
        if len(batch_ast_node_ids):
            graph_node_embeds = torch.cat((ast_node_embeddings, tree_node_embeddings), dim = 0)
        else:
            graph_node_embeds = tree_node_embeddings
        graph_node_embeds = graph_node_embeds[batch_graph_node_ids]

        split_graph_node_embeds = torch.split(graph_node_embeds, num_graph_nodes)
        code_token_embeds = self.get_code_token_embedddings(memory_bank, code_token_ids)
        code_tree_embeds = self.get_node_graph_embeds_for_token(split_graph_node_embeds, code_tree_ids)
        code_token_embeds = nn.utils.rnn.pad_sequence(code_token_embeds, batch_first = True)
        code_tree_embeds = nn.utils.rnn.pad_sequence(code_tree_embeds, batch_first = True)
        code_query_embeds = torch.cat((code_token_embeds, code_tree_embeds), dim = -1)
        code_query_embeds = self.code_query_proj(code_query_embeds)
       
        graph_node_embeds = {'node': graph_node_embeds}
        graph_node_embeds = self.graph_layer(graph_node_embeds, graphs)['node']
        graph_embeds = self.graph_aggr(graphs, graph_node_embeds).unsqueeze(1)
        # print('graph_embeds', graph_embeds.min(), graph_embeds.max())
        graphs.nodes['node'].data['ft'] = graph_node_embeds
        graph_node_embeds = []
        graph_num_nodes = []
        
        for i, graph in enumerate(dgl.unbatch(graphs)):
            graph_node_embeds.append(graph.nodes['node'].data['ft'])
            graph_num_nodes.append(len(graph_node_embeds[-1]))
            # graph_embeds.append(self.graph_aggr(graph, out_graph_node_embeddings[-1]))

        graph_node_embeds = nn.utils.rnn.pad_sequence(graph_node_embeds, batch_first = True)
        
        graph_node_mask = ~sequence_mask(torch.tensor(graph_num_nodes), max_len = max(graph_num_nodes)).unsqueeze(1)
        graph_node_mask = graph_node_mask.type_as(graph_node_embeds)
        attn_code_embeds = self.cross_attention_code_graph(key = graph_node_embeds, value = graph_node_embeds, query = code_query_embeds, mask = graph_node_mask)[0]
        # final_encoder_states = self.fused_gating_encoder(code_token_embeds, attn_code_embeds, graph_embeds)
        return code_token_embeds, graph_embeds
    def _run_forward_ml(self,
                        node_token_ids,
                        node_type_ids,
                        code_len,
                        batch_ast_node_ids, 
                        batch_ast_dfs_ids,
                        batch_tree_ids,
                        batch_tree_node_ids, 
                        batch_tree_mask,
                        batch_tree_size, 
                        tree_children_index,
                        graphs, 
                        num_graph_nodes,
                        code_token_ids,
                        code_tree_ids,
                        token_tgt_seq,
                        tgt_len, 
                        src_map,
                        alignment,
                        **kwargs):
 
        # batch_size = code_len.size(0)
        # # embed and encode the source sequence
        # code_rep = self.embedder(node_token_ids,
        #                          node_type_ids,
        #                          mode='encoder')
        # memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h

        # tree_node_embeddings = self.get_tree_node_embedddings(memory_bank.reshape(1, -1, memory_bank.size(-1)), batch_tree_node_ids.unsqueeze(0)).squeeze(0)
        # ast_node_embeddings = self.get_ast_node_embedddings(memory_bank, batch_ast_dfs_ids)

        
        # batch_ast_node_ids = reduce(list.__add__, batch_ast_node_ids)
        # if len(batch_ast_node_ids):
        #     ast_node_embeddings = torch.cat(ast_node_embeddings, dim= 0)
        #     ast_node_embeddings = self.ast_node_proj(ast_node_embeddings)

        # tree_node_embeddings = self.tree_layer(tree_node_embeddings, tree_children_index, batch_tree_mask)
        
        # batch_graph_node_ids = batch_ast_node_ids + batch_tree_ids
        # batch_graph_node_ids = np.argsort(batch_graph_node_ids)
        # if len(batch_ast_node_ids):
        #     graph_node_embeds = torch.cat((ast_node_embeddings, tree_node_embeddings), dim = 0)
        # else:
        #     graph_node_embeds = tree_node_embeddings
        # graph_node_embeds = graph_node_embeds[batch_graph_node_ids]

        # split_graph_node_embeds = torch.split(graph_node_embeds, num_graph_nodes)
        # code_token_embeds = self.get_code_token_embedddings(memory_bank, code_token_ids)
        # code_tree_embeds = self.get_node_graph_embeds_for_token(split_graph_node_embeds, code_tree_ids)
        # code_token_embeds = nn.utils.rnn.pad_sequence(code_token_embeds, batch_first = True)
        # code_tree_embeds = nn.utils.rnn.pad_sequence(code_tree_embeds, batch_first = True)
        # code_query_embeds = torch.cat((code_token_embeds, code_tree_embeds), dim = -1)
        # code_query_embeds = self.code_query_proj(code_query_embeds)
       
        # graph_node_embeds = {'node': graph_node_embeds}
        # graph_node_embeds = self.graph_layer(graph_node_embeds, graphs)['node']
        # graph_embeds = self.graph_aggr(graphs, graph_node_embeds).unsqueeze(1)
        # graphs.nodes['node'].data['ft'] = graph_node_embeds
        # graph_node_embeds = []
        # graph_num_nodes = []
        
        # masked_node_embeddings = []
        # for i, graph in enumerate(dgl.unbatch(graphs)):
        #     graph_node_embeds.append(graph.nodes['node'].data['ft'])
        #     graph_num_nodes.append(len(graph_node_embeds[-1]))
        #     masked_node_embeddings.append(graph_node_embeds[-1][masked_exp_ids[i]]) 
        #     # graph_embeds.append(self.graph_aggr(graph, out_graph_node_embeddings[-1]))
        # masked_node_embeddings = torch.stack(masked_node_embeddings, dim = 0).unsqueeze(1)

        # graph_node_embeds = nn.utils.rnn.pad_sequence(graph_node_embeds, batch_first = True)
        
        # graph_node_mask = ~sequence_mask(torch.tensor(graph_num_nodes), max_len = max(graph_num_nodes)).unsqueeze(1)
        # graph_node_mask = graph_node_mask.type_as(graph_node_embeds)
        # attn_code_embeds = self.cross_attention_code_graph(key = graph_node_embeds, value = graph_node_embeds, query = code_query_embeds, mask = graph_node_mask)[0]
        # final_encoder_states = self.fused_gating_encoder(code_token_embeds, attn_code_embeds, graph_embeds)
        final_encoder_states, source_mask = self.encode(node_token_ids,
                                node_type_ids,
                                code_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size, 
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids)
        # embed and encode the target sequence
        tgt_token_emb = self.embedder(token_tgt_seq,
                                 mode='decoder')
        # tgt_mask = sequence_mask(tgt_len, max_len = tgt_token_emb.size(1))
        tgt_mask = token_tgt_seq != 0
        decoder_outputs = self.t5_model.decode(tgt_token_emb, final_encoder_states, source_mask, tgt_mask)

        # tgt_pad_mask = ~sequence_mask(tgt_len, max_len = tgt_token_emb.size(1))
        # enc_outputs = final_encoder_states
        # layer_wise_dec_out, _ = self.decoder(enc_outputs,
        #                                          code_len,
        #                                          tgt_pad_mask,
        #                                          tgt_token_emb)
        # token_decoder_outputs = layer_wise_dec_out[-1]
        # token_decoder_output_context = token_decoder_outputs[:, 1:2]

        # token_decoder_outputs = self.decoder_gating(token_decoder_outputs, token_decoder_output_context, graph_embeds, masked_node_embeddings)
        token_target = token_tgt_seq[:, 1:].contiguous()
        token_decoder_outputs = decoder_outputs[0]

        if self._copy:
            raise NotImplementedError
        else:
            token_decoder_outputs = token_decoder_outputs * (512 ** -0.5)
            token_scores = self.t5_model.lm_head(token_decoder_outputs)
            # token_scores = self.token_generator(token_decoder_outputs)  # `batch x tgt_len x vocab_siz`
            token_scores = token_scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`

            ml_token_loss = nn.CrossEntropyLoss(ignore_index=0)(token_scores.view(-1, token_scores.size(2)),
                                     token_target.view(-1))
            # ml_token_loss = self.criterion(token_scores.view(-1, token_scores.size(2)),
                                    # token_target.view(-1))
           
        return ml_token_loss

    def forward(self,
                node_token_ids,
                node_type_ids,
                code_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size,
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids,
                token_tgt_seq,
                tgt_len, 
                src_map,
                alignment,
                **kwargs):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        if self.training:
            return self._run_forward_ml(node_token_ids,
                                        node_type_ids,
                                        code_len,
                                        batch_ast_node_ids, 
                                        batch_ast_dfs_ids,
                                        batch_tree_ids,
                                        batch_tree_node_ids, 
                                        batch_tree_mask,
                                        batch_tree_size,
                                        tree_children_index,
                                        graphs, 
                                        num_graph_nodes,
                                        code_token_ids,
                                        code_tree_ids,
                                        token_tgt_seq,
                                        tgt_len, 
                                        src_map,
                                        alignment,
                                        **kwargs)

        else:
            return self.decode(node_token_ids,
                                node_type_ids,
                                code_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size,
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids,
                            src_map,
                            alignment,
                               **kwargs)

    def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def __generate_sequence(self,
                            params,
                            choice='greedy',
                            tgt_words=None):

        batch_size = params['memory_bank'].size(0)
        use_cuda = params['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        tgt_chars = None

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = params['memory_bank'][0].shape[1] \
            if isinstance(params['memory_bank'], list) else params['memory_bank'].shape[1]
        dec_states = self.decoder.init_decoder(params['src_len'], max_mem_len)

        attns = {"coverage": None}
        enc_outputs = params['layer_wise_outputs'] if self.layer_wise_attn \
            else params['memory_bank']

        # +1 for <EOS> token

        for idx in range(params['max_len'] + 1):
            tgt = self.embedder(tgt_words,
                                mode='decoder',
                                step=idx)

            tgt_pad_mask = tgt_words.data.eq(PAD)
            layer_wise_dec_out, attns = self.decoder.decode(tgt_pad_mask,
                                                            tgt,
                                                            enc_outputs,
                                                            dec_states,
                                                            step=idx,
                                                            layer_wise_coverage=attns['coverage'])
            decoder_outputs = layer_wise_dec_out[-1]
            acc_dec_outs.append(decoder_outputs.squeeze(1))
            if self._copy:
                raise NotImplementedError
            else:
                prediction = self.token_generator(decoder_outputs.squeeze(1))
                prediction = F.softmax(prediction, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction.unsqueeze(1))
            else:
                assert False

            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))
            if self._copy:
                mask = tgt.gt(len(params['tgt_dict']) - 1)
                copy_info.append(mask.float().squeeze(1))

            # words = self.__tens2sent(tgt, params['tgt_dict'], params['source_vocab'])
            # tgt_chars = None


            # words = [params['tgt_dict'][w] for w in words]
            # words = torch.Tensor(words).type_as(tgt)
            tgt_words = tgt#tgt.unsqueeze(1)
        return dec_preds, attentions, copy_info, dec_log_probs

    def decode(self,
                node_token_ids,
                node_type_ids,
                code_len,
                batch_ast_node_ids, 
                batch_ast_dfs_ids,
                batch_tree_ids,
                batch_tree_node_ids, 
                batch_tree_mask,
                batch_tree_size,
                tree_children_index,
                graphs, 
                num_graph_nodes,
                code_token_ids,
                code_tree_ids,
                src_map,
                alignment,
               **kwargs):

        # batch_size = code_len.size(0)
        # # embed and encode the source sequence
        # code_rep = self.embedder(node_token_ids,
        #                          node_type_ids,
        #                          mode='encoder')
        # memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # B x seq_len x h

        # tree_node_embeddings = self.get_tree_node_embedddings(memory_bank.reshape(1, -1, memory_bank.size(-1)), batch_tree_node_ids.unsqueeze(0)).squeeze(0)
        # ast_node_embeddings = self.get_ast_node_embedddings(memory_bank, batch_ast_dfs_ids)

        
        # batch_ast_node_ids = reduce(list.__add__, batch_ast_node_ids)
        # if len(batch_ast_node_ids):
        #     ast_node_embeddings = torch.cat(ast_node_embeddings, dim= 0)
        #     ast_node_embeddings = self.ast_node_proj(ast_node_embeddings)

        # tree_node_embeddings = self.tree_layer(tree_node_embeddings, tree_children_index, batch_tree_mask)
        
        # batch_graph_node_ids = batch_ast_node_ids + batch_tree_ids
        # batch_graph_node_ids = np.argsort(batch_graph_node_ids)
        # if len(batch_ast_node_ids):
        #     graph_node_embeds = torch.cat((ast_node_embeddings, tree_node_embeddings), dim = 0)
        # else:
        #     graph_node_embeds = tree_node_embeddings
        # graph_node_embeds = graph_node_embeds[batch_graph_node_ids]

        # split_graph_node_embeds = torch.split(graph_node_embeds, num_graph_nodes)
        # code_token_embeds = self.get_code_token_embedddings(memory_bank, code_token_ids)
        # code_tree_embeds = self.get_node_graph_embeds_for_token(split_graph_node_embeds, code_tree_ids)
        # code_token_embeds = nn.utils.rnn.pad_sequence(code_token_embeds, batch_first = True)
        # code_tree_embeds = nn.utils.rnn.pad_sequence(code_tree_embeds, batch_first = True)
        # code_query_embeds = torch.cat((code_token_embeds, code_tree_embeds), dim = -1)
        # code_query_embeds = self.code_query_proj(code_query_embeds)
       
        # graph_node_embeds = {'node': graph_node_embeds}
        # graph_node_embeds = self.graph_layer(graph_node_embeds, graphs)['node']
        # graph_embeds = self.graph_aggr(graphs, graph_node_embeds).unsqueeze(1)
        # graphs.nodes['node'].data['ft'] = graph_node_embeds
        # graph_node_embeds = []
        # graph_num_nodes = []
        
        # masked_node_embeddings = []
        # for i, graph in enumerate(dgl.unbatch(graphs)):
        #     graph_node_embeds.append(graph.nodes['node'].data['ft'])
        #     graph_num_nodes.append(len(graph_node_embeds[-1]))
        #     masked_node_embeddings.append(graph_node_embeds[-1][masked_exp_ids[i]]) 
        #     # graph_embeds.append(self.graph_aggr(graph, out_graph_node_embeddings[-1]))
        # masked_node_embeddings = torch.stack(masked_node_embeddings, dim = 0).unsqueeze(1)

        # graph_node_embeds = nn.utils.rnn.pad_sequence(graph_node_embeds, batch_first = True)
        
        # graph_node_mask = ~sequence_mask(torch.tensor(graph_num_nodes), max_len = max(graph_num_nodes)).unsqueeze(1)
        # graph_node_mask = graph_node_mask.type_as(graph_node_embeds)
        # attn_code_embeds = self.cross_attention_code_graph(key = graph_node_embeds, value = graph_node_embeds, query = code_query_embeds, mask = graph_node_mask)[0]
        # final_encoder_states = self.fused_gating_encoder(code_token_embeds, attn_code_embeds, graph_embeds)

        final_encoder_states, graph_embeds, masked_node_embeddings = self.encode(node_token_ids,
                                node_type_ids,
                                code_len,
                                batch_ast_node_ids, 
                                batch_ast_dfs_ids,
                                batch_tree_ids,
                                batch_tree_node_ids, 
                                batch_tree_mask,
                                batch_tree_size, 
                                tree_children_index,
                                graphs, 
                                num_graph_nodes,
                                code_token_ids,
                                code_tree_ids)
        memory_bank = final_encoder_states
        params = dict()
        params['memory_bank'] = final_encoder_states
        # params['layer_wise_outputs'] = layer_wise_outputs
        params['src_len'] = code_len
        # params['source_vocab'] = kwargs['source_vocab']
        params['src_map'] = src_map
        # params['src_mask'] = kwargs['code_mask_rep']
        # params['fill'] = kwargs['fill']
        # params['blank'] = kwargs['blank']
        # params['src_dict'] = kwargs['src_dict']
        # params['tgt_dict'] = kwargs['tgt_dict']
        params['max_len'] = 42#kwargs['max_len']
        # params['src_words'] = code_word_rep

        dec_preds, attentions, copy_info, _ = self.__generate_sequence(params, choice='greedy')
        dec_preds = torch.stack(dec_preds, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        # attentions: batch_size x tgt_len x num_heads x src_len
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return dec_preds, copy_info, memory_bank, attentions
        # return {
        #     'predictions': dec_preds,
        #     'copy_info': copy_info,
        #     'memory_bank': memory_bank,
        #     'attentions': attentions
        # }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
