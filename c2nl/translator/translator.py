#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import torch
import torch.nn as nn
import dgl
import numpy as np
import torch.nn.functional as f
from functools import reduce
from c2nl.translator.beam import Beam
from c2nl.inputters import constants


class Translator(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
    """

    def __init__(self,
                 model,
                 use_gpu,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 replace_unk=False):

        self.use_gpu = use_gpu
        self.model = model
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.replace_unk = replace_unk

    def translate_batch(self, batch_inputs):
        # Eval mode
        self.model.eval()

        node_type_ids = batch_inputs['node_type_ids']
        node_token_ids = batch_inputs['node_token_ids']
        code_len = batch_inputs['code_len']
        code_token_len = batch_inputs['code_token_len']
        batch_ast_node_ids = batch_inputs['batch_ast_node_ids']
        batch_ast_dfs_ids = batch_inputs['batch_ast_dfs_ids']
        batch_tree_ids = batch_inputs['batch_tree_ids']
        batch_tree_node_ids = batch_inputs['batch_tree_node_ids']
        batch_tree_mask = batch_inputs['batch_tree_mask']
        batch_tree_size = batch_inputs['batch_tree_size']
        tree_children_index = batch_inputs['tree_children_index']
        code_token_ids = batch_inputs['code_token_ids']
        code_tree_ids = batch_inputs['code_tree_ids']
        # masked_exp_ids = batch_inputs['masked_exp_ids']
        graphs = batch_inputs['graphs']
        num_graph_nodes = batch_inputs['num_graph_nodes']
        batch_src_node_ids = batch_inputs['batch_src_node_ids']
        batch_src_token_len = batch_inputs['batch_src_token_len']
        batch_src_token_ids = batch_inputs['batch_src_token_ids']
        # code_token_ids = batch_inputs['code_token_ids']
        # fill = batch_inputs['fill']

        beam_size = self.beam_size
        batch_size = code_len.size(0)

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([self.model.tgt_dict[t]
                                for t in self.ignore_when_blocking])
    
        beam = [Beam(beam_size,
                     n_best=self.n_best,
                     cuda=self.use_gpu,
                     global_scorer=self.global_scorer,
                     pad=constants.PAD,
                     eos=constants.EOS,
                     bos=constants.BOS,
                     min_length=self.min_length,
                     stepwise_penalty=self.stepwise_penalty,
                     block_ngram_repeat=self.block_ngram_repeat,
                     exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a)

        def rvar(a):
            return var(a.repeat(beam_size, 1, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        final_encoder_states, graph_embeds, graph_node_embeds, graph_num_nodes, src_token_states = self.model.encode(node_token_ids,
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
        memory_bank = final_encoder_states
        src_lens = code_token_len.repeat(beam_size)
        subtree_lens = graph_num_nodes.repeat(beam_size)
        # print(code_len.shape, src_lens.shape)
        # graph_node_embeds = graph_node_embeds.repeat(beam_size)
        dec_states = self.model.decoder.init_decoder(src_lens, memory_bank.shape[1], subtree_lens, graph_node_embeds.shape[1])

        src_lengths = code_token_len
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank) \
                .long() \
                .fill_(memory_bank.size(1))

        # (2) Repeat src objects `beam_size` times.
       
        memory_bank = rvar(memory_bank.data)
        graph_node_embeds = rvar(graph_node_embeds.data)
        if src_token_states is not None:
            src_token_states = rvar(src_token_states.data)

        if len(batch_src_token_len):
            batch_src_token_len = batch_src_token_len.repeat(beam_size)
            batch_src_token_ids = batch_src_token_ids.repeat(beam_size, 1)
        memory_lengths = src_lengths.repeat(beam_size)
        # if code_mask_rep is not None:
        #     code_mask_rep = code_mask_rep.repeat(beam_size, 1)
        src_map = None
        attn = {"coverage": None}

        # (3) run the decoder to generate sentences, using beam search.
        cls_hidden_state = None
        for i in range(self.max_length + 1):
            if all((b.done for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.get_current_state() for b in beam])
            # Making it beam_size x batch_size and then apply view
            inp = var(inp.t().contiguous().view(-1, 1))

            # Turn any copied words to UNKs
            # if self.copy_attn:
            #     inp = inp.masked_fill(inp.gt(len(self.model.tgt_dict) - 1),
            #                           constants.UNK)

            inp_chars = None
        
            
            tgt = self.model.embedder(inp, inp_chars, mode='decoder', step=i)
            # Run one step.
            # applicable for Transformer
            tgt_pad_mask = inp.data.eq(constants.PAD)
            layer_wise_dec_out, attn = self.model.decoder.decode(tgt_pad_mask,
                                                        tgt,
                                                        memory_bank,
                                                        dec_states,
                                                        graph_node_embeds,
                                                        step=i,
                                                        layer_wise_coverage=attn['coverage'])

            dec_out = layer_wise_dec_out[-1]
            # attn["std"] is a list (of size num_heads),
            # so we pick the attention from first head
            attn["std"] = attn["std"][0]
            if "code_attn" in attn:
                attn["code_attn"] = attn["code_attn"][0]
            
            if "ast_attn" in attn:
                attn["ast_attn"] = attn["ast_attn"][0]

            # (b) Compute a vector of batch x beam word scores.
            if self.copy_attn:
                _, copy_attn, _ = self.model.global_attn_layer(dec_out, src_token_states,
                                                    memory_lengths = batch_src_token_len,
                                                    softmax_weights = True)
                out = self.model.copy_layer(dec_out, copy_attn, batch_src_token_ids)
                out = unbottle(out.squeeze(-1))
                beam_attn = unbottle(copy_attn.squeeze(1))
            else:
                # if cls_hidden_state is not None:
                #     dec_out = self.model.decoder_gating(dec_out, cls_hidden_state, graph_embeds, masked_node_embeddings)
                out = self.model.token_generator(dec_out.squeeze(1))
            
                # beam x batch_size x tgt_vocab
                out = unbottle(f.softmax(out, dim=1))
                # beam x batch_size x tgt_vocab
                beam_attn = unbottle(attn["std"].squeeze(1))

            out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if not b.done:
                    b.advance(out[:, j],
                              beam_attn.data[:, j, :memory_lengths[j]])

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(torch.tensor(hyps).cpu().numpy())
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret
