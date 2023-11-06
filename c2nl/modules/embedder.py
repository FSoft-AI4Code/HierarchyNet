import torch
import torch.nn as nn
# from modules.embeddings import Embeddings
from inputters import PAD

class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        # at least one of word or char embedding options should be True
    
        # self.use_src_word = config.use_src_word
        # self.use_tgt_word = config.use_tgt_word
        # if self.use_src_word:
        # self.src_word_embeddings = Embeddings(config.channels.in_dim,
        #                                         config.src_vocab_size,
        #                                         PAD)
            # self.enc_input_size += config.emsize
        # if self.use_tgt_word:
        # self.tgt_word_embeddings = Embeddings(config.channels.in_dim,
        #                                         config.tgt_vocab_size,
        #                                         PAD)
        self.token_embeddings = nn.Embedding(config.vocab_size.token,
                                            config.channels.in_dim,
                                            padding_idx = PAD)
            # self.dec_input_size += config.emsize

        self.type_embeddings = nn.Embedding(config.vocab_size.type,
                                            config.channels.in_dim,
                                            padding_idx = PAD)
        self.tgt_embeddings = nn.Embedding(config.vocab_size.tgt,
                                          config.channels.in_dim,
                                          padding_idx = PAD)

        self.no_relative_pos = False #all(v == 0 for v in config.max_relative_pos)

        # if self.src_pos_emb and self.no_relative_pos:
        self.src_pos_embeddings = nn.Embedding(config.max_src_len,
                                                config.channels.in_dim)
        self.src_norm = nn.LayerNorm(config.channels.in_dim)
        self.tgt_norm = nn.LayerNorm(config.channels.in_dim)
        self.dropout = nn.Dropout(0.2)
        # if self.tgt_pos_emb:
        self.tgt_pos_embeddings = nn.Embedding(config.max_tgt_len + 2,
                                                config.channels.out_dim)
        self.dim_size = config.channels.in_dim
        self.reset_parameters()
    def reset_parameters(self):
        self.token_embeddings.reset_parameters()
        self.type_embeddings.reset_parameters()
        self.tgt_embeddings.reset_parameters()
        nn.init.xavier_uniform_(self.src_pos_embeddings.weight)
        nn.init.xavier_uniform_(self.tgt_pos_embeddings.weight)
    def forward(self,
                sequence,
                sequence_type=None,
                mode='encoder',
                step=None):

        if mode == 'encoder':
            word_rep = self.token_embeddings(sequence).mean(dim = 2)
            num_words = torch.count_nonzero(sequence, dim = 2).unsqueeze(-1)
            mask = (sequence > 0).type_as(num_words).unsqueeze(-1)
            word_rep = self.token_embeddings(sequence) * mask
            word_rep = word_rep.sum(dim = 2) / num_words.clip(min = 1)
            # .mean(dim = 2)#sum(dim = 2) / num_words
            type_rep = self.type_embeddings(sequence_type)
            word_rep = word_rep + type_rep
            # pos_enc = torch.arange(start=0,
            #                         end=word_rep.size(1)).type(torch.LongTensor)
            # pos_enc = pos_enc.expand(*word_rep.size()[:-1])
            # pos_enc = pos_enc.type_as(word_rep).long()
            # pos_rep = self.src_pos_embeddings(pos_enc) * self.dim_size ** -0.5
            word_rep = word_rep #+ pos_rep
            word_rep = self.src_norm(word_rep)
        elif mode == 'decoder':
            word_rep = self.tgt_embeddings(sequence)  # B x P x d
            if step is None:
                pos_enc = torch.arange(start=0,
                                        end=word_rep.size(1)).type(torch.LongTensor)
            else:
                pos_enc = torch.LongTensor([step])  # used in inference time

            pos_enc = pos_enc.expand(*word_rep.size()[:-1])
            pos_enc = pos_enc.type_as(word_rep).long()
            pos_rep = self.tgt_pos_embeddings(pos_enc)
            word_rep = word_rep + pos_rep
            word_rep = self.tgt_norm(word_rep)
        else:
            raise ValueError('Unknown embedder mode!')
        word_rep = self.dropout(word_rep)
        return word_rep
