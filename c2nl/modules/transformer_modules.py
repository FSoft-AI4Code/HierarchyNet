import torch
import torch.nn as nn
from encoders.transformer import TransformerEncoder
from decoders.transformer import TransformerDecoder, SerialTransformerDecoder


class Encoder(nn.Module):
    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(num_layers=args.num_layers.transformer_encoder,
                                              d_model=input_size,
                                              heads=args.num_heads.transformer_encoder,
                                              d_k=args.num_heads.d_k,
                                              d_v=args.num_heads.d_v,
                                              d_ff=args.num_heads.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                input,
                input_len):
        layer_outputs, _ = self.transformer(input, input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Decoder(nn.Module):
    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        if self.split_decoder:
            # Following (https://arxiv.org/pdf/1808.07913.pdf), we split decoder
            self.transformer_c = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )
            self.transformer_d = TransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                dropout=args.trans_drop
            )

            # To accomplish eq. 19 - 21 from `https://arxiv.org/pdf/1808.07913.pdf`
            self.fusion_sigmoid = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.ReLU()
            )
        else:
            self.transformer = TransformerDecoder(
                num_layers=args.num_layers.transformer_encoder,
                d_model=input_size,
                heads=args.num_heads.transformer_decoder,
                d_k=args.num_heads.d_k,
                d_v=args.num_heads.d_v,
                d_ff=args.num_heads.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )

        if args.reload_decoder_state:
            state_dict = torch.load(
                args.reload_decoder_state, map_location=lambda storage, loc: storage
            )
            self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        if self.split_decoder:
            return self.transformer_c.count_parameters() + self.transformer_d.count_parameters()
        else:
            return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len):

        if self.split_decoder:
            state_c = self.transformer_c.init_state(src_lens, max_src_len)
            state_d = self.transformer_d.init_state(src_lens, max_src_len)
            return state_c, state_d
        else:
            return self.transformer.init_state(src_lens, max_src_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               step=None,
               layer_wise_coverage=None):

        if self.split_decoder:
            copier_out, attns = self.transformer_c(tgt_words,
                                                   tgt_emb,
                                                   memory_bank,
                                                   state[0],
                                                   step=step,
                                                   layer_wise_coverage=layer_wise_coverage)
            dec_out, _ = self.transformer_d(tgt_words,
                                            tgt_emb,
                                            memory_bank,
                                            state[1],
                                            step=step)
            f_t = self.fusion_sigmoid(torch.cat([copier_out[-1], dec_out[-1]], dim=-1))
            gate_input = torch.cat([copier_out[-1], torch.mul(f_t, dec_out[-1])], dim=-1)
            decoder_outputs = self.fusion_gate(gate_input)
            decoder_outputs = [decoder_outputs]
        else:
            decoder_outputs, attns = self.transformer(tgt_words,
                                                      tgt_emb,
                                                      memory_bank,
                                                      state,
                                                      step=step,
                                                      layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                tgt_pad_mask,
                tgt_emb):

        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        state = self.init_decoder(memory_len, max_mem_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state)

class SerialDecoder(nn.Module):
    def __init__(self, args, input_size):
        super(SerialDecoder, self).__init__()

        self.input_size = input_size

        self.split_decoder = args.split_decoder and args.copy_attn
        if self.split_decoder:
            # Following (https://arxiv.org/pdf/1808.07913.pdf), we split decoder
            self.transformer_c = SerialTransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )
            self.transformer_d = SerialTransformerDecoder(
                num_layers=args.nlayers,
                d_model=self.input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                dropout=args.trans_drop
            )

            # To accomplish eq. 19 - 21 from `https://arxiv.org/pdf/1808.07913.pdf`
            self.fusion_sigmoid = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.ReLU()
            )
        else:
            self.transformer = SerialTransformerDecoder(
                num_layers=args.num_layers.transformer_encoder,
                d_model=input_size,
                heads=args.num_heads.transformer_decoder,
                d_k=args.num_heads.d_k,
                d_v=args.num_heads.d_v,
                d_ff=args.num_heads.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )

        if args.reload_decoder_state:
            state_dict = torch.load(
                args.reload_decoder_state, map_location=lambda storage, loc: storage
            )
            self.decoder.load_state_dict(state_dict)

    def count_parameters(self):
        if self.split_decoder:
            return self.transformer_c.count_parameters() + self.transformer_d.count_parameters()
        else:
            return self.transformer.count_parameters()

    def init_decoder(self,
                     src_lens,
                     max_src_len,
                     subtree_len,
                     subtree_max_len):

        if self.split_decoder:
            state_c = self.transformer_c.init_state(src_lens, max_src_len, subtree_len, subtree_max_len)
            state_d = self.transformer_d.init_state(src_lens, max_src_len, subtree_len, subtree_max_len)
            return state_c, state_d
        else:
            return self.transformer.init_state(src_lens, max_src_len, subtree_len, subtree_max_len)

    def decode(self,
               tgt_words,
               tgt_emb,
               memory_bank,
               state,
               subtree_emb,
               step=None,
               layer_wise_coverage=None):

        if self.split_decoder:
            copier_out, attns = self.transformer_c(tgt_words,
                                                   tgt_emb,
                                                   memory_bank,
                                                   state[0],
                                                   subtree_emb,
                                                   step=step,
                                                   layer_wise_coverage=layer_wise_coverage)
            dec_out, _ = self.transformer_d(tgt_words,
                                            tgt_emb,
                                            memory_bank,
                                            state[1],
                                            subtree_emb,
                                            step=step)
            f_t = self.fusion_sigmoid(torch.cat([copier_out[-1], dec_out[-1]], dim=-1))
            gate_input = torch.cat([copier_out[-1], torch.mul(f_t, dec_out[-1])], dim=-1)
            decoder_outputs = self.fusion_gate(gate_input)
            decoder_outputs = [decoder_outputs]
        else:
            decoder_outputs, attns = self.transformer(tgt_words,
                                                      tgt_emb,
                                                      memory_bank,
                                                      state,
                                                      subtree_emb,
                                                      step=step,
                                                      layer_wise_coverage=layer_wise_coverage)

        return decoder_outputs, attns

    def forward(self,
                memory_bank,
                memory_len,
                subtree_emb,
                subtree_len,
                tgt_pad_mask,
                tgt_emb):
        max_mem_len = memory_bank[0].shape[1] \
            if isinstance(memory_bank, list) else memory_bank.shape[1]
        subtree_max_len = subtree_emb.shape[1]
        state = self.init_decoder(memory_len, max_mem_len, subtree_len, subtree_max_len)
        return self.decode(tgt_pad_mask, tgt_emb, memory_bank, state, subtree_emb)