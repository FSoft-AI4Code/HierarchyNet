from tokenizers import Tokenizer

def get_overlap_vocab(src_vocab, tgt_vocab):
    overlap_tokens = set(src_vocab.keys()) & set(tgt_vocab.keys())
    src2tgt = {src_vocab[token] : tgt_vocab[token] for token in overlap_tokens}
    return src2tgt