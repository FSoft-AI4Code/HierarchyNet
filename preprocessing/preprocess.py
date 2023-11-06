import glob
import jsonlines
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Strip, Lowercase, NFD, StripAccents
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tree_sitter import Language, Parser
from tqdm import tqdm
import re

def remove_str(text):
    text = re.sub(r'"[\s\S]*?"', ' "<str>" ', text)
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'//.*?\n', '', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    return text
def pretokenize(tree, encoded_text):
    if str(tree.type) in ['comment', 'line_comment', 'block_comment']: return ""
    if tree.type == 'string_literal': return '"<str>"'
    num_children = len(tree.children)
    if num_children > 0:
        codes = [pretokenize(child, encoded_text) for child in tree.children]
        return " ".join(codes)
    else:
        return encoded_text[tree.start_byte:tree.end_byte].decode()