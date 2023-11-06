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


if __name__ == "__main__":
    import json
    data = []
    parser = Parser()
    parser.set_language(Language('my-languages.so', 'java'))   
    total_numbers = 0
    error_numbers = 0
    data = json.load(open('train.json'))
    targets = []
    sources = []
    for v in data.values():
        targets.append(v['comment'].lower())
        sources.append(v['code'])
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>"], min_frequency = 2, end_of_word_suffix = '</w>', vocab_size = 30000)
   
    tokenizer.pre_tokenizer = Whitespace()
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Strip()])
    tokenizer.normalizer = normalizer

    tokenizer.train_from_iterator(targets, trainer=trainer)
    path = "text-sum-tokenizer.json"
    tokenizer.save(path)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>", '"<str>"', '<empty>'], min_frequency = 2, end_of_word_suffix = '</w>', vocab_size = 30000)
   
    tokenizer.pre_tokenizer = Whitespace()
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Strip()])
    tokenizer.normalizer = normalizer

    tokenizer.train_from_iterator(targets, trainer=trainer)
    path = "code-sum-tokenizer.json"
    tokenizer.save(path)