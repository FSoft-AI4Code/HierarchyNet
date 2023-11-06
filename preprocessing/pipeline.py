import jsonlines
import argparse
from tokenizers import Tokenizer
from tree_sitter import Parser, Language
import multiprocessing
from tqdm import tqdm
import os
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Strip, Lowercase, NFD, StripAccents
from tokenizers.trainers import BpeTrainer
from functools import partial
from preprocess import pretokenize
from save_data import save_one_sample

def mask_string_item(code, parser):
    encoded_code = code.encode()
    tree = parser.parse(encoded_code)
    new_code = pretokenize(tree.root_node, encoded_code)
    return new_code
def mask_string(path, lang_path, lang):
    data = []
    with jsonlines.open(path) as f:
        for line in f:
            data.append(line)
    parser = Parser()
    parser.set_language(Language(lang_path, lang))
    new_data = []
    for item in data:
        item['raw-code'] = item['code']
        new_code = mask_string_item(item['code'], parser)
        item['code'] = new_code
        new_data.append(item)
    with jsonlines.open(path, 'w') as f:
        f.write_all(data)

def train_tokenzier(data, tokenizer_path):
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)
    sources, targets = [], []
    for item in data:
        targets.append(item['comment'].lower())
        sources.append(item['code'])
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>"], min_frequency = 2, end_of_word_suffix = '</w>', vocab_size = 30000)
   
    tokenizer.pre_tokenizer = Whitespace()
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Strip()])
    tokenizer.normalizer = normalizer

    tokenizer.train_from_iterator(targets, trainer=trainer)
    path = f"{tokenizer_path}/text-sum-tokenizer.json"
    tokenizer.save(path)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>", '"<str>"', '<empty>'], min_frequency = 2, end_of_word_suffix = '</w>', vocab_size = 30000)
   
    tokenizer.pre_tokenizer = Whitespace()
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Strip()])
    tokenizer.normalizer = normalizer

    tokenizer.train_from_iterator(targets, trainer=trainer)
    path = f"{tokenizer_path}/code-sum-tokenizer.json"
    tokenizer.save(path)
def process_raw_data(data_path, lang_path, lang, has_train_tokenizer = True, tokenizer_path = None):
    mask_string(data_path, lang_path, lang)
    data = []
    with jsonlines.open(data_path) as f:
        for line in f:
            data.append(line)
    if has_train_tokenizer:
        assert tokenizer_path is not None
        train_tokenzier(data, tokenizer_path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-path', type = str)
    ap.add_argument('--val-path', type = str)
    ap.add_argument('--test-path', type = str)
    ap.add_argument('--tokenizer-path', type = str)
    ap.add_argument('--lang-path', type = str)
    ap.add_argument('--lang', type = str)
    ap.add_argument('--output-path', type = str)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    
    lang_path = args.lang_path
    lang = args.lang
    output_path = args.output_path
    for mode in ['train', 'val', 'test']:
        _path = eval(f'{mode}_path')
        if mode == 'train':
            has_train_tokenizer = True
            tokenizer_path = args.tokenizer_path
        else:
            has_train_tokenizer = False
        if _path is not None and os.path.exists(_path):
            process_raw_data(_path, lang_path, lang, has_train_tokenizer, tokenizer_path)

    text_tokenizer = Tokenizer.from_file(f'{tokenizer_path}/text-sum-tokenizer.json')
    code_tokenizer = Tokenizer.from_file(f'{tokenizer_path}/code-sum-tokenizer.json')
    cpu_count = multiprocessing.cpu_count()
    for mode in ['train', 'val', 'test']:
        _path = eval(f'{mode}_path')
        data = []
        with jsonlines.open(_path) as f:
            for line in f:
                data.append(line)
        data_items = list(zip(data, range(len(data))))
        with multiprocessing.Pool(cpu_count) as pool:
            pool.map(partial(save_one_sample, mode = mode, code_tokenizer = code_tokenizer, text_tokenizer = text_tokenizer, base_path = output_path, lang_path = lang_path, lang = lang), tqdm(data_items))
            





