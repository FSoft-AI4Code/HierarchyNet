from tree_sitter import Parser, Language
from convert_java import convert_into_java_graph
import os
import jsonlines
import pickle
from functools import partial
from tqdm import tqdm
from tokenizers import Tokenizer
import json
import multiprocessing
from tokenizers.decoders import BPEDecoder

def save_one_sample(item, mode, code_tokenizer, text_tokenizer, base_path, lang_path, lang):
    line, idx = item
    base_path = f'{base_path}/{mode}/{idx}' 
    raw_code = line['code']
    parser = Parser()
    parser.set_language(Language(lang_path, lang))
    try:
        refactor_json, flattened_nodes, ast_node_ids, ast_dfs_ids, tree_stmts, stmt_indices, G = convert_into_java_graph(parser.parse(raw_code.encode()), raw_code, lambda x: code_tokenizer.encode(x).tokens)
    except: return
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with jsonlines.open(f'{base_path}/flattened_nodes.jsonl', 'w') as f:
        f.write_all(flattened_nodes)
    with jsonlines.open(f'{base_path}/stmts.jsonl', 'w') as f:
        f.write_all(tree_stmts)
    with open(f'{base_path}/ast_node_ids.txt', 'w') as f:
        f.write(' '.join(list(map(str, ast_node_ids))))
    with open(f'{base_path}/ast_dfs_ids.txt', 'w') as f:
        f.write(' '.join(list(map(str, ast_dfs_ids))))
        # f.write_all(tree_stmts)
    with open(f'{base_path}/stmt_indices.txt', 'w') as f:
        f.write(' '.join(list(map(str, stmt_indices))))
    with open(f'{base_path}/G.pickle', 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    target = line['comment'].lower()
    target_tokens = text_tokenizer.encode(target).tokens
    with open(f'{base_path}/target.json', 'w') as f:
        json.dump({
            'target_tokens': target_tokens,
            'target': target
        }, f)
    with open(f'{base_path}/metadata.json', 'w') as f:
        json.dump({
            'code': raw_code,
            'target': target
        }, f)

if __name__ == "__main__":
    code_tokenizer = Tokenizer.from_file('code-sum-tokenizer-funcom.json')
    text_tokenizer = Tokenizer.from_file('text-sum-tokenizer-funcom.json')
    # code_tokenizer.decoder = BPEDecoder()
    # text_tokenizer.decoder = BPEDecoder()
   
    for mode in ['test', 'train', 'valid']:
        paths = os.listdir(f'FunCom/processed-v1/final-{mode}')
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(partial(process_target, mode = mode, code_tokenizer = code_tokenizer, text_tokenizer = text_tokenizer), tqdm(paths))
        # print(mode, sum(counts))