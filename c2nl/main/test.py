# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py
import sys
sys.path.append(".")
sys.path.append("..")
from c2nl.models.codesummodel import CodeSumModel
from data.node_tokenizer import NodeTokenizer
from c2nl.translator.translation import TranslationBuilder
from c2nl.translator.beam import GNMTGlobalScorer
from c2nl.translator.translator import Translator
from c2nl.eval.meteor import Meteor
from c2nl.eval.rouge import Rouge
from c2nl.eval.cider import Cider
from c2nl.eval.bleu import Bleu, nltk_corpus_bleu, corpus_bleu
from c2nl.inputters import constants    
from c2nl.inputters.timer import AverageMeter, Timer
from c2nl.utils.copy_utils import collapse_copy_scores, make_src_map, align
from collections import OrderedDict
from tqdm import tqdm  
import c2nl.inputters.dataset as data
import c2nl.inputters.vector as vector
import c2nl.inputters.utils as util
from c2nl.vocab.get_overlap_vocab import get_overlap_vocab
from inputters import EOS
import c2nl.config as config
import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from types import SimpleNamespace
import numpy as np
import argparse
import subprocess
import logging
import torch
import json
import os
from collections import OrderedDict, Counter
from data.custom_dataset import CustomDataset, CustomFileDataset, custom_collate



logger = logging.getLogger()

def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))

def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1
def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_test_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--device_index', type=int, required=True)
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--config_path', type=str, required=True)
    # Files
    files = parser.add_argument_group('Filesystem')
    # files.add_argument('--dataset_name', nargs='+', type=str, required=True,
    #                    help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    # files.add_argument('--model_name', type=str, default='',
    #                    help='Unique model identifier (.mdl, .txt, .checkpoint)')
    # files.add_argument('--data_dir', type=str, default='/data/',
    #                    help='Directory of training/validation data')
    # files.add_argument('--dev_src', nargs='+', type=str, required=True,
    #                    help='Preprocessed dev source file')
    # files.add_argument('--dev_src_tag', nargs='+', type=str,
    #                    help='Preprocessed dev source tag file')
    # files.add_argument('--dev_tgt', nargs='+', type=str,
    #                    help='Preprocessed dev target file')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_generate', type='bool', default=False,
                         help='Only generate code summaries')

    # Beam Search
    bsearch = parser.add_argument_group('Beam Search arguments')
    bsearch.add_argument('--beam_size', type=int, default=4,
                         help='Set the beam size (=1 means greedy decoding)')
    bsearch.add_argument('--n_best', type=int, default=1,
                         help="""If verbose is set, will output the n_best
                           decoded sentences""")
    bsearch.add_argument('--stepwise_penalty', type='bool', default=False,
                         help="""Apply penalty at every decoding step.
                           Helpful for summary penalty.""")
    bsearch.add_argument('--length_penalty', default='none',
                         choices=['none', 'wu', 'avg'],
                         help="""Length Penalty to use.""")
    bsearch.add_argument('--coverage_penalty', default='none',
                         choices=['none', 'wu', 'summary'],
                         help="""Coverage Penalty to use.""")
    bsearch.add_argument('--block_ngram_repeat', type=int, default=0,
                         help='Block repetition of ngrams during decoding.')
    bsearch.add_argument('--ignore_when_blocking', nargs='+', type=str,
                         default=[],
                         help="""Ignore these strings when blocking repeats.
                           You want to block sentence delimiters.""")
    bsearch.add_argument('--gamma', type=float, default=0.,
                         help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    bsearch.add_argument('--beta', type=float, default=0.,
                         help="""Coverage penalty parameter""")
    bsearch.add_argument('--replace_unk', action="store_true",
                         help="""Replace the generated UNK tokens with the
                           source token that had highest attention weight. If
                           phrase_table is provided, it will lookup the
                           identified source token and give the corresponding
                           target token. If it is not provided(or the identified
                           source token does not exist in the table) then it
                           will copy the source token""")
    bsearch.add_argument('--verbose', action="store_true",
                         help='Print scores and predictions for each sentence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_src_files = []
    args.dev_tgt_files = []
    args.dev_src_tag_files = []

  

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    # Set log + model file names
    args.model_name = "code-summ"
    args.log_file = os.path.join(args.model_dir, args.model_name + '_beam.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.pred_file = os.path.join(
        args.model_dir, args.model_name + '_beam.json')


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------

def build_translator(model, args):
    scorer = GNMTGlobalScorer(args.gamma,
                              args.beta,
                              args.coverage_penalty,
                              args.length_penalty)

    translator = Translator(model,
                            args.cuda,
                            args.beam_size,
                            n_best=args.n_best,
                            max_length=args.max_tgt_len, 
                            copy_attn=0,
                            global_scorer=scorer,
                            min_length=3,
                            stepwise_penalty=args.stepwise_penalty,
                            block_ngram_repeat=args.block_ngram_repeat,
                            ignore_when_blocking=args.ignore_when_blocking,
                            replace_unk=args.replace_unk)
    return translator


def convert_into_device(device, **ex):
    return {k: v.to(device) if not isinstance(v,  (list, tuple, int)) else v for k, v in ex.items()}


def validate_official(args, data_loader, model, device, tokenizer):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """

    eval_time = Timer()
    translator = build_translator(model, args)
    # builder = TranslationBuilder(model.tgt_dict,
    #                              n_best=args.n_best,
    #                              replace_unk=args.replace_unk)

    # Run through examples
    examples = 0
    trans_dict, sources = dict(), dict()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        text_preds = []
        targets = []
        for batch_no, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ids = list(range(batch_no * batch_size,
                             (batch_no * batch_size) + batch_size))

            ex = convert_into_device(device, **ex)
            ret = translator.translate_batch(ex)

            predictions = ret['predictions']
            # predictions = torch.tensor(ret['predictions']).cpu().numpy()

            for i in range(len(predictions)):
                _preds = []
                for pred in predictions[i]:
                    pred = pred.tolist()
                    if EOS in pred:
                        eos_idx = pred.index(EOS)
                        pred = pred[:eos_idx + 1]
                    _preds.append(tokenizer.decode(pred))
                text_preds.append(_preds)
            targets.extend(tokenizer.decode_batch(
                ex['token_tgt_seq'].cpu().numpy()))

            examples += batch_size
    hypotheses, references = dict(), dict()
    for i in range(len(text_preds)):
        hypotheses[i] = text_preds[i]
        references[i] = [targets[i]]

    if args.only_generate:
        with open(args.pred_file, 'w') as fw:
            json.dump(hypotheses, fw, indent=4)

    else:
        bleu, rouge_l, meteor, precision, recall, f1, em, ind_bleu, ind_rouge = \
            eval_accuracies(hypotheses, references)
        logger.info('beam evaluation official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) + 'em = %.2f | ' % em +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

        with open(args.pred_file, 'w') as fw:
            for eid, translation in trans_dict.items():
                out_dict = OrderedDict()
                out_dict['id'] = eid
                out_dict['code'] = sources[eid]
                # printing all beam search predictions
                out_dict['predictions'] = [
                    ' '.join(pred) for pred in translation.pred_sents]
                out_dict['references'] = references[eid]
                out_dict['bleu'] = ind_bleu[eid]
                out_dict['rouge_l'] = ind_rouge[eid]
                fw.write(json.dumps(out_dict) + '\n')

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('Datasets/emse/text-sum-tokenizer-bpe-esme.json')
def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    cider_calculator = Cider()
    cider, _ = cider_calculator.compute_score(references, hypotheses)

    print('CIDER', cider)

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    em = AverageMeter()
    for key in references.keys():
        # print(hypotheses[key][0], references[key])
        _prec, _rec, _f1 = compute_eval_score(
            hypotheses[key][0], references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        em.update(hypotheses[key][0].replace(' ', '') ==
                  references[key][0].replace(' ', ''))
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
        recall.avg * 100, f1.avg * 100, em.avg * 100, ind_bleu, ind_rouge


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------
def parse_config(data):
    if type(data) is list:
        return list(map(parse_config, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, parse_config(value))
        return sns
    else:
        return data


def load_local_cp(path, model, device):
    checkpoint = torch.load(path, device)
    model.load_state_dict(checkpoint['model'])
    return model


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')
    configuration = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    configuration = parse_config(configuration)
    src_vocab = json.load(open(configuration.vocab.code_tokenizer))['model']['vocab']
    tgt_vocab = json.load(open(configuration.vocab.text_tokenizer))['model']['vocab']
    overlap_vocab = get_overlap_vocab(src_vocab, tgt_vocab)
    device = torch.device(
        f'cuda:{args.device_index}') if args.cuda else torch.device('cpu')
   
    all_cps = [f'cp_{epoch}.tar' for epoch in range(30, 100, 1)]
    for cp in all_cps:
        model = CodeSumModel(configuration).to(device)
        model = load_local_cp(
             f'checkpoints/emse/{cp}', model, device)

        code_tokenizer = Tokenizer.from_file(
            configuration.vocab.code_tokenizer)
        text_tokenizer = Tokenizer.from_file(
            configuration.vocab.text_tokenizer)
        code_tokenizer.decoder = BPEDecoder()
        text_tokenizer.decoder = BPEDecoder()
        node_tokenizer = NodeTokenizer(
            configuration.vocab.node_type, code_tokenizer)

        data_path_lst = 'Datasets/emse/processed/test'
        valid_dataset = CustomFileDataset(data_path_lst, node_tokenizer = node_tokenizer, code_tokenizer = code_tokenizer, text_tokenizer = text_tokenizer, src2tgt = overlap_vocab)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=16,
            num_workers=args.data_workers,
            collate_fn=custom_collate,
        )
        # --------------------------------------------------------------------------
        print('=' * 100)
        print('CP', cp)
        validate_official(args, valid_loader, model, device, text_tokenizer)
        print('=' * 100)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_test_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler('test-code-sum.log', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
