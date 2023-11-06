# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import torch.nn as nn
import logging
import subprocess
import argparse
import numpy as np
from datetime import datetime
from tree_sitter import Language, Parser
import yaml
from types import SimpleNamespace
from tokenizers import Tokenizer
from transformers import get_linear_schedule_with_warmup
from inputters import EOS
import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data.custom_dataset import CustomDataset, CustomFileDataset, UniCustomFileDataset, custom_collate, DistSortedBatchSampler
from data.node_tokenizer import NodeTokenizer

import torch
torch.nn

import c2nl.config as config
import c2nl.inputters.utils as util
from c2nl.inputters import constants

from collections import OrderedDict, Counter
from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data


from c2nl.models.codesummodel import CodeSumModel
from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor
from utils.log import Log
import wandb
import math
from c2nl.translator.translator import Translator
from c2nl.translator.beam import GNMTGlobalScorer
from c2nl.vocab.get_overlap_vocab import get_overlap_vocab

logger = logging.getLogger()

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def build_translator(model, args):
    scorer = GNMTGlobalScorer(0,
                              0,
                              None,
                              None)


    translator = Translator(model,
                            args.cuda,
                            1,
                            n_best=1,
                            max_length=50,
                            copy_attn=False,
                            global_scorer=scorer,
                            min_length=0,
                            stepwise_penalty=False,
                            block_ngram_repeat=3,
                            ignore_when_blocking='',
                            replace_unk=False)
    return translator
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
                _preds= []
                for pred in predictions[i]:
                    pred = pred.tolist()
                    if EOS in pred:
                        eos_idx = pred.index(EOS)
                        pred = pred[:eos_idx + 1]
                    _preds.append(tokenizer.decode(pred))
                text_preds.append(_preds)
            targets.extend(tokenizer.decode_batch(ex['token_tgt_seq'].cpu().numpy()))
            # print('&'*50)
            # print(targets)
            # targets = [[summ] for summ in ex['summ_text']]
            # translations = builder.from_batch(ret,
            #                                   ex['code_tokens'],
            #                                   targets,
            #                                   ex['src_vocab'])

            # src_sequences = [code for code in ex['code_text']]
            # for eid, trans, src in zip(ids, translations, src_sequences):
            #     trans_dict[eid] = trans
            #     sources[eid] = src

            examples += batch_size
    # print('text_preds', text_preds)
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
                out_dict['predictions'] = [' '.join(pred) for pred in translation.pred_sents]
                out_dict['references'] = references[eid]
                out_dict['bleu'] = ind_bleu[eid]
                out_dict['rouge_l'] = ind_rouge[eid]
                fw.write(json.dumps(out_dict) + '\n')

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--cuda', type = int, default = 1)
    runtime.add_argument('--device_index', type = int, default = 0)
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--world_size', type=int, default=1)
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--project', type = str, required = True)
    runtime.add_argument('--config_path', type = str, required = True)
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/',
                       help='Directory of training/validation data')
    files.add_argument('--train_src', nargs='+', type=str,
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', nargs='+', type=str,
                       help='Preprocessed train source tag file')
    files.add_argument('--train_tgt', nargs='+', type=str,
                       help='Preprocessed train target file')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=None,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if not args.only_test:
        args.train_src_files = []
        args.train_tgt_files = []
        args.train_src_tag_files = []

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
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
def init_from_scratch(args):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    model = CodeSumModel(args)

    return model

def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location = device)
    ddp_model = checkpoint['ddp_model']
    model.load_state_dict(ddp_model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

def load_local_cp(path, model, device):
    checkpoint = torch.load(path, device)
    model.load_state_dict(checkpoint['model'])
    return model

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12299"
    init_process_group(backend="nccl", rank = rank, world_size = world_size)

# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------
def save_checkpoint(base_path, epoch, model, optimizer, scheduler = None):
    path = os.path.join(base_path, f'cp_{epoch}.tar')
    saved_checkpoint_data = {
        'epoch': epoch,
        'ddp_model': model.state_dict(),
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        saved_checkpoint_data['scheduler'] = scheduler.state_dict()      
    torch.save(saved_checkpoint_data, path)
def save_local_cp(base_path, epoch, model):
    path = os.path.join(base_path, f'cp_local_{epoch}.tar')
    saved_checkpoint_data = {
        'epoch': epoch,
        'model': model.module.state_dict()
    }
    torch.save(saved_checkpoint_data, path)
def convert_into_device(device, **ex):
    return {k: v.to(device) if not isinstance(v,  (list, tuple, int)) else v for k, v in ex.items()}

import traceback
def train(rank, args, config, data_loader, model, optimizer,  device, global_stats, scheduler = None):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss_avg = AverageMeter()
    ml_token_loss_avg = AverageMeter()
    ml_type_loss_avg = AverageMeter()
    ml_cls_loss_avg = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)

    pbar.set_description("%s" % 'Epoch = %d []' %
                         current_epoch)
    # Run one epoch

    for idx, ex in enumerate(pbar):
        # bsz = ex['batch_size']
        # if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
        #     cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
        #     for param_group in model.optimizer.param_groups:
        #         param_group['lr'] = cur_lrate
        if idx and idx % 4000 == 0:
            if rank == 0:
                save_checkpoint(config.checkpoint.tmp, current_epoch, model, optimizer, scheduler)
        optimizer.zero_grad()
        ex = convert_into_device(device, **ex)
        net_loss = model(**ex)
        loss = net_loss['ml_loss']
        loss_per_token = net_loss['loss_per_token']

        loss_per_token = loss_per_token.item()
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
            
        if scheduler is not None: scheduler.step()
        ml_loss_avg.update(loss.detach().cpu().item())
        ml_token_loss_avg.update(loss.detach().cpu().item())
        log_info = 'Epoch = %d [perplexity = %.2f, token_loss = %.2f]' % \
                   (current_epoch, perplexity, loss)
        if rank == 0:
            wandb.log({
                'token_loss': loss,
                'perplexity': perplexity,
                'lr': optimizer.param_groups[0]['lr']
            })
        pbar.set_description("%s" % log_info)
   
    logger.info(f'[RANK = {rank}] ' + 'train: Epoch %d | total = %.2f | token_loss = %.2f | type_loss = %.2f | cls_loss: %.2f | Time for epoch = %.2f (s)' %
                (current_epoch, ml_loss_avg.avg, ml_token_loss_avg.avg, ml_type_loss_avg.avg, ml_cls_loss_avg.avg, epoch_time.time()))

    if rank == 0:
        wandb.log({
                'token_loss_avg': ml_token_loss_avg.avg,
                'total_loss_avg': ml_loss_avg.avg
            })
    # Checkpoint
    


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------
@torch.no_grad()
def validate(rank, args, data_loader, model, device, global_stats, mode='dev'):
    acc_token_loss, acc_perplexity, acc_cls_loss = 0, 0, 0
    count = 0
    model.train()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ex = convert_into_device(device, **ex)
            net_loss = model(**ex)
            loss = net_loss['ml_loss'].item()
            loss_per_token = net_loss['loss_per_token']

            loss_per_token = loss_per_token.item()
            loss_per_token = 10 if loss_per_token > 10 else loss_per_token
            perplexity = math.exp(loss_per_token)
            acc_token_loss += loss
            acc_perplexity += perplexity
            count += 1
    acc_token_loss /= count
    acc_perplexity /= count
    if rank == 0:
        wandb.log({
            'val token loss': acc_token_loss,
            'val perplexity': acc_perplexity
        })



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

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    em = AverageMeter()
    for key in references.keys():
        print(hypotheses[key][0], references[key])
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0], references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        em.update(hypotheses[key][0].replace(' ', '') == references[key][0].replace(' ', ''))
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100, em.avg * 100, ind_bleu, ind_rouge


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(rank, args):
    import torch
    import torch.nn as nn
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')   
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')
    configuration = yaml.load(open(args.config_path), Loader = yaml.FullLoader)
    configuration = parse_config(configuration)
    world_size = args.world_size
    
    if not os.path.exists(configuration.checkpoint.dir):
        os.makedirs(configuration.checkpoint.dir)
    if not os.path.exists(configuration.checkpoint.tmp):
        os.makedirs(configuration.checkpoint.tmp)
    if not args.only_test:
        ddp_setup(rank, world_size)
        if rank == 0:
            wandb.init(project = configuration.wandb.project, name = configuration.wandb.name)
        
    timestamp = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0

    device = torch.device(f'cuda:{rank}') if args.cuda else torch.device('cpu')

    model = init_from_scratch(configuration)
  
    model = model.to(device)

    code_tokenizer = Tokenizer.from_file(configuration.vocab.code_tokenizer)
    text_tokenizer = Tokenizer.from_file(configuration.vocab.text_tokenizer)
    node_tokenizer = NodeTokenizer(configuration.vocab.node_type, code_tokenizer)
    limit_tgt_len = configuration.limit_tgt_len
    src_vocab = json.load(open(configuration.vocab.code_tokenizer))['model']['vocab']
    tgt_vocab = json.load(open(configuration.vocab.text_tokenizer))['model']['vocab']
    overlap_vocab = None#get_overlap_vocab(src_vocab, tgt_vocab)
    if not args.only_test:
        model = DDP(model, device_ids=[rank], find_unused_parameters = True)

        data_path_lst = configuration.dataset_paths.train_lst
        

        train_dataset = CustomFileDataset(data_path_lst, node_tokenizer = node_tokenizer, code_tokenizer = code_tokenizer, text_tokenizer = text_tokenizer, src2tgt = overlap_vocab, limit_tgt_len = limit_tgt_len)
        train_bs = 64
        train_sampler = DistributedSampler(train_dataset) 

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_bs,
            num_workers=args.data_workers,
            collate_fn=custom_collate,
            sampler = train_sampler
        )
        optimizer = getattr(torch.optim, configuration.optimizer.name)(model.parameters(), **vars(configuration.optimizer.params))
        num_warmup_steps = configuration.scheduler.num_warmup_steps
        num_training_steps = configuration.scheduler.num_training_epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
      
    start_epoch += 1
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    

    data_path_lst = configuration.dataset_paths.test_lst
    valid_dataset = CustomFileDataset(data_path_lst, node_tokenizer = node_tokenizer, code_tokenizer = code_tokenizer, text_tokenizer = text_tokenizer, src2tgt = overlap_vocab, limit_tgt_len = limit_tgt_len)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=72,
        num_workers=args.data_workers,
        collate_fn=custom_collate,
        sampler = None if args.only_test else DistributedSampler(valid_dataset)
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if 0 and args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, valid_loader, model, device, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}
        torch.autograd.set_detect_anomaly(True)
        # validate(rank, args, valid_loader, model, device, stats)
        for epoch in range(start_epoch, 100):
            stats['epoch'] = epoch
            train_loader.sampler.set_epoch(epoch)
            train(rank, args, configuration, train_loader, model, optimizer, device, stats, scheduler = scheduler)
            if rank == 0:
                save_checkpoint(configuration.checkpoint.dir, epoch, model, optimizer, scheduler)
            validate(rank, args, valid_loader, model, device, stats)

        destroy_process_group()

if __name__ == '__main__':
    # Parse cmdline args and setup environment
    print('RUNNNNNNN')
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)
    
    # Set cuda

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
    configuration = yaml.load(open(args.config_path), Loader = yaml.FullLoader)
    
    if not os.path.exists(configuration['log']['file']):
        dirname = os.path.dirname(configuration['log']['file'])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        logfile = logging.FileHandler(configuration['log']['file'], 'w')
    else:
        logfile = logging.FileHandler(configuration['log']['file'], 'a')
    
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    # main(args)

    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    if args.only_test:
        main(args.device_index, args)
    else:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
