#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime


import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


import torch.nn.functional as F
import numpy as np
import h5py
import time
# from optim_n2n import OptimN2N
from data import Dataset
# import utils
import logger
import math


# from preprocess_text import Indexer

# from torch.utils.tensorboard import SummaryWriter

import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict, Counter
from dataloader_bases import DataLoader
# from dgmvae import get_chat_tokenize

from sklearn.metrics.cluster import homogeneity_score
import subprocess



parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='data/ptb/ptb-train.hdf5')
parser.add_argument('--val_file', default='data/ptb/ptb-val.hdf5')
parser.add_argument('--test_file', default='data/ptb/ptb-test.hdf5')
parser.add_argument('--vocab_file', default='data/ptb/ptb.dict')
parser.add_argument('--train_from', default='')


# SRI options
parser.add_argument('--z_n_iters', type=int, default=20)
parser.add_argument('--z_step_size', type=float, default=0.5)
parser.add_argument('--z_with_noise', type=int, default=0)
parser.add_argument('--num_z_samples', type=int, default=10)


# EBM
parser.add_argument('--prior_hidden_dim', type=int, default=200)
parser.add_argument('--z_prior_with_noise', type=int, default=1)
parser.add_argument('--prior_step_size', type=float, default=0.5)
parser.add_argument('--z_n_iters_prior', type=int, default=40)
parser.add_argument('--max_grad_norm_prior', default=1, type=float)
parser.add_argument('--ebm_reg', default=0.001, type=float)
parser.add_argument('--ref_dist', default='gaussian', type=str, choices=['gaussian', 'uniform'])
parser.add_argument('--ref_sigma', type=float, default=1.)
parser.add_argument('--init_factor', type=float, default=1.)



# LM
parser.add_argument('--lm_lr', default=0.0001, type=float)
parser.add_argument('--revserse_lm_lr', default=0.0001, type=float)
parser.add_argument('--reverse_lm_num_epoch', type=int, default=8)
parser.add_argument('--lm_pretrain', type=int, default=0)
parser.add_argument('--pretrained_lm', type=str, default="output/012_ptb_lm_pretraining/2020-05-24-01-16-46-nll103.07/forward_lm.pt")
parser.add_argument('--reverse_lm_eval', type=int, default=1)



# Model options
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--dec_word_dim', default=512, type=int)
parser.add_argument('--dec_h_dim', default=512, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.2, type=float)
parser.add_argument('--model', default='abp', type=str, choices = ['abp', 'vae', 'autoreg', 'savae', 'svi'])
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--train_kl', default=1, type=int)

# Optimization options
parser.add_argument('--log_dir', default='/media/hdd/cyclical_annealing/log')
parser.add_argument('--checkpoint_dir', default='models/ptb')
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--min_epochs', default=15, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--decay', default=0, type=int)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--prior_lr', default=0.0001, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--seed', default=859, type=int)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=1000)
parser.add_argument('--kl_every', type=int, default=100)
parser.add_argument('--compute_kl', type=int, default=1)
parser.add_argument('--test', type=int, default=0)


# corpus config
parser.add_argument('--max_utt_len', type=int, default=40)
parser.add_argument('--data_dir', type=str, default='data/stanford')
parser.add_argument('--max_vocab_cnt', type=int, default=10000)
parser.add_argument('--fix_batch', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--backward_size', type=int, default=5)
parser.add_argument('--num_cls', type=int, default=125)
parser.add_argument('--n_per_cls', type=int, default=3)
parser.add_argument('--embedding_path', type=str, default="data/word2vec/smd.txt")


parser.add_argument('--debug', type=bool, default=False)



# KL-annealing
parser.add_argument('--anneal', type=bool, default=False) #TODO (bp): anneal KL weight
parser.add_argument('--anneal_function', type=str, default='logistic')
parser.add_argument('--anneal_k', type=float, default=0.0025)
parser.add_argument('--anneal_x0', type=int, default=2500)
parser.add_argument('--anneal_warm_up_step', type=int, default=0)
parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)
parser.add_argument('--pretrain_ae_step', type=int, default=0)
parser.add_argument('--ae_epochs', type=int, default=8)
parser.add_argument('--dim_target_kl', type=float, default=1.0)
parser.add_argument('--max_kl_weight', type=float, default=0.8)
parser.add_argument('--num_cycle', type=int, default=5)
parser.add_argument('--num_cycle_epoch', type=int, default=100)

parser.add_argument('--num_z_rep', type=int, default=10)


##------------------------------------------------------------------------------------------------------------------##
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"

INT = 0
LONG = 1
FLOAT = 2


class Pack(OrderedDict):
# class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    @staticmethod
    def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
        pack = Pack()
        for k, v in dictionary.items():
            pack[k] = v
        pack['speaker'] = speaker2id[pack.speaker]
        pack['conf'] = dictionary.get('conf', 1.0)
        utt = pack['utt']
        if 'QUERY' in utt or "RET" in utt:
            utt = str(utt)
            utt = utt.translate(None, ''.join([':', '"', "{", "}", "]", "["]))
            utt = unicode(utt)
        if include_domain:
            pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
        else:
            pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
        return pack


class StanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        self._build_vocab(config.max_vocab_cnt)
        # self._output_hyps(os.path.join(self._path, 'kvret_test_public.hyp'))
        print("Done loading corpus")

    def _output_hyps(self, path):
        if not os.path.exists(path):
            f = open(path, "w", encoding="utf-8")
            for utts in self.test_corpus:
                for utt in utts:
                    if utt['speaker'] != 0:
                        f.write(' '.join(utt['utt_ori']) + "\n")
            f.close()

    def _read_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            intent = raw_dialog['scenario']['task']['intent']
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta={'intent': intent, "text": ' '.join(bod_utt[1:-1])})]
            for turn in raw_dialog['dialogue']:

                utt = turn['data']['utterance']
                utt_ori = self.tokenize(utt)
                utt = [BOS, speaker_map[turn['turn']]] + utt_ori + [EOS]
                all_lens.append(len(utt))
                # meta={"text": line.strip()}
                dialog.append(Pack(utt=utt, speaker=turn['turn'], utt_ori=utt_ori, meta={'intent': intent,
                                                                                         'text': ' '.join(utt[1:-1])}))

            if hasattr(self.config, 'include_eod') and self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0, meta={'intent': intent,
                                                                 'text': ' '.join(eod_utt[1:-1])}))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]
        print("<d> index %d" % self.rev_vocab[BOD])

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class SMDDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                # response['utt_orisent'] = response.utt
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)
            # ori_out_utts.append(resp.utt_orisent)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)

##------------------------------------------------------------------------------------------------------------------##
class Word2VecEvaluator():
    def __init__(self, word2vec_file):
        print("Loading word2vecs")
        f = open(word2vec_file, "r")
        self.word2vec = {}
        for line in f:
            line_split = line.strip().split()
            word = line_split[0]
            try:
                vecs = list(map(float, line_split[1:]))
            except:
                pass
                # print(line_split)
            self.word2vec[word] = torch.FloatTensor(np.array(vecs))
        f.close()

    def _sent_vec(self, wvecs):
        m = torch.stack(wvecs, dim=0)
        average = torch.mean(m, dim=0)

        extrema_max, _ = torch.max(m, dim=0)
        extrema_min, _ = torch.min(m, dim=0)
        extrema_min_abs = torch.abs(extrema_min)
        extrema = extrema_max * (extrema_max > extrema_min_abs).float() + extrema_min * (extrema_max <= extrema_min_abs).float()

        average = average / torch.sqrt(torch.sum(average * average))
        extrema = extrema / torch.sqrt(torch.sum(extrema * extrema))
        return average, extrema

    def _cosine(self, v1, v2):
        return torch.sum((v1 * v2) / torch.sqrt(torch.sum(v1 * v1)) / torch.sqrt(torch.sum(v2 * v2)))

    def _greedy(self, wlist1, wlist2):

        max_cosine_list = []
        for v1 in wlist1:
            max_cosine = -2.0
            for v2 in wlist2:
                cos = self._cosine(v1, v2)
                max_cosine = max(cos, max_cosine)
            if max_cosine > -2.0:
                max_cosine_list.append(max_cosine)

        simi = sum(max_cosine_list) / len(max_cosine_list)

        return simi.item()


    def eval_from_file(self, tgt_fn, pred_fn):
        tgt_f = open(tgt_fn, "r")
        pred_f = open(pred_fn, "r")

        tgt_s = []
        pred_s = []

        for tgt_line, pred_line in zip(tgt_f, pred_f):
            tgt = tgt_line.strip().split()
            tgt = [w for w in tgt if w[0] != "<" and w[-1] != ">"] # remove illegal words
            tgt_s.append(tgt)

            pred = pred_line.strip().split()
            pred = [w for w in pred if w[0] != "<" and w[-1] != ">"]  # remove illegal words
            pred_s.append(pred)

        ave_scores = []
        ext_scores = []
        grd_scores = []
        for tgt, pred in zip(tgt_s, pred_s):
            tgt_vecs = [self.word2vec[w] for w in tgt if w in self.word2vec]
            pred_vecs = [self.word2vec[w] for w in pred if w in self.word2vec]
            if len(tgt_vecs) == 0 or len(pred_vecs) == 0:
                continue
            else:
                ave_tgt, ext_tgt = self._sent_vec(tgt_vecs)
                ave_pred, ext_pred = self._sent_vec(pred_vecs)
                ave_scores.append(torch.sum(ave_tgt * ave_pred).item())
                ext_scores.append(torch.sum(ext_tgt * ext_pred).item())
                grd_scores.append((self._greedy(tgt_vecs, pred_vecs) + self._greedy(pred_vecs, tgt_vecs)) / 2)

        logger.info("Average: %lf" % (sum(ave_scores) / len(ave_scores)))
        logger.info("Extrema: %lf" % (sum(ext_scores) / len(ext_scores)))
        logger.info("Greedy: %lf" % (sum(grd_scores) / len(grd_scores)))

##------------------------------------------------------------------------------------------------------------------##

class LM(nn.Module):
    def __init__(self, vocab_size=10000, word_dim=512, h_dim=1024, num_layers=1):
        super(LM, self).__init__()
        self.word_vecs = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.LSTM(word_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(*[nn.Linear(h_dim, vocab_size), nn.LogSoftmax(dim=-1)])

    def forward(self, sent, training=True):
        word_embed = F.dropout(self.word_vecs(sent[:, :-1]), training=training, p=0.5)
        rnn_out, _ = self.rnn(word_embed)
        rnn_out = F.dropout(rnn_out, training=training, p=0.5).contiguous()
        preds = self.linear(rnn_out)
        return preds


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)
##------------------------------------------------------------------------------------------------------------------##
def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


class BaseRNN(nn.Module):
    SYM_MASK = PAD
    SYM_EOS = EOS

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_LATENT = 'latent'
    KEY_CLASS = 'class'
    KEY_RECOG_LATENT = 'recog_latent'
    KEY_POLICY = "policy"
    KEY_G = 'g'
    KEY_PTR_SOFTMAX = 'ptr_softmax'
    KEY_PTR_CTX = "ptr_context"


    def __init__(self, vocab_size, input_size, hidden_size, input_dropout_p,
                 dropout_p, n_layers, rnn_cell, bidirectional):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p,
                                 bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)


    def multinomial_sampling(self, log_probs):
        """
        Sampling element according to log_probs [batch_size x vocab_size]
        Return: [batch_size x 1] selected token IDs
        """
        return torch.multinomial(log_probs, 1)

    def gumbel_max(self, log_probs):
        """
        Obtain a sample from the Gumbel max. Not this is not differentibale.
        :param log_probs: [batch_size x vocab_size]
        :return: [batch_size x 1] selected token IDs
        """
        sample = torch.Tensor(log_probs.size()).uniform_(0, 1)
        sample = cast_type(Variable(sample), FLOAT, self.use_gpu)

        # compute the gumbel sample
        matrix_u = -1.0 * torch.log(-1.0 * torch.log(sample))
        gumbel_log_probs = log_probs + matrix_u
        max_val, max_ids = torch.max(gumbel_log_probs, dim=-1, keepdim=True)
        return max_ids

    def repeat_state(self, state, batch_size, times):
        new_s = state.repeat(1, 1, times)
        return new_s.view(-1, batch_size * times, self.hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, input_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, rnn_cell='gru',
                 variable_lengths=False, bidirection=False):

        super(EncoderRNN, self).__init__(-1, input_size, hidden_size,
                                         input_dropout_p, dropout_p, n_layers,
                                         rnn_cell, bidirection)

        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirection else hidden_size

    def forward(self, input_var, input_lengths=None, init_state=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len, embedding size): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                         batch_first=True)
        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, utt_cell_size, dropout,
                 rnn_cell='gru', bidirection=True, use_attn=False,
                 embedding=None, vocab_size=None, embed_dim=None,
                 feat_size=0):
        super(RnnUttEncoder, self).__init__()
        self.bidirection = bidirection
        self.utt_cell_size = utt_cell_size

        if embedding is None:
            self.embed_size = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = embedding
            self.embed_size = embedding.embedding_dim

        self.rnn = EncoderRNN(self.embed_size+feat_size,
                              utt_cell_size, 0.0, dropout,
                              rnn_cell=rnn_cell, variable_lengths=False,
                              bidirection=bidirection)

        self.multipler = 2 if bidirection else 1
        self.output_size = self.utt_cell_size * self.multipler
        self.use_attn = use_attn
        self.feat_size = feat_size
        if use_attn:
            self.key_w = nn.Linear(self.utt_cell_size*self.multipler,
                                   self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, return_all=False):
        batch_size = int(utterances.size()[0])
        max_ctx_lens = int(utterances.size()[1])
        max_utt_len = int(utterances.size()[2])

        # repeat the init state
        if init_state is not None:
            init_state = init_state.repeat(1, max_ctx_lens, 1)

        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len)
        words_embeded = self.embedding(flat_words)

        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            words_embeded = torch.cat([words_embeded, flat_feats], dim=2)

        enc_outs, enc_last = self.rnn(words_embeded, init_state=init_state)

        if self.use_attn:  # weighted add enc_outs, whose weights are calculated by attention...
            fc1 = torch.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim()-1).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_lens, self.output_size)

        if return_all:
            return utt_embedded, enc_outs, enc_last, attn
        else:
            return utt_embedded


##------------------------------------------------------------------------------------------------------------------##
class RNNVAE(nn.Module):
    def __init__(self, args, rev_vocab, vocab_size=10000,
                 enc_word_dim=200,
                 enc_h_dim=512,
                 enc_num_layers=1,
                 dec_word_dim=200,
                 dec_h_dim=512,
                 dec_num_layers=1,
                 dec_dropout=0.3,
                 latent_dim=32,
                 max_sequence_length=40):
        super(RNNVAE, self).__init__()
        self.args = args
        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers
        self.embedding_size = dec_word_dim
        self.dropout = dec_dropout
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.rev_vocab = rev_vocab

        # context hyperparameters
        self.ctx_embed_size = 200
        self.utt_cell_size = 256
        self.ctx_dropout = 0.3
        self.utt_type = 'attn_rnn'
        self.ctx_cell_size = 512
        self.ctx_num_layer = 1
        self.rnn_cell = 'gru'
        self.fix_batch = False
        self.ctx_encoding_dim = dec_h_dim 



        # encoder
        self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
        self.enc_latent_linear_mean = nn.Linear(enc_h_dim, latent_dim)
        self.enc_latent_linear_logvar = nn.Linear(enc_h_dim, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers,
                                batch_first=True)
        self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                    self.enc_latent_linear_mean, self.enc_latent_linear_logvar])

        # decoder
        self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
        dec_input_size = dec_word_dim
        dec_input_size += latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers,
                               batch_first=True)
        self.dec_linear = nn.Sequential(*[nn.Linear(dec_h_dim, vocab_size),
                                          nn.LogSoftmax(dim=-1)])
        self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])

        # decoder hidden state init
        if latent_dim > 0:
            self.latent_hidden_linear_h = nn.Linear(latent_dim, dec_h_dim)
            self.latent_hidden_linear_c = nn.Linear(latent_dim, dec_h_dim)
            self.dec.append(self.latent_hidden_linear_h)
            self.dec.append(self.latent_hidden_linear_c)


        # ebm prior
        self.prior_dim = self.latent_dim
        self.prior_hidden_dim = args.prior_hidden_dim

        self.prior_network = nn.Sequential(
            nn.Linear(self.prior_dim + self.ctx_encoding_dim, self.prior_hidden_dim),
            GELU(),
            nn.Linear(self.prior_hidden_dim, self.prior_hidden_dim),
            GELU(),
            nn.Linear(self.prior_hidden_dim, args.num_cls)
        )

        # Context Encoder
        self.ctx_embedding = nn.Embedding(vocab_size, self.ctx_embed_size,
                                      padding_idx=self.rev_vocab[PAD])
        self.utt_encoder = RnnUttEncoder(self.utt_cell_size, self.ctx_dropout,
                                         use_attn=self.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.ctx_embedding)
        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      self.ctx_cell_size,
                                      0.0,
                                      self.dropout,
                                      self.ctx_num_layer,
                                      self.rnn_cell,
                                      variable_lengths=self.fix_batch)

        self.q_zx_mlp = nn.Sequential(
                                      nn.Linear(2 * dec_h_dim, self.ctx_encoding_dim),
                                      nn.ReLU()
                                      ) # TODO (bp): better design?
        self.ctx_proj_dec_input = nn.Sequential(
            nn.Linear(self.ctx_encoding_dim, dec_input_size)
        )

    def ebm_prior(self, z, ctx_encoding, cls_output=False, temperature=1.):
        assert len(z.size()) == 2
        assert len(ctx_encoding.size()) == 2
        z = torch.cat((z, ctx_encoding.detach().clone()), dim=-1)
        if cls_output:
            return self.prior_network(z)
        else:
            return temperature * (self.prior_network(z.squeeze()) / temperature).logsumexp(dim=1)

    def q_zx_forward(self, response_encoding, ctx_encoding):
        assert len(response_encoding.size()) == 2
        assert len(ctx_encoding.size()) == 2
        resp_ctx_encoding = torch.cat((response_encoding, ctx_encoding), dim=-1)
        encoding = self.q_zx_mlp(resp_ctx_encoding)
        mean = self.enc_latent_linear_mean(encoding)
        log_var = self.enc_latent_linear_logvar(encoding)
        return mean, log_var


    def context_encoding(self, ctx_utts, ctx_lens):
        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)
        return c_last



    def compute_mi(self, ctx_encoding, z=None, mu=None, log_var=None, n=10, eps=1e-15):
        if z is None:
            assert mu is not None and log_var is not None
            assert len(mu.size()) == 2 and len(log_var.size()) == 2 and len(ctx_encoding.size()) == 2
            mu = mu.repeat(n, 1)
            log_var = log_var.repeat(n, 1) 
            ctx_encoding = ctx_encoding.repeat(n, 1) 
            z = self.sample_amortized_posterior_sample(mu, log_var, sample=True)
        z = z.squeeze()
        assert len(z.size()) == 2
        batch_size = z.size(0)
        log_p_y_z = F.log_softmax(self.ebm_prior(z, ctx_encoding, cls_output=True), dim=-1)
        p_y_z = torch.exp(log_p_y_z)
        
        # H(y)
        log_p_y = torch.log(torch.mean(p_y_z, dim=0) + eps)
        H_y = - torch.sum(torch.exp(log_p_y) * log_p_y)

        # H(y|z)
        H_y_z = - torch.sum(log_p_y_z * p_y_z) / batch_size

        mi = H_y - H_y_z

        return mi

    def encoder(self, x, args):
        word_vecs = self.enc_word_vecs(x)
        h0 = torch.zeros(self.enc_num_layers, word_vecs.size(0), self.enc_h_dim).type_as(word_vecs.data)
        c0 = torch.zeros(self.enc_num_layers, word_vecs.size(0), self.enc_h_dim).type_as(word_vecs.data)
        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]
        # mean = self.enc_latent_linear_mean(enc_h_states_last)
        # logvar = self.enc_latent_linear_logvar(enc_h_states_last)
        return enc_h_states_last

    def sample_amortized_posterior_sample(self, mean, logvar, z=None, sample=True):
        if sample:
            std = logvar.mul(0.5).exp()    
            if z is None:
                z = torch.cuda.FloatTensor(std.size()).normal_(0, 1)
            return z.mul(std) + mean
        else:
            return mean



    def infer_prior_z(self, z, ctx_encoding, args, n_steps=0, verbose=False, y=None):
        z_prior_grads_norm = []

        if n_steps < args.z_n_iters_prior:
            _n_steps = args.z_n_iters_prior
        else:
            _n_steps = n_steps

        for i in range(_n_steps):
            z = z.detach().clone().requires_grad_(True)
            assert z.grad is None
            if y is None:
                f = self.ebm_prior(z, ctx_encoding)
            else:
                f = self.ebm_prior(z, ctx_encoding, cls_output=True)[range(z.size(0)), y]         
            f = f.sum()

            z_grad = torch.autograd.grad(-f, z)[0]
            _z_grad = z_grad.detach().clone()
            if args.ref_dist is 'gaussian':
                z = z - 0.5 * args.prior_step_size * args.prior_step_size * (z_grad + z / (args.ref_sigma * args.ref_sigma))
            else:
                z = z - 0.5 * args.prior_step_size * args.prior_step_size * z_grad
            if args.z_prior_with_noise:
                z += args.prior_step_size * torch.randn_like(z)
            z_prior_grads_norm.append(torch.norm(_z_grad, dim=1).mean().cpu().numpy())

            if (i % 5 == 0 or i == _n_steps - 1) and verbose:
                logger.info('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, _n_steps, -f.item()))


        z = z.detach().clone()
        return z, z_prior_grads_norm


    def infer_z(self, z, sent, beta=1., step_size=0.8, training=True, dropout=0.2):
        args = self.args
        target = sent.detach().clone()
        target = target[:, 1:]
        z_f_grads_norm = []
        z_nll_grads_norm = []

        for i in range(args.z_n_iters):
            # z = torch.autograd.Variable(z.detach().clone(), requires_grad=True)
            z = z.detach().clone()
            z.requires_grad = True
            assert z.grad is None
            logp = self.decoder(sent, z, training=training, dropout=dropout)  # TODO: turn off dropout in inference?
            logp = logp.view(-1, logp.size(2))
            nll = F.nll_loss(logp, target.reshape(-1), reduction='sum', ignore_index=0)
            f = self.ebm_prior(z).sum()
            z_grad_f = torch.autograd.grad(-f, z)[0]
            z_grad_nll = torch.autograd.grad(nll, z)[0]
            _z_grad_f = z_grad_f.detach().clone()
            _z_grad_nll = z_grad_nll.detach().clone()
            if args.ref_dist is 'gaussian':
                z = z - 0.5 * step_size * step_size * (z_grad_nll + beta * z_grad_f + beta * z / (args.ref_sigma * args.ref_sigma))
            else:
                z = z - 0.5 * step_size * step_size * (z_grad_nll + beta * z_grad_f)

            if args.z_with_noise:
                z += step_size * torch.randn_like(z)

            z_f_grads_norm.append(torch.norm(_z_grad_f, dim=1).mean().cpu().numpy())
            z_nll_grads_norm.append(torch.norm(_z_grad_nll, dim=1).mean().cpu().numpy())


        z = z.detach().clone()

        return z, (z_f_grads_norm, z_nll_grads_norm)

    def decoder(self, sent, q_z, ctx_encoding, init_h=True, training=True, dropout=0.2):
        self.word_vecs = F.dropout(self.dec_word_vecs(sent[:, :-1]), training=training, p=dropout)
        if init_h:
            self.h0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
            self.c0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()


        if q_z is not None:
            q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                                 self.word_vecs.size(1), q_z.size(1))
            dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
        else:
            dec_input = self.word_vecs
        if q_z is not None:
            self.h0[-1] = self.latent_hidden_linear_h(q_z)
            self.c0[-1] = self.latent_hidden_linear_c(q_z)
        
        dec_input = self.ctx_proj_dec_input(ctx_encoding).unsqueeze(1) + dec_input

        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()
        dec_linear_input = F.dropout(dec_linear_input, training=training, p=dropout)
        preds = self.dec_linear(dec_linear_input.view(
            self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
            self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds

    def inference(self, device, sos_idx, ctx_encoding, max_len=None, z=None, init_h=True, training=False):

        batch_size = z.size(0)

        if init_h:
            self.h0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
            self.c0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        self.h0[-1] = self.latent_hidden_linear_h(z)
        self.c0[-1] = self.latent_hidden_linear_c(z)

        if max_len is None:
            max_len = self.max_sequence_length
        generations = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        preds_sequence = torch.zeros(batch_size, max_len, self.vocab_size, dtype=torch.float, device=device)
        input_sequence = torch.tensor([sos_idx]*batch_size, dtype=torch.long, device=device)

        hidden = (self.h0, self.c0)
        for i in range(max_len):
            input_embedding = F.dropout(self.dec_word_vecs(input_sequence).view(batch_size, 1, self.embedding_size), training=training)
            dec_input = torch.cat([input_embedding, z.view(batch_size, 1, self.latent_dim)], dim=2)  #TODO: project z to embedding space before concat?
            dec_input = self.ctx_proj_dec_input(ctx_encoding).unsqueeze(1) + dec_input
            output, hidden = self.dec_rnn(dec_input, hidden)
            dec_linear_input = output.contiguous()
            dec_linear_input = F.dropout(dec_linear_input, training=training) #TODO: this dropout is necessary?
            preds = self.dec_linear(dec_linear_input.view(batch_size, self.dec_h_dim))
            probs = F.softmax(preds, dim=1)
            samples = probs.argmax(dim=1)
            generations[:, i] = samples.view(-1).data
            preds_sequence[:, i, :] = preds
            input_sequence = samples.view(-1)

        return generations, preds_sequence


##--------------------------------------------------------------------------------------------------------------------##

def main(args, output_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_data = Dataset(args.train_file)
    # val_data = Dataset(args.val_file)
    # test_data = Dataset(args.test_file)
    # train_sents = train_data.batch_size.sum()
    # vocab_size = int(train_data.vocab_size)
    # logger.info('Train data: %d batches' % len(train_data))
    # logger.info('Val data: %d batches' % len(val_data))
    # logger.info('Test data: %d batches' % len(test_data))
    # logger.info('Word vocab size: %d' % vocab_size)


    corpus_client = StanfordCorpus(args)
    corpus = corpus_client.get_corpus()
    train_dial = corpus['train']
    test_dial = corpus['test']
    train_feed = SMDDataLoader("Train", train_dial, args)
    test_feed = SMDDataLoader("Test", test_dial, args)


    vocab_size = len(corpus_client.vocab)

    checkpoint_dir = output_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    suffix = "%s_%s.pt" % (args.model, 'bl')
    checkpoint_path = os.path.join(checkpoint_dir, suffix)

    writer = None

    # indexer = Indexer()
    # indexer.load_vocab(args.vocab_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # reverse_lm = LM(vocab_size=vocab_size, word_dim=args.dec_word_dim, h_dim=args.dec_h_dim, num_layers=args.dec_num_layers)
    # reverse_lm.cuda()

    # forward_lm_ckpt = torch.load(args.pretrained_lm)
    # forward_lm = forward_lm_ckpt['model']

    if args.train_from == '':
        model = RNNVAE(args, corpus_client.rev_vocab, vocab_size=vocab_size,
                       dec_dropout=args.dec_dropout,
                       latent_dim=args.latent_dim)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
    else:
        logger.info('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']

    logger.info("model architecture")
    print(model)

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]

    # optimizer_prior = torch.optim.Adam(prior_params, lr=args.prior_lr, weight_decay=args.ebm_reg)
    # optimizer = torch.optim.Adam(likelihood_params, lr=args.lr)
    # reverse_lm_optim = torch.optim.Adam(reverse_lm.parameters(), lr=args.revserse_lm_lr)

    optimizer = torch.optim.Adam([
                                     {'params': prior_params, 'lr': args.prior_lr, 'weight_decay': args.ebm_reg},
                                     {'params': likelihood_params},
                                     ],
                                     lr=args.lr)


    if args.warmup == 0:
        args.beta = 1.
    else:
        args.beta = 0.001

    criterion = nn.NLLLoss(ignore_index=corpus_client.rev_vocab[PAD], reduction='sum')
    model.cuda()
    # criterion.cuda()
    model.train()

    if args.test == 1:
        args.beta = 1
        test_data = Dataset(args.test_file)
        eval(test_data, model)
        exit()

    t = 0
    best_val_nll = 1e5
    best_epoch = 0
    val_stats = []
    epoch = 0
    z_means = torch.zeros(5, args.latent_dim, device=device, dtype=torch.float)


    # train_recons_batch = train_data[100][0][:10, :].to(device)
    # test_recons_batch = test_data[100][0][:10, :].to(device)


    # compute_homogeneity(model, test_feed, args)

    i = 0
    vae_kl_weights = frange_cycle_zero_linear(args.num_cycle_epoch*424, start=0.0, stop=args.max_kl_weight, 
                                                n_cycle=args.num_cycle,
                                                ratio_increase=0.25,
                                                ratio_zero=0.1)

    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        logger.info('Starting epoch %d' % epoch)
        train_nll_abp = 0.
        num_sents = 0
        num_words = 0
        b = 0

        train_feed.epoch_init(args, shuffle=True)

        # if epoch == args.ae_epochs + 1:
        #     i = 0

        while True:
            # i += 1
            data_feed = train_feed.next_batch()
            if data_feed is None:
                break

            if args.debug and b > 100:
                break

            # vae_kl_weight = kl_anneal_function(args.anneal_function, i,
            #                                    args.anneal_k, args.anneal_x0,
            #                                    args.anneal_warm_up_step,
            #                                    args.anneal_warm_up_value)

            # batch_size = len(data_feed['output_lens'])
            sents = torch.tensor(data_feed['outputs'])

            if args.gpu >= 0:
                sents = sents.cuda().long()
            vae_kl_weight = vae_kl_weights[i]
            ae_train = True if vae_kl_weight == 0.0 else False
            b += 1
            i += 1

            # generator update
            optimizer.zero_grad()

            # ctx encoding
            ctx_lens = data_feed['context_lens']
            batch_size = len(ctx_lens)
            ctx_utts = torch.tensor(data_feed['contexts']).cuda().long()
            ctx_encoding = model.context_encoding(ctx_utts, ctx_lens)

            # vae
            response_encoding = model.encoder(sents, args)
            mu, log_var = model.q_zx_forward(response_encoding, ctx_encoding)

            if ae_train:
                z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=False)
            else:
                z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
            preds = model.decoder(sents, z_samples, ctx_encoding, dropout=args.dec_dropout)
            labels = sents[:, 1:].contiguous()
            nll_abp =  criterion(preds.transpose(1, 2), labels) / batch_size          
            loss_g_kl = - 0.5 * (1 + log_var - mu ** 2 - log_var.exp())
            kl_mask = (loss_g_kl > args.dim_target_kl).float()
            loss_g_kl = (kl_mask * loss_g_kl).sum(dim=1).mean()
            loss_g_ebm_cross_ent = - model.ebm_prior(z_samples, ctx_encoding).squeeze().mean()
            positive_potential = - loss_g_ebm_cross_ent

            mi = model.compute_mi(ctx_encoding, mu=mu, log_var=log_var, n=args.num_z_rep)

            z_0_prior = sample_p_0(sents, args)
            z_prior, z_prior_grads_norm = model.infer_prior_z(z_0_prior, ctx_encoding, args, verbose=(b%500==0))
            negative_potential = model.ebm_prior(z_prior.detach(), ctx_encoding).mean()




            train_nll_abp += nll_abp.item()*batch_size
            cd = positive_potential - negative_potential

            if ae_train:
                abp_loss = nll_abp
            else:
                abp_loss = nll_abp + vae_kl_weight * (loss_g_kl + loss_g_ebm_cross_ent + negative_potential) - 50 * mi



            optimizer.zero_grad()

            abp_loss.backward()

            if args.max_grad_norm > 0:
                llhd_grad_norm = torch.nn.utils.clip_grad_norm_(likelihood_params, args.max_grad_norm)
            else:
                llhd_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.clone().detach()) for p in likelihood_params]))

            if args.max_grad_norm_prior > 0:
                prior_grad_norm = torch.nn.utils.clip_grad_norm_(prior_params, args.max_grad_norm_prior)
            else:
                prior_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.clone().detach()) for p in prior_params]))

            optimizer.step()


            num_sents += batch_size
            # num_words += batch_size * length


            if b % args.print_every == 0:


                with torch.no_grad():
                    positive_potential_2 = model.ebm_prior(z_samples, ctx_encoding).mean()
                    negative_potential_2 = model.ebm_prior(z_prior, ctx_encoding).mean()

                    # z_var = ' '.join(['{:10.6f}'.format(_z_var) for _z_var in z_means.std(dim=0).pow(2)])

                    prior_param_norm = torch.norm(torch.stack([torch.norm(p.clone().detach()) for p in prior_params]))
                    llhd_param_norm = torch.norm(torch.stack([torch.norm(p.clone().detach()) for p in likelihood_params]))
                    param_norm_str = '[ebm:{:8.2f} lh:{:8.2f}]'.format(prior_param_norm, llhd_param_norm)

                    grad_norm_str = '[ebm:{:8.2f} lh:{:8.2f}]'.format(prior_grad_norm, llhd_grad_norm)

                    posterior_z_disp_str = torch.norm(z_0_prior - z_samples, dim=1).mean()
                    prior_z_disp_str = torch.norm(z_0_prior - z_prior, dim=1).mean()
                    z_disp_str = '[pr:{:8.2f} po:{:8.2f}]'.format(prior_z_disp_str, posterior_z_disp_str)

                    prior_posterior_z_norm_str = '[noise:{:8.2f} pr:{:8.2f} po:{:8.2f}]'.format(torch.norm(z_0_prior, dim=1).mean(),
                                                                  torch.norm(z_prior, dim=1).mean(),
                                                                  torch.norm(z_samples, dim=1).mean())

                    prior_z_grad_norm_str = ' '.join(['{:8.2f}'.format(g) for g in z_prior_grads_norm])
                    posterior_z_f_grad_norm_str = prior_z_grad_norm_str
                    posterior_z_nll_grad_norm_str = prior_z_grad_norm_str

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_prior.mean(), z_prior.std(), z_prior.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_samples.mean(), z_samples.std(), z_samples.abs().max())


                logger.info('Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, TrainABP_REC={:10.4f}, MI={:10.4f}, KL={:10.4f}, kl_weight={:10.4f}, ae={}'
                            ' CD={:10.4f}, PP={:10.4f} / {:10.4f} / {:10.4f}, NP={:10.4f} / {:10.4f} / {:10.4f}, |params|={}, |grad|={}, |z|={},'
                            ' z_disp={}, |prior_z_grad|={}, |posterior_z_f_grad|={}, |posterior_z_nll_grad|={}, prior_moments={}, posterior_moments={},'
                            ' BestValPerf={:10.4f}, BestEpoch={:4d}, Beta={:10.4f}'.format(
                    epoch, b+1, train_feed.num_batch, args.lr, train_nll_abp / num_sents, mi, loss_g_kl, vae_kl_weight, str(ae_train), cd, positive_potential, positive_potential_2, positive_potential_2 - positive_potential,
                    negative_potential, negative_potential_2, negative_potential_2 - negative_potential,
                    param_norm_str, grad_norm_str, prior_posterior_z_norm_str, z_disp_str,
                    prior_z_grad_norm_str, posterior_z_f_grad_norm_str, posterior_z_nll_grad_norm_str, prior_moments, posterior_moments, best_val_nll, best_epoch, args.beta))

        epoch_train_time = time.time() - start_time
        logger.info('Time Elapsed: %.1fs' % epoch_train_time)

        # logger.info('---')
        # logger.info('---')
        # logger.info('---')
        # compute_homogeneity(model, test_feed, args)
        # logger.info('---')
        # logger.info('---')
        # logger.info('---')

        # logger.info('--------------------------------')
        # logger.info('Checking validation perf...')
        # logger.record_tabular('Epoch', epoch)
        # logger.record_tabular('Mode', 'Val')
        # logger.record_tabular('LR', args.lr)
        # logger.record_tabular('Epoch Train Time', epoch_train_time)
        # val_nll, writer = eval(args, val_data, model, forward_lm, indexer, writer, epoch, verbose=True)
        # val_stats.append(val_nll)

        cluster_keys = get_cluster_examples(args, model, test_feed, corpus_client, epoch=epoch)

        logger.info('--------------------------------')
        logger.info('Checking train perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Train')
        logger.record_tabular('LR', args.lr)
        logger.record_tabular('Epoch Train Time', epoch_train_time)
        train_nll = eval(args, train_feed, model, corpus_client, mode='train')


        logger.info('--------------------------------')
        logger.info('Checking test perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Test')
        logger.record_tabular('LR', args.lr)
        logger.record_tabular('Epoch Train Time', epoch_train_time)
        test_nll = eval(args, test_feed, model, corpus_client, vae_kl_weight, epoch=epoch, debug=args.debug, cluster_keys=cluster_keys)


        # if val_nll < best_val_nll:
        #     best_val_nll = val_nll
        #     best_epoch = epoch
        #     model.cpu()
        #     checkpoint = {
        #         'args': args.__dict__,
        #         'model': model,
        #         'val_stats': val_stats
        #     }
        #     logger.info('Save checkpoint to %s' % checkpoint_path)
        #     torch.save(checkpoint, checkpoint_path)
        #     model.cuda()
        # else:
        #     if epoch >= args.min_epochs:
        #         args.decay = 1
        # if args.decay == 1:
        #   args.lr = args.lr*0.5
        #   for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.lr
        #   if args.lr < 0.03:
        #     break

        # if args.reverse_lm_eval:
        #     logger.info('--------------------------------')
        #     logger.info('Checking reverse nll...')
        #     all_samples = collect_reverse_samples(args, train_data, model, indexer, output_dir, epoch, reconstruction=False, gaussian=False)
        #     eval_reverse_lm(args, reverse_lm, all_samples, reverse_lm_optim, args.reverse_lm_num_epoch, test_data, indexer)




##--------------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------------------------------------------------------------------------##
# def collect_reverse_samples(args, data, model, indexer, output_path, train_epoch, reconstruction=False, gaussian=False):
#     model.prior_network.eval() # TODO: eval or train mode?
#     device = torch.device('cuda')

#     all_samples = []

#     for b in range(len(data)):
#         # if b > 300:
#         #     break
#         sents = data[b][0].cuda()
#         z_0_prior = sample_p_0(sents, args)
#         if reconstruction:
#             z_prior, _ = model.infer_z(z_0_prior, sents, args.beta, args.z_step_size, training=False, dropout=args.dec_dropout)
#         else:
#             if gaussian:
#                 z_prior = z_0_prior
#             else:
#                 z_prior = model.infer_prior_z(z_0_prior, args, n_steps=30)[0]
#         samples, _ = model.inference(device, indexer.d[BOS], z=z_prior)
#         samples, length = process_generated_samples(samples, indexer.d[indexer.BOS], indexer.d[indexer.EOS], indexer.d[indexer.PAD])
#         bos_pad = torch.tensor([indexer.d[indexer.BOS]]*samples.size(0), device=samples.device).unsqueeze(1)
#         samples = torch.cat((bos_pad, samples), dim=1)

#         all_samples.append((samples.cpu(), length))

#         if (b+1) % 20 == 0:
#             logger.info('batch{:4d} generated'.format(b+1))

#         torch.save(all_samples, output_path + '/{:03d}_reverse_samples.pt'.format(train_epoch))

#     return all_samples


# def eval_reverse_lm(args, lm, model_data, optim, num_epoch, eval_data, indexer):

#     criterion = nn.NLLLoss(ignore_index=indexer.d[indexer.PAD], reduction='sum')
#     criterion_sum = nn.NLLLoss(ignore_index=indexer.d[indexer.PAD], reduction='sum')

#     lm.train()
#     device = torch.device('cuda')

#     total_nll = 0.
#     num_sents = 0.
#     num_words = 0.

#     total_nll_eval = 0.
#     num_sents_eval = 0.
#     num_words_eval = 0.


#     for epoch in range(num_epoch):
#         for b in range(len(model_data)):
#             samples, length = model_data[b]
#             samples = samples.to(device)
#             batch_size = samples.size(0)

#             optim.zero_grad()
#             forward_preds = lm(samples, training=True)
#             forward_nll = sum([criterion(forward_preds[:, l], samples[:, l + 1]) for l in range(samples.size(1)-1)]) # minus 1 accounting for the bos
#             forward_nll.backward()
#             optim.step()

#             total_nll += forward_nll.cpu().item()
#             num_sents += batch_size
#             num_words += sum(length)

#             if (b+1) % 100 == 0:
#                 logger.info('reverse lm training epoch={:4d}, batch={:4d}/{:4d}, nll={:10.4f} ppl={:10.4f}'.format(epoch, b+1, len(model_data), total_nll/num_sents, np.exp(total_nll/num_words)))

#         # reverse eval
#         with torch.no_grad():
#             lm.eval()
#             for eval_b in range(len(eval_data)):
#                 sents, length, batch_size = eval_data[eval_b]
#                 sents = sents.to(device)
#                 forward_preds = lm(sents, training=False) # TODO: training True or False
#                 forward_nll = sum([criterion_sum(forward_preds[:, l], sents[:, l + 1]) for l in range(sents.size(1)-1)]) # minus 1 accounting for the bos

#                 total_nll_eval += forward_nll.cpu().item()
#                 num_sents_eval += batch_size
#                 num_words_eval += batch_size * length
#             logger.info('-----reverse nll and ppl-------')
#             logger.info('reverse lm eval epoch={:4d}, nll={:10.4f}, ppl={:10.4f}'.format(epoch, total_nll_eval/num_sents_eval, np.exp(total_nll_eval / num_words_eval)))
#         lm.train()



# def eval_forward_lm(args, data, model, lm, indexer, nbatch=5, reconstruction=False, gaussian=False, verbose=False):
#     model.prior_network.eval()
#     criterion = nn.NLLLoss(ignore_index=indexer.d[indexer.PAD], reduction='sum').cuda()  # TODO: reduction='sum'
#     lm.eval()
#     device = torch.device('cuda')
#     total_nll_abp = 0.
#     num_sents = 0.
#     num_words = 0.

#     for b in range(1, nbatch+1):
#         sents = data[b][0].cuda()
#         z_0_prior = sample_p_0(sents, args)
#         if reconstruction:
#             z_prior, _ = model.infer_z(z_0_prior, sents, args.beta, args.z_step_size, training=False, dropout=args.dec_dropout)
#         else:
#             if gaussian:
#                 z_prior = z_0_prior
#             else:
#                 z_prior = model.infer_prior_z(z_0_prior, args, n_steps=30)[0]
#         samples, _ = model.inference(device, indexer.d[indexer.BOS], z=z_prior, training=False)
#         samples, length = process_generated_samples(samples, indexer.d[indexer.BOS], indexer.d[indexer.EOS], indexer.d[indexer.PAD])
#         bos_pad = torch.tensor([indexer.d[indexer.BOS]]*samples.size(0), device=samples.device).unsqueeze(1)
#         samples = torch.cat((bos_pad, samples), dim=1)

#         if verbose:
#             logger.info(*idx2word(samples, i2w=indexer.idx2word, ending_idx=indexer.d[indexer.EOS]))

#         batch_size = samples.size(0)
#         forward_preds = lm(samples, training=False)
#         forward_nll = sum([criterion(forward_preds[:, l], samples[:, l + 1]) for l in range(samples.size(1)-1)]) # minus 1 accounting for the bos
#         total_nll_abp += forward_nll.cpu().item()
#         num_sents += batch_size
#         num_words += sum(length)

#     model.train()
#     return total_nll_abp / num_sents, np.exp(total_nll_abp / num_words)

def process_generated_samples(samples, starting_idx, ending_idx, padding_idx):
    sents = []
    max_len = 0
    for s in samples:
        _s = []
        for w in s:
            _s.append(w)
            if w.item() == ending_idx:
                break
        if len(_s) > max_len:
            max_len = len(_s)
        s = torch.stack(_s)
        sents.append(s)

    padded_sents = []
    sents_len = []
    for s in sents:
        len_s = len(s)
        sents_len.append(len_s)
        diff = max_len - len_s
        padded_s = s
        if diff > 0:
            padded_s = torch.cat((s, torch.tensor([padding_idx]*diff, device=samples.device)))
        padded_sents.append(padded_s)
    return torch.stack(padded_sents, dim=0), sents_len

def get_sent(ids, vocab, stop_eos=True, stop_pad=True):
    ws = []
    for w_id in ids:
        w = vocab[w_id]
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        if w != PAD:
            ws.append(w)
    return " ".join(ws)

    

def eval(args, test_feed, model, corpus_client, kl_weight=0.0, epoch=0, mode='test', debug=False, cluster_keys=None, max_num_cluster=15):
    model.eval()
    rev_vocab = corpus_client.rev_vocab
    vocab = corpus_client.vocab
    criterion = nn.NLLLoss(ignore_index=rev_vocab[PAD], reduction='sum')
    total_nll_abp = 0.
    num_sents = 0

    # flags
    do_generation_cond_y = mode is 'test' and (True if debug else kl_weight >= args.max_kl_weight) and cluster_keys 
    do_w2v_eval = mode is 'test' and kl_weight >= args.max_kl_weight

    # set up file handles
    hyps_f = open(os.path.join(output_dir, "hyp-" + mode), "w")
    refs_f = open(os.path.join(output_dir, "ref-" + mode), "w")
    if mode is 'test':
        hyps_prior_f_name = os.path.join(output_dir, "hyp-prior-{:0>2d}".format(epoch) + mode)
        hyps_prior_f = open(hyps_prior_f_name, "w")
    if do_generation_cond_y:
        cond_prior_f = open(os.path.join(output_dir, "cond-prior-ep{:0>2d}".format(epoch) + mode), "w")

    root_dir = os.getcwd()
    perl_path = os.path.join(root_dir, "multi-bleu.perl")



    test_feed.epoch_init(args, shuffle=False)
    i = 0
    while True:
        i += 1
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break
        if mode is not 'test' and i > 200:
            break
        
        batch_size = len(data_feed['output_lens'])
        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()
        num_sents += batch_size
        device = sents.device

        ctx_lens = data_feed['context_lens']
        ctx_utts = torch.tensor(data_feed['contexts']).cuda().long()
        ctx_encoding = model.context_encoding(ctx_utts, ctx_lens)

        response_encoding = model.encoder(sents, args)
        mu, log_var = model.q_zx_forward(response_encoding, ctx_encoding)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=False)
        preds = model.decoder(sents, z_samples, ctx_encoding, dropout=args.dec_dropout)
        labels = sents[:, 1:].contiguous()
        nll_abp =  criterion(preds.transpose(1, 2), labels) / batch_size
        total_nll_abp += nll_abp.item() * batch_size
        

        recons_sample_ids, _ = model.inference(device, rev_vocab[BOS], ctx_encoding, z=z_samples)
        
        # prior sample
        if mode is 'test':
            z_0_prior = sample_p_0(sents, args)
            z_prior, _ = model.infer_prior_z(z_0_prior, ctx_encoding, args)
            prior_sample_ids, _ = model.inference(device, rev_vocab[BOS], ctx_encoding, z=z_prior)

        for s, pred_ids in enumerate(recons_sample_ids):
            pred_str = get_sent(pred_ids.cpu().numpy(), vocab)
            true_str = get_sent(labels[s].cpu().numpy(), vocab)

            hyps_f.write(pred_str.strip() + "\n")
            refs_f.write(true_str.strip() + "\n")

            # prior sample
            if mode is 'test':
                prior_str = get_sent(prior_sample_ids[s].cpu().numpy(), vocab)
                hyps_prior_f.write(prior_str.strip() + "\n")

        if do_generation_cond_y:
            num_ctx_per_batch = 1
            num_sample_per_cls = 5

            cluster_keys = cluster_keys if len(cluster_keys) <= max_num_cluster else cluster_keys[:max_num_cluster]
            y = torch.tensor(cluster_keys).repeat_interleave(num_sample_per_cls).to(sents.device).long()

            batch_size_cond_y = len(cluster_keys) * num_sample_per_cls
            ctx_id = torch.randint(0, ctx_encoding.size(0), size=(num_ctx_per_batch,))
            _ctx_encoding = ctx_encoding[ctx_id].squeeze()
            _ctx_encoding = _ctx_encoding.repeat(batch_size_cond_y, 1)
            z_0_prior = sample_p_0(_ctx_encoding, args)
            z_prior_cond_y, _ = model.infer_prior_z(z_0_prior, _ctx_encoding, args, y=y)
            prior_sample_ids_cond_y, _ = model.inference(device, rev_vocab[BOS], _ctx_encoding, z=z_prior_cond_y)
            
            ctx_vector = data_feed['contexts'][ctx_id][:data_feed['context_lens'][ctx_id]]
            ctx_str = [get_sent(el, vocab) for el in ctx_vector]
            ctx_str = '--'.join(ctx_str)
            cond_prior_f.write("\n" + '<<<<< context: {}>>>>>>'.format(ctx_str) + "\n")
            for s, sample_ids in enumerate(prior_sample_ids_cond_y):
                prior_str = get_sent(sample_ids.cpu().numpy(), vocab)
                if s % num_sample_per_cls == 0:
                    cond_prior_f.write('<----------------------------------->' + "\n")
                cond_prior_f.write(prior_str.strip() + "\n")

    ### close file handles ###
    hyps_f.close()
    refs_f.close()
    if mode is 'test':
        hyps_prior_f.close()
    if do_generation_cond_y:
        cond_prior_f.close()


    ### compute reconstruction nll ###
    rec_abp = total_nll_abp / num_sents
    logger.record_tabular('ABP REC', rec_abp)
    logger.dump_tabular()


    ### compute bleu ###
    p = subprocess.Popen(["perl", perl_path, os.path.join(output_dir, "ref-"+mode)], stdin=open(os.path.join(output_dir, "hyp-"+mode), "r"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    while p.poll() == None:
        pass
    print("multi-bleu.perl return code: ", p.returncode)
    for line in p.stdout:
        line = line.decode("utf-8")
        if line[:4] == "BLEU":
            logger.info('---')
            logger.info('---')
            logger.info(line)
            logger.info('---')
            logger.info('---')

    ### compute prior bleu ###
    if mode is 'test':
        p = subprocess.Popen(["perl", perl_path, os.path.join(output_dir, "ref-"+mode)], stdin=open(hyps_prior_f_name, "r"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        while p.poll() == None:
            pass
        print("multi-bleu.perl return code: ", p.returncode)
        for line in p.stdout:
            line = line.decode("utf-8")
            if line[:4] == "BLEU":
                logger.info('---')
                logger.info('---')
                logger.info('---prior sample---')
                logger.info(line)
                logger.info('---')
                logger.info('---')


    ### do word-embedding evaluation ###
    if do_w2v_eval:
        w2v_evaluator = Word2VecEvaluator(args.embedding_path)
        w2v_evaluator.eval_from_file(tgt_fn=os.path.join(output_dir, "ref-"+mode), pred_fn=hyps_prior_f_name)

    model.train()
    return rec_abp


##--------------------------------------------------------------------------------------------------------------------##
def kl_anneal_function(anneal_function, step, k, x0, warmup_step=0, warmup_value=0.0):
    if step <= warmup_step:
        return warmup_value
    step -= warmup_step
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, x0 + step / k)
    elif anneal_function == 'const':
        return k

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.25, ratio_zero=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L 


def get_cluster_examples(args, model, test_feed, corpus_client, epoch=0, max_samples=50):
    from collections import defaultdict

    model.eval()
    vocab = corpus_client.vocab
    test_feed.epoch_init(args, shuffle=False)    
    cluster_dict = defaultdict(list)
    cluster_dict_num = defaultdict(int)
    
    while True:
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break

        # true_labels = data_feed['z_labels']
        # act_labels = true_labels[:, 0]
        # emt_labels = true_labels[:, 1]

        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()

        ctx_lens = data_feed['context_lens']
        ctx_utts = torch.tensor(data_feed['contexts']).cuda().long()
        ctx_encoding = model.context_encoding(ctx_utts, ctx_lens)

        response_encoding = model.encoder(sents, args)
        mu, log_var = model.q_zx_forward(response_encoding, ctx_encoding)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
        pred_logits = model.ebm_prior(z_samples, ctx_encoding, cls_output=True)
        pred_labels = pred_logits.argmax(dim=-1).cpu()

        for b_id, sent in enumerate(sents):
            # act_label = act_labels[b_id]
            # emt_label = emt_labels[b_id]
            pred_label = '{:0>3d}'.format(pred_labels[b_id])
            pred_label_num = pred_labels[b_id]
            sent_text = get_sent(sent.cpu().numpy(), vocab)
            save_str = sent_text
            cluster_dict[pred_label].append(save_str)
            cluster_dict_num[pred_label_num] += 1

    keys = cluster_dict.keys()
    keys = sorted(keys)
    logger.info("Find {} clusters".format(len(keys)))

    keys_num = cluster_dict_num.keys()
    keys_num = sorted(keys_num, key=lambda el: cluster_dict_num[el])

    cluster_f = open(os.path.join(output_dir, "cluster-examples-{:0>2d}".format(epoch)), "w")
    for symbol in keys:
        sents = cluster_dict[symbol]
        if len(sents) < max_samples:
            subset_ids = list(range(len(sents)))
            np.random.shuffle(subset_ids)
        else:
            subset_ids = np.random.choice(range(len(sents)), max_samples, replace=False)
        
        cluster_f.write('<--------------{}-------------->'.format(symbol) + "\n")
        for s_id in subset_ids:
            cluster_f.write(sents[s_id] + "\n")
    cluster_f.close()

    return keys_num


def get_chat_tokenize():
    import nltk
    return nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize


def compute_homogeneity(model, test_feed, args):
    test_feed.epoch_init(args, shuffle=False)
    act_scores = []
    emt_scores = []
    batch_sizes = []
    while True:
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break

        labels = data_feed['z_labels']
        act_labels = labels[:, 0]
        emt_labels = labels[:, 1]
        batch_size = labels.shape[0]

        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()

        ctx_lens = data_feed['context_lens']
        ctx_utts = torch.tensor(data_feed['contexts']).cuda().long()
        ctx_encoding = model.context_encoding(ctx_utts, ctx_lens)

        response_encoding = model.encoder(sents, args)
        mu, log_var = model.q_zx_forward(response_encoding, ctx_encoding)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
        pred_logits = model.ebm_prior(z_samples, ctx_encoding, cls_output=True)
        pred_labels = pred_logits.argmax(dim=-1).cpu()

        act_score = homogeneity_score(act_labels, pred_labels)
        emt_score = homogeneity_score(emt_labels, pred_labels)

        act_scores.append(act_score * batch_size)
        emt_scores.append(emt_score * batch_size)
        batch_sizes.append(batch_size)

    act_scores = sum(act_scores)
    emt_scores = sum(emt_scores)
    total_size = sum(batch_sizes)

    act_scores = act_scores / total_size
    emt_scores = emt_scores / total_size

    logger.info('------ act homogeneity {} '.format(act_scores)+ 
                'emt homogeneity {}'.format(emt_scores))
    


    


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / (0.5 * sigma.expand_as(w)))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def sample_p_0(x, args):
    if args.ref_dist is 'gaussian':
        return args.init_factor * torch.randn(*[x.size(0), args.latent_dim], device=x.device)
    else:
        return torch.Tensor(*[x.size(0), args.latent_dim]).uniform_(-1, 1).to(x.device)


# def idx2word(idx, i2w, ending_idx):
#     sent_str = [str()] * len(idx)

#     for i, sent in enumerate(idx):

#         for word_id in sent:
#             word_id = word_id.item()

#             if word_id == ending_idx:
#                 break
#             sent_str[i] += i2w[word_id] + " "

#         sent_str[i] = sent_str[i].strip() + "\n"
#     return sent_str

# def interpolate(start, end, steps):

#     interpolation = np.zeros((start.shape[0], steps + 2))

#     for dim, (s, e) in enumerate(zip(start, end)):
#         interpolation[dim] = np.linspace(s, e, steps + 2)

#     return interpolation.T


# def sentence2tensor(sent, dictionary, unk_idx, device):
#     idx = []
#     for t in sent:
#         idx.append(dictionary.get(t, unk_idx))
#     tensor = torch.tensor(idx, dtype=torch.long, device=device)
#     return tensor

# def end_points(dictionary, unk_idx, device):
#     # sents = [["<s>", "i", "want", "to", "talk", "to", "you", "</s>"],
#     #          ["<s>", "she", "did", "n't", "want", "to", "be", "with", "him", "</s>"],
#     #          ["<s>", "he", "was", "silent", "for", "a", "long", "moment", "</s>"],
#     #          ["<s>", "it", "was", "my", "turn", "</s>"]]
#     sents = [["<s>", "the", "men", "are", "feeling", "competitive", "</s>"],
#              ["<s>", "a", "young", "woman", "is", "sitting", "in", "a", "field", "</s>"],

#              ["<s>", "a", "man", "walking", "across", "a", "bridge", "near", "a", "steak", "restaurant", ".", "</s>"],
#              ["<s>", "a", "woman", "in", "a", "white", "shirt", "and", "shorts", "is", "playing", "a", "red", "guitar", ".", "</s>"],

#              ["<s>", "a", "girl", "with", "glasses", "next", "red", "white", "and", "blue", "flags", ".", "</s>"],
#              ["<s>", "three", "greyhounds", "are", "taking", "a", "walk", "with", "their", "owner", ".", "</s>"],

#              ["<s>",  "one", "young", "child", "in", "a", "swimsuit", "jumping", "off", "a", "blue", "inflatable", "slide", "with", "water", ".", "</s>"],
#              ["<s>", "a", "girl", "swings", "from", "a", "rope", "swing", "in", "front", "</s>"],

#              ["<s>", "both", "men", "are", "wearing", "similar", "colors", ".", "</s>"],
#              ["<s>", "a", "huge", "animal", "surrounded", "</s>"],

#              ["<s>", "a", "child", "is", "eating", "with", "utensils", ".", "</s>"],
#              ["<s>", "a", "youth", "wearing", "a", "blue", "and", "red", "jersey", "and", "yellow", "helmet", "is", "crouching", "in", "a", "football", "position", ".", "</s>"],

#              ["<s>", "a", "middle-aged", "man", "with", "long", ",", "curly", "red-hair", "wearing", "a", "dark", "vest", ",",
#               "shirt", "and", "pants", "is", "holding", "a", "microphone", "in", "front", "of", "a", "black", "backdrop", ".", "</s>"],
#              ["<s>", "people", "are", "doing", "<unk>", "</s>"],

#              ["<s>", "the", "animals", "are", "near", "the", "water", ".", "</s>"],
#              ["<s>", "a", "truck", "is", "going", "tow", "an", "illegally", "parked", "white", "volkswagon", "." "</s>"]]

#     lens = [len(sent) for sent in sents]
#     max_len = max(lens)
#     tensors = torch.zeros((len(sents), max_len), dtype=torch.long, device=device)
#     for i, sent in enumerate(sents):
#         tensor = sentence2tensor(sent, dictionary, unk_idx, device)
#         tensors[i, :tensor.size(0)] = tensor
#     return tensors


# def shuffle(sents, length, batch_size, device):
#     sents = sents.detach().clone()
#     originals = torch.randint(low=1, high=length.item()-2, size=(batch_size.item(),))
#     assert originals.max() < length.item() - 2
#     for i in range(batch_size.item()):
#         original = originals[i]
#         temp = sents[i, original].detach().clone()
#         coinflip = torch.rand(1).to(device)
#         if original == 1:
#             sents[i, original] = sents[i, original+1].detach().clone()
#             sents[i, original+1] = temp
#         elif original == length.item() - 3:
#             sents[i, original] = sents[i, original - 1].detach().clone()
#             sents[i, original-1] = temp
#         elif coinflip > 0.5:
#             sents[i, original] = sents[i, original + 1].detach().clone()
#             sents[i, original+1] = temp
#         else:
#             sents[i, original] = sents[i, original - 1].detach().clone()
#             sents[i, original - 1] = temp

#     return sents





def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

def set_seed(seed, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    args = parser.parse_args()
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)
    # logger = setup_logging('main', output_dir)

    set_gpu(args.gpu)
    set_seed(args.seed)

    with logger.session(dir=output_dir, format_strs=['stdout', 'csv', 'log']):
        main(args, output_dir)
