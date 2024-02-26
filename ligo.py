
import argparse
import glob
import json
import logging
import os
import pickle
import random
import math
import re
import shutil
import sys
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from model import SimpleBertForMaskedLM, SimpleRobertaForMaskedLM

def is_bert(model):
    return isinstance(model, SimpleBertForMaskedLM)

def is_roberta(model):
    return isinstance(model, SimpleRobertaForMaskedLM)

def num_layer_of(model):
    if is_bert(model):
        return len(model.bert.encoder.layer)
    elif is_roberta(model):
        return len(model.roberta.encoder.layer)
    else:
        raise NotImplementedError

def is_encoder_layer(name):
    return name.startswith('bert.encoder.layer') or name.startswith('roberta.encoder.layer')

K2N = lambda s: '__sm_' + '_'.join(s.split('.'))
N2K = lambda s: '.'.join(s[5:].split('_'))

def normalized_uniform_init(w, init_scheme):
    init_weight = torch.rand_like(w)
    # nn.init.uniform_(init_weight, 0.0, 1.0)
    if 'softmax' in init_scheme:
        init_weight = F.softmax(init_weight, -1) # softmax normalize
    else:
        init_weight = init_weight / torch.sum(init_weight, -1, keepdim=True) # normalize
    w.copy_(init_weight)

def stackbert_init(w, layer_index, init_scheme, init_noise=0.03):
    init_weight = torch.zeros_like(w)
    if 'noisy' in init_scheme:
        init_weight.uniform_(0.0, init_noise)
    init_weight[layer_index % len(init_weight)] = 1.

    init_weight = init_weight / torch.sum(init_weight) # normalize
    w.copy_(init_weight)

def interlace_init(w, layer_index, init_scheme, init_noise=0.03):
    init_weight = torch.zeros_like(w)
    if 'noisy' in init_scheme:
        init_weight.uniform_(0.0, init_noise)

    init_weight[layer_index // 2] = 1.0

    init_weight = init_weight / torch.sum(init_weight) # normalize
    w.copy_(init_weight)

class FusedDepthParams(nn.Module):

    def __init__(self, layer_index, num_layers, bias=False, learnable=True, init_scheme='rand', init_noise=0.03):

        super(FusedDepthParams, self).__init__()

        assert init_scheme in ['rand', 'rand_softmax', 'stackbert', 'interlace', 'stackbert_noisy', 'interlace_noisy']

        self.layer_index = layer_index
        self.init_scheme = init_scheme
        self.init_noise = init_noise

        if learnable:
            self.coeffs_weight = nn.Parameter(torch.zeros(num_layers))
            if bias:
                self.coeffs_bias = nn.Parameter(torch.zeros(num_layers))
            else:
                self.coeffs_bias = None
        else:
            self.register_buffer('coeffs_weight', torch.zeros(num_layers), persistent=True)
            if bias:
                self.register_buffer('coeffs_bias', torch.zeros(num_layers), persistent=True)
            else:
                self.coeffs_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # init depth
        if self.init_scheme in ['rand', 'rand_softmax']:
            normalized_uniform_init(self.coeffs_weight, self.init_scheme)
            if self.coeffs_bias is not None:
                normalized_uniform_init(self.coeffs_bias, self.init_scheme)
        elif self.init_scheme in ['stackbert', 'stackbert_noisy']:
            stackbert_init(self.coeffs_weight, self.layer_index, self.init_scheme, self.init_noise)
            if self.coeffs_bias is not None:
                stackbert_init(self.coeffs_bias, self.layer_index, self.init_scheme, self.init_noise)
        elif self.init_scheme in ['interlace', 'interlace_noisy']:
            interlace_init(self.coeffs_weight, self.layer_index, self.init_scheme, self.init_noise)
            if self.coeffs_bias is not None:
                interlace_init(self.coeffs_bias, self.layer_index, self.init_scheme, self.init_noise)

class FusedWidthParams(nn.Module):

    def __init__(self, small_dim, large_dim, learnable=False, init_scheme='rand', init_noise=0.03):

        super(FusedWidthParams, self).__init__()

        assert init_scheme in ['rand', 'rand_softmax', 'sel', 'sel_noisy']

        self.init_scheme = init_scheme
        self.init_noise = init_noise

        if large_dim - small_dim > 0:
            if learnable:
                self.coeffs_weight = nn.Parameter(torch.zeros(large_dim - small_dim, small_dim))
            else:
                self.register_buffer('coeffs_weight', torch.zeros(large_dim - small_dim, small_dim))
        else:
            self.coeffs_weight = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.coeffs_weight is not None:
            if self.init_scheme in ['rand', 'rand_softmax']:
                normalized_uniform_init(self.coeffs_weight, self.init_scheme)
            elif self.init_scheme in ['sel', 'sel_noisy']:
                sel = torch.randint(0, self.coeffs_weight.shape[1], (self.coeffs_weight.shape[0],))
                init_weight = torch.zeros_like(self.coeffs_weight, dtype=torch.float32)
                if 'noisy' in self.init_scheme:
                    init_weight.uniform_(0.0, self.init_noise)
                init_weight[torch.arange(self.coeffs_weight.shape[0]), sel] = 1.
                self.coeffs_weight.copy_(init_weight)

class FusedLinear(nn.Module):

    def __init__(self, model, module_name, in_features, out_features, layer_index=-1, init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True, residual=False, depth_tie=None, width_in_tie=None, width_out_tie=None, residual_noise=0.01):

        super(FusedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weights for attention layers if depth expansion
        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')
        self.get_bias = lambda: getattr(model, K2N(module_name) + '_bias', None)

        self.bias = (self.get_bias() is not None)

        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))
            if self.bias:
                self.residual_bias = nn.Parameter(torch.empty(self.out_features))
            else:
                self.register_parameter('residual_bias', None)


        # for embedding or classifier layer, specify layer_index to -1
        if layer_index >= 0:
            if depth_tie is None:
                num_layers_small = self.get_weights().shape[-1]
                self.fuse_depth_coeffs = FusedDepthParams(layer_index, num_layers_small, bias=self.bias,
                    learnable=learn_depth, init_scheme=init_scheme_depth, init_noise=init_noise_depth)
                depth_tie = self.fuse_depth_coeffs

            self.get_depth_coeffs = lambda: (depth_tie.coeffs_weight, getattr(depth_tie, 'coeffs_bias', depth_tie.coeffs_weight))

        if width_in_tie is None:
            hidden_dim_small = self.get_weights().shape[:-1] if layer_index >= 0 else self.get_weights().shape
            self.fuse_width_in = FusedWidthParams(hidden_dim_small[1], self.in_features,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width)
            width_in_tie = self.fuse_width_in


        if width_out_tie is None:
            hidden_dim_small = self.get_weights().shape[:-1] if layer_index >= 0 else self.get_weights().shape
            self.fuse_width_out = FusedWidthParams(hidden_dim_small[0], self.out_features,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width)
            width_out_tie = self.fuse_width_out

        self.get_width_coeffs = lambda: (width_in_tie.coeffs_weight, width_out_tie.coeffs_weight)

        self.reset_parameters()

    def reset_parameters(self):

        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)
            if self.bias:
                nn.init.uniform_(self.residual_bias, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'fuse_depth_coeffs'):
            self.fuse_depth_coeffs.reset_parameters()
        if hasattr(self, 'fuse_width_in'):
            self.fuse_width_in.reset_parameters()
        if hasattr(self, 'fuse_width_out'):
            self.fuse_width_out.reset_parameters()

    def get_params(self):
        bias = None

        if hasattr(self, 'get_depth_coeffs'):
            coeffs_weights, coeffs_bias = self.get_depth_coeffs()
            weight = torch.sum(self.get_weights() * coeffs_weights, -1)
            if self.bias:
                bias = torch.sum(self.get_bias() * coeffs_bias, -1)
        else:
            weight = self.get_weights()
            if self.bias:
                bias = self.get_bias()

        in_dim_expand, out_dim_expand = self.get_width_coeffs()
        if in_dim_expand is not None:
            in_dim_expand = torch.transpose(in_dim_expand, 0, 1)
            weight = torch.cat([weight, torch.matmul(weight, in_dim_expand)], 1) # expand in dimension
        if out_dim_expand is not None:
            weight = torch.cat([weight, torch.matmul(out_dim_expand, weight)], 0) # expand out dimension
            if self.bias:
                bias = torch.cat([bias, torch.matmul(out_dim_expand, bias)], 0) # expand out dimension

        if self.residual:
            weight = weight + self.residual_weight
            if self.bias:
                bias = bias + self.residual_bias

        return weight, bias

    def forward(self, input):
        weight, bias = self.get_params()
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class FusedLayeredNorm(nn.Module):

    def __init__(self, model, module_name, normalized_shape, layer_index=-1, init_scheme_depth='rand', init_noise_depth=0.03, learn_depth=True,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True, eps=1e-5, residual=False, depth_tie=None, width_out_tie=None, residual_noise=0.01):

        super(FusedLayeredNorm, self).__init__()
        
        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')
        self.get_bias = lambda: getattr(model, K2N(module_name) + '_bias', None)

        self.normalized_shape = normalized_shape
        self.elementwise_affine = True # only support elementwise_affine
        self.eps = eps
        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty(self.normalized_shape))
            self.residual_bias = nn.Parameter(torch.empty(self.normalized_shape))

        # for embedding or classifier layer, specify layer_index to -1
        if layer_index >= 0:
            if depth_tie is None:
                num_layers_small = self.get_weights().shape[-1]
                self.fuse_depth_coeffs = FusedDepthParams(layer_index, num_layers_small, bias=True,
                    learnable=learn_depth, init_scheme=init_scheme_depth, init_noise=init_noise_depth)
                depth_tie = self.fuse_depth_coeffs

            self.get_depth_coeffs = lambda: (depth_tie.coeffs_weight, getattr(depth_tie, 'coeffs_bias', depth_tie.coeffs_weight))

        if width_out_tie is None:
            hidden_dim_small = self.get_weights().shape[:-1] if layer_index >= 0 else self.get_weights().shape
            self.fuse_width_out = FusedWidthParams(hidden_dim_small[0], self.normalized_shape[0],
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width)
            width_out_tie = self.fuse_width_out

        self.get_width_coeffs = lambda: width_out_tie.coeffs_weight

    def reset_parameters(self):

        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)
            nn.init.uniform_(self.residual_bias, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'fuse_depth_coeffs'):
            self.fuse_depth_coeffs.reset_parameters()
        if hasattr(self, 'fuse_width_out'):
            self.fuse_width_out.reset_parameters()

    def get_params(self):

        if hasattr(self, 'get_depth_coeffs'):
            coeffs_weights, coeffs_bias = self.get_depth_coeffs()
            weight = torch.sum(self.get_weights() * coeffs_weights, -1)
            bias = torch.sum(self.get_bias() * coeffs_bias, -1)

        else:
            weight = self.get_weights()
            bias = self.get_bias()

        out_dim_expand = self.get_width_coeffs()
        if out_dim_expand is not None:
            weight = torch.cat([weight, torch.matmul(out_dim_expand, weight)], 0) # expand out dimension
            bias = torch.cat([bias, torch.matmul(out_dim_expand, bias)], 0) # expand out dimension

        if self.residual:
            weight = weight + self.residual_weight
            bias = bias + self.residual_bias

        return weight, bias

    def forward(self, input):
        weight, bias = self.get_params()
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
    
    def extra_repr(self):
            return '{}, eps={}, elementwise_affine={}'.format(self.normalized_shape, self.eps, self.elementwise_affine)

class FusedEmbedding(nn.Module):

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, model, module_name, num_embeddings, embedding_dim, padding_idx = None,
        max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
        init_scheme_width='rand', init_noise_width=0.03, learn_width=True,
        residual=False, width_out_tie=None, residual_noise=0.01):

        super(FusedEmbedding, self).__init__()

        self.get_weights = lambda: getattr(model, K2N(module_name) + '_weight')

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.residual = residual
        self.residual_noise = residual_noise

        if residual:
            self.residual_weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))

        if width_out_tie is None:
            hidden_dim_small = self.get_weights().shape
            self.fuse_width_out = FusedWidthParams(hidden_dim_small[1], self.embedding_dim,
                learnable=learn_width, init_scheme=init_scheme_width, init_noise=init_noise_width)
            width_out_tie = self.fuse_width_out

        self.get_width_coeffs = lambda: width_out_tie.coeffs_weight

        self.sparse = sparse

    def reset_parameters(self):
        if self.residual:
            nn.init.uniform_(self.residual_weight, -self.residual_noise, self.residual_noise)

        if hasattr(self, 'fuse_width_out'):
            self.fuse_width_out.reset_parameters()

    def get_params(self):

        weight = self.get_weights()

        out_dim_expand = self.get_width_coeffs()
        if out_dim_expand is not None:
            out_dim_expand = torch.transpose(out_dim_expand, 0, 1)
            weight = torch.cat([weight, torch.matmul(weight, out_dim_expand)], 1) # expand out dimension

        if self.residual:
            weight = weight + self.residual_weight

        return weight

    def forward(self, input):
        weight = self.get_params()
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

@torch.no_grad()
def create_ligo_from_model(model_large, args, source_model=None):

    # load source model from the setting
    if source_model is None:
        assert len(args.source_model_path) == 1, 'Not support multiple model.'

        source_model_path = args.source_model_path[0]
        # Small model
        if source_model_path:
            small_config = AutoConfig.from_pretrained(source_model_path, cache_dir=args.cache_dir)
        else:
            raise ValueError("No config for small model is specified.")
        model_small = model_large.__class__.from_pretrained(
            source_model_path,
            from_tf=bool(".ckpt" in source_model_path),
            config=small_config,
            cache_dir=args.cache_dir,
        )

    # directly use given model
    else:
        model_small = source_model

    dict_model_small = model_small.state_dict()

    # save map from module to name
    dict_M2N = {}
    for name, module in model_large.named_modules():
        # if not name.startswith('bert.encoder.layer'):
        if not is_encoder_layer(name):
            dict_M2N[id(module)] = name
        else:
            dict_M2N[id(module)] = '.'.join(name.split('.')[4:])
    M2N = lambda m: dict_M2N[id(m)]

    # extract parameters of embedding and classifier
    # NOTE use state_dict can automatically break the tie between embedding and classifier
    for name, param in dict_model_small.items():
        if not is_encoder_layer(name):
            if args.tune_small_model:
                model_large.register_parameter(K2N(name), nn.Parameter(param, requires_grad=True))
            else:
                model_large.register_buffer(K2N(name), param, persistent=True)

    # extract parameters of same module at different layers
    if is_bert(model_small):
        enc_layers = model_small.bert.encoder.layer
        template_key = 'bert.encoder.layer'
    elif is_roberta(model_small):
        enc_layers = model_small.roberta.encoder.layer
        template_key = 'roberta.encoder.layer'
    else:
        raise NotImplementedError
    for name, param in enc_layers[0].named_parameters():
        weight_list = []
        for l, _ in enumerate(enc_layers):
            k = f'{template_key}.{l}.{name}'
            weight_list.append(dict_model_small[k])
        w = torch.stack(weight_list, -1)
        if args.tune_small_model:
            model_large.register_parameter(K2N(name), nn.Parameter(w, requires_grad=True))
        else:
            model_large.register_buffer(K2N(name), w, persistent=True)

    def create_embed_layer(module_large, args, width_out_tie=None):
        return FusedEmbedding(model_large, M2N(module_large), module_large.num_embeddings, module_large.embedding_dim,
            padding_idx=module_large.padding_idx, max_norm=module_large.max_norm, norm_type=module_large.norm_type,
            scale_grad_by_freq=module_large.scale_grad_by_freq, sparse=module_large.sparse,
            init_scheme_width=args.fuse_init_scheme_width, init_noise_width=args.fuse_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, width_out_tie=width_out_tie
        )

    def create_lin_layer(module_large, args, layer_index=-1, depth_tie=None, width_in_tie=None, width_out_tie=None):
        return FusedLinear(model_large, M2N(module_large), module_large.in_features, module_large.out_features,
            layer_index=layer_index, init_scheme_depth=args.fuse_init_scheme_depth, init_noise_depth=args.fuse_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.fuse_init_scheme_width, init_noise_width=args.fuse_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_in_tie=width_in_tie, width_out_tie=width_out_tie
        )

    def create_ln_layer(module_large, args, layer_index=-1, depth_tie=None, width_out_tie=None):
        return FusedLayeredNorm(model_large, M2N(module_large), module_large.normalized_shape, eps=module_large.eps, 
            layer_index=layer_index, init_scheme_depth=args.fuse_init_scheme_depth, init_noise_depth=args.fuse_init_noise_depth, learn_depth=args.tune_depth,
            init_scheme_width=args.fuse_init_scheme_width, init_noise_width=args.fuse_init_noise_width, learn_width=args.tune_width,
            residual=args.tune_residual, residual_noise=args.tune_residual_noise, depth_tie=depth_tie, width_out_tie=width_out_tie
        )

    #### Bert2Bert style coefficient tying

    kwargs_depth_param = dict(learnable=args.tune_depth, init_scheme=args.fuse_init_scheme_depth, init_noise=args.fuse_init_noise_depth)
    kwargs_width_param = dict(learnable=args.tune_width, init_scheme=args.fuse_init_scheme_width, init_noise=args.fuse_init_noise_width)

    # Embedding module
    if is_bert(model_small) and is_bert(model_large):
        emb_small, emb_large = model_small.bert.embeddings, model_large.bert.embeddings
    elif is_roberta(model_small) and is_roberta(model_large):
        emb_small, emb_large = model_small.roberta.embeddings, model_large.roberta.embeddings
    else:
        raise NotImplementedError

    if args.fuse_tie_param:
        setattr(emb_large, 'fuse_width_emb', FusedWidthParams(emb_small.word_embeddings.weight.shape[-1], emb_large.word_embeddings.weight.shape[-1], **kwargs_width_param))
    else:
        setattr(emb_large, 'fuse_width_emb', None)

    g_e = getattr(emb_large, 'fuse_width_emb')
    setattr(emb_large, 'word_embeddings', create_embed_layer(emb_large.word_embeddings, args, width_out_tie=g_e))
    setattr(emb_large, 'position_embeddings', create_embed_layer(emb_large.position_embeddings, args, width_out_tie=g_e))
    setattr(emb_large, 'token_type_embeddings', create_embed_layer(emb_large.token_type_embeddings, args, width_out_tie=g_e))
    setattr(emb_large, 'LayerNorm', create_ln_layer(emb_large.LayerNorm, args, layer_index=-1, width_out_tie=g_e))

    # Encoder layers
    gs = [] # index of selected columns
    if is_bert(model_small) and is_bert(model_large):
        small_layers, large_layers = model_small.bert.encoder.layer, model_large.bert.encoder.layer
    elif is_roberta(model_small) and is_roberta(model_large):
        small_layers, large_layers = model_small.roberta.encoder.layer, model_large.roberta.encoder.layer
    else:
        raise NotImplementedError

    for i, l_large in enumerate(large_layers):
        if args.fuse_tie_param:
            setattr(l_large, 'fuse_width_key', FusedWidthParams(small_layers[0].attention.self.key.weight.shape[0], l_large.attention.self.key.weight.shape[0], **kwargs_width_param))
            setattr(l_large, 'fuse_width_query', FusedWidthParams(small_layers[0].attention.self.query.weight.shape[0], l_large.attention.self.query.weight.shape[0], **kwargs_width_param))
            setattr(l_large, 'fuse_width_value', FusedWidthParams(small_layers[0].attention.self.value.weight.shape[0], l_large.attention.self.value.weight.shape[0], **kwargs_width_param))
            setattr(l_large, 'fuse_width_ffn', FusedWidthParams(small_layers[0].intermediate.dense.weight.shape[0], l_large.intermediate.dense.weight.shape[0], **kwargs_width_param))
        else:
            setattr(l_large, 'fuse_width_key', None)
            setattr(l_large, 'fuse_width_query', None)
            setattr(l_large, 'fuse_width_value', None)
            setattr(l_large, 'fuse_width_ffn', None)

        # MHA - Attention
        attn_large = l_large.attention.self
        for name in ['query', 'key', 'value']:
            setattr(attn_large, name, create_lin_layer(getattr(attn_large, name), args, layer_index=i, width_in_tie=g_e, width_out_tie=getattr(l_large, f'fuse_width_{name}')))

        # MHA - W_o
        setattr(l_large.attention.output, 'dense', create_lin_layer(l_large.attention.output.dense, args, layer_index=i, width_in_tie=getattr(l_large, f'fuse_width_value'), width_out_tie=g_e))

        # MHA - LayerNorm
        setattr(l_large.attention.output, 'LayerNorm', create_ln_layer(l_large.attention.output.LayerNorm, args, layer_index=i, width_out_tie=g_e))

        # FFN - Layer 1
        setattr(l_large.intermediate, 'dense', create_lin_layer(l_large.intermediate.dense, args, layer_index=i, width_in_tie=g_e, width_out_tie=getattr(l_large, 'fuse_width_ffn')))

        # FFN - Layer 2
        setattr(l_large.output, 'dense', create_lin_layer(l_large.output.dense, args, layer_index=i, width_in_tie=getattr(l_large, 'fuse_width_ffn'), width_out_tie=g_e))

        # FFN LayerNorm
        setattr(l_large.output, 'LayerNorm', create_ln_layer(l_large.output.LayerNorm, args, layer_index=i, width_out_tie=g_e))

    # Classifier
    if is_bert(model_small) and is_bert(model_large):
        cls_small, cls_large = model_small.cls.predictions, model_large.cls.predictions
        if args.fuse_tie_param:
            setattr(cls_large, 'fuse_width_cls', FusedWidthParams(cls_small.transform.dense.weight.shape[0], cls_large.transform.dense.weight.shape[0], **kwargs_width_param))
        else:
            setattr(cls_large, 'fuse_width_cls', None)

        setattr(cls_large.transform, 'dense', create_lin_layer(cls_large.transform.dense, args, layer_index=-1, width_in_tie=g_e, width_out_tie=getattr(cls_large, 'fuse_width_cls')))
        setattr(cls_large.transform, 'LayerNorm', create_ln_layer(cls_large.transform.LayerNorm, args, layer_index=-1, width_out_tie=getattr(cls_large, 'fuse_width_cls')))
        setattr(cls_large, 'decoder', create_lin_layer(cls_large.decoder, args, layer_index=-1, width_in_tie=getattr(cls_large, 'fuse_width_cls'), width_out_tie=None))
    
    elif is_roberta(model_small) and is_roberta(model_large):
        cls_small, cls_large = model_small.lm_head, model_large.lm_head
        if args.fuse_tie_param:
            setattr(cls_large, 'fuse_width_cls', FusedWidthParams(cls_small.dense.weight.shape[0], cls_large.dense.weight.shape[0], **kwargs_width_param))
        else:
            setattr(cls_large, 'fuse_width_cls', None)

        setattr(cls_large, 'dense', create_lin_layer(cls_large.dense, args, layer_index=-1, width_in_tie=g_e, width_out_tie=getattr(cls_large, 'fuse_width_cls')))
        setattr(cls_large, 'layer_norm', create_ln_layer(cls_large.layer_norm, args, layer_index=-1, width_out_tie=getattr(cls_large, 'fuse_width_cls')))
        setattr(cls_large, 'decoder', create_lin_layer(cls_large.decoder, args, layer_index=-1, width_in_tie=getattr(cls_large, 'fuse_width_cls'), width_out_tie=None))
    
    else:
        raise NotImplementedError

    return model_large


@torch.no_grad()
def initialize_model_with_ligo(model_large, args):

    # load coefficient model
    coeff_model_path = args.pretrained_ligo_path
    dict_model_coeff = torch.load(os.path.join(coeff_model_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    model_coeff = model_large.__class__(config=model_large.config, args=args)
    model_coeff = create_ligo_from_model(model_coeff, args)
    model_coeff.load_state_dict(dict_model_coeff)

    modules_coeff = {name:module for name, module in model_coeff.named_modules()}
    for name, module in model_large.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.weight.copy_(modules_coeff[name].get_params()[0])
            if hasattr(module, 'bias'):
                module.bias.copy_(modules_coeff[name].get_params()[1])
        elif isinstance(module, nn.Embedding):
            module.weight.copy_(modules_coeff[name].get_params())

    return model_large
