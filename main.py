import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd
import os
from training import train_model
import json
from testing import test_model
from finetune import finetune_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--hyperedge_dim_hidden', type=int, default=64)
    parser.add_argument('--hyper_node_dim_hidden', type=int, default=64)
    parser.add_argument('--gnn_layer_num', type=int, default=3)
    parser.add_argument('--imputer_layer_num', type=int, default=1)
    parser.add_argument('--gnn_activation', type=str, default='relu')
    # parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.6) # 1 - edge dropout rate
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_epoch_gap', type=int, default=500)

    parser.add_argument('--delta', type=float, default=1)

    parser.add_argument('--data', type=str, default='wine')
    parser.add_argument('--missing_ratio', type=float, default=0.2)
    parser.add_argument('--missing_mechanism', type=str, default='MCAR')

    parser.add_argument('--plot_name', type=str, default='v0')
    parser.add_argument('--save_name', type=str, default='v0')
    parser.add_argument('--load_model_name', type=str, default='None')

    parser.add_argument('--header_type', type=str, default='Linear')
    parser.add_argument('--bs_embedding', type=int, default=32)
    parser.add_argument('--device', type=int, default=0, help='Device cuda id')
    parser.add_argument('--chunk_size', type=int, default=500, help='the number of row in each chunk')
    parser.add_argument('--chunk_batch', type=int, default=32, help='the number of chunk in each batch')

    parser.add_argument('--save_emb', action='store_true', default=False)
    parser.add_argument('--load_emb', action='store_true', default=False)
    parser.add_argument('--llm_path', type=str, default='../models_hf/llama2_7b/')

    parser.add_argument('--relation_type', type=str, default='cross_attn')
    parser.add_argument('--mode', type=str, default='training')
    
    args = parser.parse_args()

    f = open(args.llm_path+"config.json", 'r')
    config = json.load(f)
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(args)
    
    device = torch.device(f'cuda:{args.device}')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if args.test_mode:
    #     # print("start test model...")
    #     test_model(args, device)
    # else:
    #     # print("start train model...")
    #     train_model(args, device)
    if args.mode == "training":
        train_model(args, device)
    elif args.mode == "finetune":
        finetune_model(args, device)
    elif args.mode == "testing":
        test_model(args, device)
    else:
        raise ValueError("Invalid mode")


if __name__ == '__main__':
    main()

    