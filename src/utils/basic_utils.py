import os
import random
import yaml

import numpy as np
import torch

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cfg(root_dir):
    cfg_path = os.path.join(root_dir, "configs/config.yaml")
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError:
        print("[INFO] \t ERROR: 'Config.yaml' IS NOT FOUND")

def merge_args_cfg(args, cfg):
    if args.model != None:
        cfg["model"]["model_name"] = args.model
    if args.dataset != None:
        cfg["data"]["dataset_name"] = args.dataset
    if args.epochs != None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size != None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr != None:
        cfg["train"]["lr"] = args.lr
    if args.lr_decay != None:
        cfg["train"]["lr_decay"] = args.lr_decay
    if args.momentum != None:
        cfg["train"]["momentum"] = args.momentum
    if args.wd != None:
        cfg["train"]["wd"] = args.wd
    if args.betas != None:
        cfg["train"]["betas"] = args.betas
    if args.resume != None:
        cfg["train"]["resume"] = args.resume
    return cfg

def merge_test_args_cfg(args, cfg):
    if args.model != None:
        cfg["model"]["model_name"] = args.model
    if args.dataset != None:
        cfg["data"]["dataset_name"] = args.dataset
    if args.batch_size != None:
        cfg["train"]["batch_size"] = args.batch_size
    
    return cfg