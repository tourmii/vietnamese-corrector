import argparse
import random

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from dataset import get_dataloaders
from model import TransformerMoE
from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total / 1e6:.1f}M  trainable={trainable / 1e6:.1f}M"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",            type=str,   default=None)
    parser.add_argument("--dataset",           type=str,   default=None)
    parser.add_argument("--tokenizer",         type=str,   default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--batch_size",        type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None)
    parser.add_argument("--num_experts",       type=int,   default=None)
    parser.add_argument("--top_k",             type=int,   default=None)
    parser.add_argument("--moe_every_n",       type=int,   default=None)
    parser.add_argument("--max_train_samples", type=int,   default=None,
                        help="Subsample N rows from train set. -1 = full dataset.")
    parser.add_argument("--eval_strategy",    type=str,   default=None,
                        choices=["epoch", "steps", "no"],
                        help="When to evaluate: epoch | steps | no")
    parser.add_argument("--eval_steps",       type=int,   default=None,
                        help="Eval every N steps (used when eval_strategy=steps)")
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.dataset:           train_cfg.dataset_name      = args.dataset
    if args.tokenizer:         train_cfg.tokenizer_name    = args.tokenizer
    if args.epochs:            train_cfg.num_epochs        = args.epochs
    if args.batch_size:        train_cfg.batch_size        = args.batch_size
    if args.lr:                train_cfg.lr                = args.lr
    if args.num_experts:       model_cfg.num_experts       = args.num_experts
    if args.top_k:             model_cfg.top_k             = args.top_k
    if args.moe_every_n:       model_cfg.moe_every_n       = args.moe_every_n
    if args.max_train_samples is not None:
                               train_cfg.max_train_samples = args.max_train_samples
    if args.eval_strategy:     train_cfg.eval_strategy     = args.eval_strategy
    if args.eval_steps:        train_cfg.eval_steps        = args.eval_steps
    if args.no_fp16:           train_cfg.fp16              = False

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, tokenizer = get_dataloaders(train_cfg, model_cfg)
    print(f"Train: {len(train_loader.dataset):,} samples | Val: {len(val_loader.dataset):,} samples")

    model = TransformerMoE(model_cfg)
    print(f"Model params: {count_params(model)}")

    moe_layers = sum(
        1 for i in range(model_cfg.num_encoder_layers + model_cfg.num_decoder_layers)
        if i % model_cfg.moe_every_n == 0
    )
    print(
        f"MoE layers: {moe_layers} / {model_cfg.num_encoder_layers + model_cfg.num_decoder_layers}"
        f"  |  experts: {model_cfg.num_experts}  |  top-k: {model_cfg.top_k}"
    )

    trainer = Trainer(model, train_loader, val_loader, tokenizer, train_cfg, model_cfg)
    trainer.fit(device, resume=args.resume)


if __name__ == "__main__":
    main()