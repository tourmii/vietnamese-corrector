from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig


class ViCorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        src = self.tokenizer(
            row["noisy"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )["input_ids"]
        tgt = self.tokenizer(
            row["gt"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )["input_ids"]
        return {"src_ids": src, "tgt_ids": tgt}


def _collate(batch, pad_id: int):
    src = pad_sequence(
        [torch.tensor(b["src_ids"]) for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    tgt = pad_sequence(
        [torch.tensor(b["tgt_ids"]) for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    return {
        "src": src,
        "tgt": tgt,
        "src_padding_mask": src == pad_id,
        "tgt_padding_mask": tgt == pad_id,
    }


def get_dataloaders(train_cfg: TrainConfig, model_cfg: ModelConfig):
    raw = load_dataset(train_cfg.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name)

    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    model_cfg.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2

    if "validation" in raw:
        train_data = raw["train"]
        val_data = raw["validation"]
    else:
        splits = raw["train"].train_test_split(test_size=train_cfg.val_split, seed=train_cfg.seed)
        train_data = splits["train"]
        val_data = splits["test"]

    if 0 < train_cfg.max_train_samples < len(train_data):
        train_data = train_data.shuffle(seed=train_cfg.seed).select(range(train_cfg.max_train_samples))

    train_ds = ViCorrectionDataset(train_data, tokenizer, model_cfg.max_seq_len)
    val_ds = ViCorrectionDataset(val_data, tokenizer, model_cfg.max_seq_len)

    collate = partial(_collate, pad_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer