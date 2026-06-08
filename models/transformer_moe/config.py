from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 64000
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 256
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    num_experts: int = 8
    top_k: int = 2
    moe_every_n: int = 2


@dataclass
class TrainConfig:
    dataset_name: str = "tourmii/vietnamese-corrector-errors"
    tokenizer_name: str = "vinai/bartpho-syllable-base"
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 3e-4
    num_epochs: int = 20
    warmup_steps: int = 4000
    aux_loss_weight: float = 0.01
    label_smoothing: float = 0.1
    save_dir: str = "./checkpoints"
    log_steps: int = 100
    eval_strategy: str = "epoch"  # "epoch" | "steps" | "no"
    eval_steps: int = 2000        # used only when eval_strategy="steps"
    fp16: bool = True
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    val_split: float = 0.01
    max_train_samples: int = -1  # -1 = use full dataset