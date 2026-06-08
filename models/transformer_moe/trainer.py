import os

import torch
import torch.nn as nn
import sacrebleu
import wandb
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from model import TransformerMoE

EVAL_STRATEGIES = ("epoch", "steps", "no")


class Trainer:
    def __init__(
        self,
        model: TransformerMoE,
        train_loader,
        val_loader,
        tokenizer,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
    ):
        assert train_cfg.eval_strategy in EVAL_STRATEGIES, \
            f"eval_strategy must be one of {EVAL_STRATEGIES}"

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.cfg = train_cfg
        self.model_cfg = model_cfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        self.scaler = torch.amp.GradScaler(enabled=train_cfg.fp16)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model_cfg.pad_token_id,
            label_smoothing=train_cfg.label_smoothing,
        )

        self.global_step = 0
        self.best_bleu = 0.0
        os.makedirs(train_cfg.save_dir, exist_ok=True)

    # ── learning rate (Noam schedule) ──────────────────────────────────────────
    def _noam_lr(self, step: int) -> float:
        step = max(step, 1)
        d = self.model_cfg.d_model
        w = self.cfg.warmup_steps
        return (d ** -0.5) * min(step ** -0.5, step * w ** -1.5)

    def _set_lr(self):
        lr = self._noam_lr(self.global_step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    # ── single forward/loss ────────────────────────────────────────────────────
    def _forward(self, batch, device):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        dec_in  = tgt[:, :-1]
        dec_tgt = tgt[:, 1:]
        dec_mask = tgt_mask[:, :-1]

        with torch.amp.autocast(device_type="cuda", enabled=self.cfg.fp16):
            logits, aux_loss = self.model(src, dec_in, src_mask, dec_mask)
            B, T, V = logits.shape
            ce   = self.criterion(logits.reshape(B * T, V), dec_tgt.reshape(-1))
            loss = ce + self.cfg.aux_loss_weight * aux_loss

        return loss, ce.item(), aux_loss.item()

    # ── training epoch ─────────────────────────────────────────────────────────
    def train_epoch(self, device: torch.device, epoch: int):
        self.model.train()
        running_ce = running_aux = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for batch in pbar:
            self._set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            loss, ce, aux = self._forward(batch, device)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_ce  += ce
            running_aux += aux
            self.global_step += 1
            pbar.set_postfix(ce=f"{ce:.4f}", aux=f"{aux:.4f}")

            if self.global_step % self.cfg.log_steps == 0:
                wandb.log({
                    "train/ce_loss": ce,
                    "train/aux_loss": aux,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step,
                })

            if self.cfg.eval_strategy == "steps" and self.global_step % self.cfg.eval_steps == 0:
                self._run_eval(device, tag=f"step {self.global_step}")
                self.model.train()

        n = len(self.train_loader)
        return running_ce / n, running_aux / n

    # ── evaluation helper ──────────────────────────────────────────────────────
    def _run_eval(self, device: torch.device, tag: str = "") -> float:
        bleu = self.evaluate(device)
        wandb.log({"val/bleu": bleu, "step": self.global_step})
        print(f"\n[{tag}] BLEU = {bleu:.2f}")
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.save("best.pt")
            print(f"  → new best saved ({bleu:.2f})")
        return bleu

    # ── evaluation ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, device: torch.device) -> float:
        self.model.eval()
        hyps, refs = [], []

        for batch in tqdm(self.val_loader, desc="Eval", leave=False, dynamic_ncols=True):
            src      = batch["src"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            preds    = self.model.generate(src, src_mask, max_len=self.model_cfg.max_seq_len)

            for pred, ref_ids in zip(preds, batch["tgt"]):
                hyps.append(self.tokenizer.decode(pred.tolist(),    skip_special_tokens=True))
                refs.append(self.tokenizer.decode(ref_ids.tolist(), skip_special_tokens=True))

        return sacrebleu.corpus_bleu(hyps, [refs]).score

    # ── checkpointing ──────────────────────────────────────────────────────────
    def save(self, name: str):
        path = os.path.join(self.cfg.save_dir, name)
        torch.save({
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict(),
            "step":      self.global_step,
            "best_bleu": self.best_bleu,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        self.best_bleu   = ckpt["best_bleu"]
        print(f"Loaded checkpoint — step={self.global_step}  best_bleu={self.best_bleu:.2f}")

    # ── main loop ──────────────────────────────────────────────────────────────
    def fit(self, device: torch.device, resume: str | None = None):
        self.model.to(device)
        if resume:
            self.load(resume)

        wandb.init(
            project="vi-correction-moe",
            config={**vars(self.cfg), **vars(self.model_cfg)},
        )

        strategy = self.cfg.eval_strategy
        print(f"Eval strategy : {strategy}" +
              (f"  (every {self.cfg.eval_steps} steps)" if strategy == "steps" else ""))

        for epoch in range(self.cfg.num_epochs):
            ce, aux = self.train_epoch(device, epoch)
            print(f"Epoch {epoch + 1:02d}: ce={ce:.4f}  aux={aux:.4f}")

            # always save by epoch
            self.save(f"epoch_{epoch + 1:02d}.pt")

            # eval at end of epoch (if strategy="epoch")
            if strategy == "epoch":
                self._run_eval(device, tag=f"epoch {epoch + 1}")

        wandb.finish()