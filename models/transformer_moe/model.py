import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x), x.new_tensor(0.0)


class MoELayer(nn.Module):
    """Top-K sparse Mixture of Experts with load-balancing auxiliary loss."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.routing_cache: list | None = None
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor):
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)
        N = x_flat.size(0)

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if self.routing_cache is not None:
            self.routing_cache.append(indices.detach().cpu())

        # load-balancing auxiliary loss (Switch Transformer style)
        counts = x_flat.new_zeros(self.num_experts)
        for k in range(self.top_k):
            counts.scatter_add_(0, indices[:, k], x_flat.new_ones(N))
        f = counts / (N * self.top_k)
        p = probs.mean(0)
        aux_loss = self.num_experts * (f * p).sum()

        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            for k in range(self.top_k):
                mask = indices[:, k] == e
                if not mask.any():
                    continue
                out = self.experts[e](x_flat[mask])
                output[mask] = output[mask] + weights[mask, k : k + 1] * out

        return output.reshape(B, T, d), aux_loss


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        use_moe: bool,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = (
            MoELayer(d_model, d_ff, num_experts, top_k, dropout)
            if use_moe
            else FFN(d_model, d_ff, dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None):
        attn, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.drop(attn))
        ffn_out, aux = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, aux


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        use_moe: bool,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = (
            MoELayer(d_model, d_ff, num_experts, top_k, dropout)
            if use_moe
            else FFN(d_model, d_ff, dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        sa, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.drop(sa))
        ca, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.drop(ca))
        ffn_out, aux = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)
        return tgt, aux


class TransformerMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.scale = math.sqrt(cfg.d_model)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.d_ff,
                    cfg.num_experts,
                    cfg.top_k,
                    cfg.dropout,
                    use_moe=(i % cfg.moe_every_n == 0),
                )
                for i in range(cfg.num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.d_ff,
                    cfg.num_experts,
                    cfg.top_k,
                    cfg.dropout,
                    use_moe=(i % cfg.moe_every_n == 0),
                )
                for i in range(cfg.num_decoder_layers)
            ]
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tgt_embed.weight  # weight tying
        self._init_weights()

    # ── routing cache helpers ─────────────────────────────────────────────────
    def _moe_layers(self):
        return {name: m for name, m in self.named_modules() if isinstance(m, MoELayer)}

    def enable_routing_cache(self):
        for m in self._moe_layers().values():
            m.routing_cache = []

    def disable_routing_cache(self):
        for m in self._moe_layers().values():
            m.routing_cache = None

    def clear_routing_cache(self):
        for m in self._moe_layers().values():
            if m.routing_cache is not None:
                m.routing_cache.clear()

    def get_routing_cache(self) -> dict:
        out = {}
        for name, m in self._moe_layers().items():
            if m.routing_cache:
                out[name] = torch.cat(m.routing_cache, dim=0)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, src: torch.Tensor, src_padding_mask=None):
        x = self.pos_enc(self.src_embed(src) * self.scale)
        aux = x.new_tensor(0.0)
        for layer in self.encoder:
            x, a = layer(x, src_key_padding_mask=src_padding_mask)
            aux = aux + a
        return x, aux

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        tgt_padding_mask=None,
        src_padding_mask=None,
    ):
        x = self.pos_enc(self.tgt_embed(tgt) * self.scale)
        aux = x.new_tensor(0.0)
        for layer in self.decoder:
            x, a = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            aux = aux + a
        return x, aux

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        T = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        memory, enc_aux = self.encode(src, src_padding_mask)
        out, dec_aux = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        logits = self.lm_head(out)
        return logits, enc_aux + dec_aux

    @torch.no_grad()
    def generate(self, src, src_padding_mask, max_len: int = 256, beam_size: int = 1):
        """Greedy decoding (beam_size=1) — extend for beam search as needed."""
        device = src.device
        B = src.size(0)
        memory, _ = self.encode(src, src_padding_mask)

        ys = src.new_full((B, 1), self.cfg.bos_token_id)
        done = src.new_zeros(B, dtype=torch.bool)

        for _ in range(max_len):
            T = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            out, _ = self.decode(ys, memory, tgt_mask, None, src_padding_mask)
            next_tok = self.lm_head(out[:, -1]).argmax(-1)
            done = done | (next_tok == self.cfg.eos_token_id)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            if done.all():
                break

        return ys[:, 1:]