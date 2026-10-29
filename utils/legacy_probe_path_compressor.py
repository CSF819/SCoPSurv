# Archived verbatim from repository commit d954198 for legacy checkpoint inference.
"""Gene-free WSI path compression blocks for CFSurv."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFSurvSlotAttention(nn.Module):
    """WSI slot attention for gene-free pathology tokens."""

    def __init__(
        self,
        num_slots: int,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: Optional[int] = 128,
        train_init_noise: bool = True,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.iters = int(iters)
        self.eps = float(eps)
        self.train_init_noise = bool(train_init_noise)
        self.scale = self.dim ** -0.5

        inner_dim = self.heads * self.dim_head
        hidden_dim = max(self.dim, int(hidden_dim or self.dim))

        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.norm_input = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)

        self.to_q = nn.Linear(self.dim, inner_dim)
        self.to_k = nn.Linear(self.dim, inner_dim)
        self.to_v = nn.Linear(self.dim, inner_dim)
        self.combine_heads = nn.Linear(inner_dim, self.dim)

        self.gru = nn.GRUCell(self.dim, self.dim)
        self.norm_pre_ff = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, _ = x.shape
        return x.view(bsz, tokens, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        bsz, heads, tokens, dim_head = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, tokens, heads * dim_head)

    def forward(self, inputs: torch.Tensor, return_attn: bool = False):
        if inputs.dim() != 3:
            raise ValueError(f"Expected inputs [B, N, D], got {tuple(inputs.shape)}")

        bsz, _, dim = inputs.shape
        if dim != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {dim}")

        mu = self.slots_mu.expand(bsz, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(bsz, self.num_slots, -1)
        if self.training and self.train_init_noise:
            slots = mu + sigma * torch.randn(mu.shape, device=inputs.device, dtype=inputs.dtype)
        else:
            slots = mu

        inputs = self.norm_input(inputs)
        keys = self._split_heads(self.to_k(inputs))
        values = self._split_heads(self.to_v(inputs))

        last_attn = None
        for _ in range(self.iters):
            slots_prev = slots
            queries = self._split_heads(self.to_q(self.norm_slots(slots)))
            dots = torch.einsum("bhsd,bhnd->bhsn", queries, keys) * self.scale

            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-1)
            last_attn = attn

            updates = torch.einsum("bhsn,bhnd->bhsd", attn, values)
            updates = self.combine_heads(self._merge_heads(updates))

            slots = self.gru(
                updates.reshape(bsz * self.num_slots, self.dim),
                slots_prev.reshape(bsz * self.num_slots, self.dim),
            ).view(bsz, self.num_slots, self.dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        if return_attn:
            return slots, last_attn
        return slots


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(t)
    return -log(-log(noise))


def relaxed_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    scores = logits
    soft_k_hot = torch.zeros_like(logits)

    for _ in range(k):
        probs = F.softmax(scores / temperature, dim=-1)
        soft_k_hot = soft_k_hot + probs
        scores = scores + torch.log((1.0 - probs).clamp(min=1e-20))

    return soft_k_hot


def gumbel_topk_st(
    logits: torch.Tensor,
    k: int = 1,
    temperature: float = 1.0,
    add_noise: bool = True,
):
    noised_logits = logits + gumbel_noise(logits) if add_noise else logits
    topk_indices = noised_logits.topk(k=k, dim=-1).indices

    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)

    soft_k_hot = relaxed_topk(noised_logits, k=k, temperature=temperature)
    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


def parallel_topk_st(
    logits: torch.Tensor,
    k: int = 1,
    temperature: float = 1.0,
    add_noise: bool = True,
):
    noised_logits = logits + gumbel_noise(logits) if add_noise else logits
    topk_indices = noised_logits.topk(k=k, dim=-1).indices

    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)

    soft = F.softmax(noised_logits / temperature, dim=-1)
    soft_k_hot = k * soft

    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


class CFSurvPathSlotCompressor(nn.Module):
    """Slot-based, gene-free WSI compressor for CFSurv path_vec."""

    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 16,
        iters: int = 3,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.01,
        topk_ratio: float = 0.25,
        top_k_method: str = "parallel_topk_st",
        train_init_noise: bool = True,
    ):
        super().__init__()

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)
        self.top_k_method = str(top_k_method)
        if self.top_k_method not in {"gumbel_topk_st", "parallel_topk_st"}:
            raise ValueError(f"Invalid top_k_method: {self.top_k_method}")

        safe_ratio = min(max(self.topk_ratio, 0.0), 1.0)
        self.keep_k = max(1, int(self.num_slots * safe_ratio))

        self.slot_attention = CFSurvSlotAttention(
            num_slots=self.num_slots,
            dim=self.dim,
            heads=heads,
            dim_head=dim_head,
            iters=iters,
            train_init_noise=train_init_noise,
        )

        self.slot_map = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        self.slot_score = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1),
        )

    def forward(self, z_tokens: torch.Tensor):
        if z_tokens.dim() != 3:
            raise ValueError(f"Expected z_tokens [B, N, D], got {tuple(z_tokens.shape)}")

        _, _, dim = z_tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {dim}")

        slots, token_attn = self.slot_attention(z_tokens, return_attn=True)
        slots = self.slot_map(slots)

        keep_logits = self.slot_score(slots).squeeze(-1)
        if self.keep_k < self.num_slots:
            if self.top_k_method == "gumbel_topk_st":
                keep_mask, _ = gumbel_topk_st(
                    keep_logits,
                    k=self.keep_k,
                    temperature=self.temperature,
                    add_noise=self.training,
                )
            elif self.top_k_method == "parallel_topk_st":
                keep_mask, _ = parallel_topk_st(
                    keep_logits,
                    k=self.keep_k,
                    temperature=self.temperature,
                    add_noise=self.training,
                )
            else:
                raise ValueError(f"Invalid top_k_method: {self.top_k_method}")
        else:
            keep_mask = torch.ones_like(keep_logits)

        gate = torch.softmax(keep_logits / self.temperature, dim=-1)
        gate = gate * keep_mask
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)

        path_vec = torch.einsum("bs,bsd->bd", gate, slots)

        if token_attn is not None:
            slot_token_attn = token_attn.mean(dim=1)
            slot_counts = slot_token_attn.sum(dim=-1)
            token_attn_entropy = -(
                slot_token_attn * slot_token_attn.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        else:
            slot_counts = gate.new_zeros(gate.shape)
            token_attn_entropy = gate.new_tensor(0.0)

        slot_usage = gate.mean(dim=0)
        avg_tokens_per_slot = slot_counts.mean(dim=0)
        viz = {
            "slot_usage": slot_usage.detach(),
            "slot_token_counts": avg_tokens_per_slot.detach(),
            "avg_tokens_per_slot": avg_tokens_per_slot.detach(),
            "slot_confidence": gate.max(dim=-1).values.mean().detach(),
            "slot_gate": gate.detach(),
            "slot_keep_mask": keep_mask.detach(),
            "token_attn_entropy": token_attn_entropy.detach(),
        }
        return path_vec, viz
