"""Gene-free WSI path compression blocks for CFSurv."""

import math
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
        if self.num_slots > 1 and not self.train_init_noise:
                    raise ValueError(
                        "Vanilla Slot Attention requires independent Gaussian "
                        "initialization when num_slots > 1."
                    )
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
        # Vanilla Slot Attention independently samples every slot from the
        # shared learnable Gaussian during both training and evaluation.
        mu = self.slots_mu.expand(bsz, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(bsz, self.num_slots, -1)
        if self.train_init_noise:
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
    noise = torch.rand(
        t.shape,
        device=t.device,
        dtype=torch.float32,
    )
    return -log(-log(noise))


def relaxed_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    scores = logits.float()
    soft_k_hot = torch.zeros_like(scores)

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
    logits_fp32 = logits.float()
    noised_logits = (
        logits_fp32 + gumbel_noise(logits_fp32)
        if add_noise
        else logits_fp32
    )
    topk_indices = noised_logits.topk(k=k, dim=-1).indices

    hard_k_hot = torch.zeros_like(logits_fp32)
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
    logits_fp32 = logits.float()
    noised_logits = (
        logits_fp32 + gumbel_noise(logits_fp32)
        if add_noise
        else logits_fp32
    )
    topk_indices = noised_logits.topk(k=k, dim=-1).indices

    hard_k_hot = torch.zeros_like(logits_fp32)
    hard_k_hot.scatter_(1, topk_indices, 1.0)

    soft = F.softmax(noised_logits / temperature, dim=-1)
    soft_k_hot = k * soft

    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


class CFSurvPathSlotCompressor(nn.Module):
    """Slot-based, gene-free WSI compressor for CFSurv path_vec."""

    ADAPTIVE_CAPACITY_CANDIDATES = (1, 2, 4, 8)

    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 16,
        num_classes: int = 4,
        iters: int = 3,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.01,
        topk_ratio: float = 0.25,
        top_k_method: str = "parallel_topk_st",
        train_init_noise: bool = True,
        adaptive_capacity: bool = False,
        capacity_temperature: float = 1.0,
    ):
        super().__init__()

        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)
        self.top_k_method = str(top_k_method)
        self.adaptive_capacity = bool(adaptive_capacity)
        self.capacity_temperature = float(capacity_temperature)
        if self.top_k_method not in {"gumbel_topk_st", "parallel_topk_st"}:
            raise ValueError(f"Invalid top_k_method: {self.top_k_method}")
        if self.num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if self.adaptive_capacity and self.num_slots != 8:
            raise ValueError(
                "Adaptive path Slot capacity requires path_slot_num=8 "
                "for candidate capacities K={1,2,4,8}."
            )
        if (
            not math.isfinite(self.capacity_temperature)
            or self.capacity_temperature <= 0
        ):
            raise ValueError(
                "capacity_temperature must be finite and positive."
            )

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

        if self.adaptive_capacity:
            # SlotSPE MoESlotDecoder: predict discrete-time survival logits
            # for every mapped Slot before predictive Top-K activation.
            self.slot_decoder = nn.Linear(self.dim, self.num_classes)
            evidence_dim = 2 * (self.num_classes + 1)
            self.capacity_head = nn.Linear(
                evidence_dim,
                len(self.ADAPTIVE_CAPACITY_CANDIDATES),
            )

    def _stable_masked_slot_gate(
        self,
        keep_logits: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate SlotSPE's masked softmax without low-temperature underflow."""
        if keep_logits.dim() != 2:
            raise ValueError(
                f"Expected keep_logits [B, S], got {tuple(keep_logits.shape)}"
            )
        if keep_mask.dim() not in {2, 3}:
            raise ValueError(
                "Expected keep_mask [B, S] or [B, K, S], "
                f"got {tuple(keep_mask.shape)}"
            )

        scaled_logits = keep_logits.float() / self.temperature
        if keep_mask.dim() == 3:
            scaled_logits = scaled_logits.unsqueeze(1)
        hard_mask = keep_mask.detach() > 0.5
        if not bool(hard_mask.any(dim=-1).all()):
            raise ValueError("Every Slot gate must retain at least one Slot.")

        selected_max = scaled_logits.masked_fill(
            ~hard_mask,
            -torch.inf,
        ).amax(dim=-1, keepdim=True)
        shifted_logits = scaled_logits - selected_max

        # Selected logits are <= 0 after the shift. Clipping only the
        # unselected surrogate values prevents inf * 0 while preserving the
        # exact forward gate and the ST mask gradient path.
        shifted_logits = torch.where(
            hard_mask,
            shifted_logits,
            shifted_logits.clamp(max=0.0),
        )
        weights = torch.exp(shifted_logits) * keep_mask.float()
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        gate = weights / denominator
        return gate.to(dtype=keep_logits.dtype)

    def _adaptive_keep_masks(self, keep_logits: torch.Tensor) -> torch.Tensor:
        """Build nested ST masks for K={1,2,4,8} from one shared ranking."""
        keep_logits_fp32 = keep_logits.float()
        ranked_logits = (
            keep_logits_fp32 + gumbel_noise(keep_logits_fp32)
            if self.training
            else keep_logits_fp32
        )
        ranked_indices = ranked_logits.topk(k=self.num_slots, dim=-1).indices
        masks = []

        for k in self.ADAPTIVE_CAPACITY_CANDIDATES:
            hard_k_hot = torch.zeros_like(keep_logits_fp32)
            hard_k_hot.scatter_(1, ranked_indices[:, :k], 1.0)

            if self.top_k_method == "gumbel_topk_st":
                soft_k_hot = relaxed_topk(
                    ranked_logits,
                    k=k,
                    temperature=self.temperature,
                )
            elif self.top_k_method == "parallel_topk_st":
                soft_k_hot = k * F.softmax(
                    ranked_logits / self.temperature,
                    dim=-1,
                )
            else:
                raise ValueError(f"Invalid top_k_method: {self.top_k_method}")

            masks.append(hard_k_hot + soft_k_hot - soft_k_hot.detach())

        return torch.stack(masks, dim=1)

    def _forward_adaptive_capacity(
        self,
        slots: torch.Tensor,
        keep_logits: torch.Tensor,
    ):
        slot_logits = self.slot_decoder(slots)
        keep_masks = self._adaptive_keep_masks(keep_logits)

        candidate_gates = self._stable_masked_slot_gate(
            keep_logits,
            keep_masks,
        )

        candidate_path_vecs = torch.einsum(
            "bks,bsd->bkd",
            candidate_gates.float(),
            slots.float(),
        )
        candidate_survival_logits = torch.einsum(
            "bks,bsc->bkc",
            candidate_gates.float(),
            slot_logits.float(),
        )

        centered_keep_logits = keep_logits - keep_logits.mean(
            dim=1,
            keepdim=True,
        )
        slot_evidence = torch.cat(
            [slot_logits, centered_keep_logits.unsqueeze(-1)],
            dim=-1,
        )
        patient_evidence = torch.cat(
            [
                slot_evidence.mean(dim=1),
                slot_evidence.max(dim=1).values,
            ],
            dim=-1,
        )
        capacity_logits = self.capacity_head(patient_evidence)
        capacity_logits_fp32 = capacity_logits.float()
        capacity_probs = F.softmax(
            capacity_logits_fp32 / self.capacity_temperature,
            dim=-1,
        )

        if self.training:
            capacity_choice = F.gumbel_softmax(
                capacity_logits_fp32,
                tau=self.capacity_temperature,
                hard=True,
                dim=-1,
            )
        else:
            selected_index = capacity_logits_fp32.argmax(dim=-1)
            capacity_choice = F.one_hot(
                selected_index,
                num_classes=len(self.ADAPTIVE_CAPACITY_CANDIDATES),
            ).to(dtype=capacity_logits_fp32.dtype)

        path_vec = torch.einsum(
            "bk,bkd->bd",
            capacity_choice,
            candidate_path_vecs,
        ).to(dtype=slots.dtype)
        selected_survival_logits = torch.einsum(
            "bk,bkc->bc",
            capacity_choice,
            candidate_survival_logits,
        )
        selected_gate = torch.einsum(
            "bk,bks->bs",
            capacity_choice,
            candidate_gates.float(),
        ).to(dtype=slots.dtype)
        selected_keep_mask = torch.einsum(
            "bk,bks->bs",
            capacity_choice,
            keep_masks,
        )

        candidate_ks = capacity_probs.new_tensor(
            self.ADAPTIVE_CAPACITY_CANDIDATES
        )
        selected_k = torch.einsum("bk,k->b", capacity_choice, candidate_ks)
        expected_capacity = torch.einsum(
            "bk,k->b",
            capacity_probs,
            candidate_ks,
        )
        slot_aux = {
            "survival_logits": selected_survival_logits,
            "capacity_probs": capacity_probs,
            "expected_capacity_ratio": expected_capacity / float(self.num_slots),
            "selected_k": selected_k,
        }
        adaptive_viz = {
            "selected_k": selected_k.detach(),
            "capacity_probs": capacity_probs.detach(),
            "expected_capacity": expected_capacity.detach(),
            "candidate_ks": candidate_ks.detach(),
        }
        return (
            path_vec,
            selected_gate,
            selected_keep_mask,
            slot_aux,
            adaptive_viz,
        )

    def forward(self, z_tokens: torch.Tensor,return_token_attention: bool = False):
        if z_tokens.dim() != 3:
            raise ValueError(f"Expected z_tokens [B, N, D], got {tuple(z_tokens.shape)}")

        _, _, dim = z_tokens.shape
        if dim != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {dim}")

        slots, token_attn = self.slot_attention(z_tokens, return_attn=True)
        slots = self.slot_map(slots)

        keep_logits = self.slot_score(slots).squeeze(-1)
        slot_aux = None
        adaptive_viz = {}
        if self.adaptive_capacity:
            (
                path_vec,
                gate,
                keep_mask,
                slot_aux,
                adaptive_viz,
            ) = self._forward_adaptive_capacity(slots, keep_logits)
        else:
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

            gate = self._stable_masked_slot_gate(
                keep_logits,
                keep_mask,
            )

            path_vec = torch.einsum("bs,bsd->bd", gate, slots)

        slot_token_attn = None
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
        viz.update(adaptive_viz)
        if return_token_attention and slot_token_attn is not None:
            token_importance = torch.einsum(
                "bs,bsn->bn",
                gate,
                slot_token_attn,
            )

            token_importance = token_importance.clamp_min(0.0)
            token_importance = token_importance / token_importance.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)

            viz["slot_token_attn"] = slot_token_attn.detach()
            viz["token_importance"] = token_importance.detach()
        return path_vec, viz, slot_aux
