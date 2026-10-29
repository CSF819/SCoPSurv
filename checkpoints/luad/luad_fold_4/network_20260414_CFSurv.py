"""Standalone factual multimodal survival model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .path_compressor import CFSurvPathSlotCompressor
except ImportError:
    from models.CFSurv.path_compressor import CFSurvPathSlotCompressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFSurv_DEFAULT_CONFIG = {
    "max_tokens": 4096,
    "min_tokens": 50,
    "adaptive_sampling": False,
    "path_slot_num": 8,
    "path_slot_iters": 10,
    "path_slot_heads": 8,
    "path_slot_dim_head": 64,
    "path_slot_dropout": 0.0,
    "path_slot_temperature": 0.01,
    "path_slot_topk_ratio": 0.25,
    "path_slot_top_k_method": "parallel_topk_st",
    "path_slot_train_init_noise": True,
    "token_keep_ratio": 0.8,
    "tri_dim": 256,
    "prog_proto_num": 4,
    "proto_temperature": 0.2,
    "risk_conditioning": "film",
    "grl_lambda": 1.0,
    "lambda_site_bg": 0.1,
    "lambda_site_adv": 0.05,
    "lambda_orth": 0.01,
    "lambda_rec": 0.001,
    "lambda_proto": 0.01,
    "enable_rec": True,
    "enable_tri_factor": True,
    "tri_use_zb": True,
    "tri_use_zs": True,
    "tri_use_zu": True,
    "site_bg_stop_grad": False,
    "film_zero_init": True,
    "site_loss_warmup_epochs": 5,
    "lambda_proto_usage": 0.01,
    "proto_usage_queue_size": 256,
    "export_tri_embeddings": True,
    "run_tri_probe": True,
    "enable_bg_calibration": False,
    "bg_calibration_weight": 1.0,
    "bg_calibration_stop_grad": True,
    "bg_calibration_zero_init": True,
}

DATASET_SPECIFIC = {}

genomics_idx = {
    "blca": [0, 94, 428, 949, 1417, 2913, 3392],
    "brca": [0, 91, 444, 997, 1477, 3043, 3523],
    "gbmlgg": [0, 84, 398, 896, 1311, 2707, 3135],
    "luad": [0, 89, 423, 957, 1428, 2938, 3420],
    "ucec": [0, 3, 27, 48, 70, 135, 150],
}


def get_auto_config(dataset_name=None):
    """Return parser defaults for the factual CFSurv variant."""
    base_config = {
        "modal": "multi",
        "loss": "nllsurv,tri",
        "sets": "blca,brca,luad,ucec,gbmlgg",
        "seed": 0,
    }

    customized_config = {
        key: {"type": type(value), "default": value}
        for key, value in CFSurv_DEFAULT_CONFIG.items()
    }

    if dataset_name:
        dataset_name = dataset_name.lower()
        if dataset_name in DATASET_SPECIFIC:
            specific_params = DATASET_SPECIFIC[dataset_name]
            for key in ["lr", "optimizer", "num_epoch", "weight_decay"]:
                if key in specific_params:
                    base_config[key] = specific_params[key]
            for key, value in specific_params.items():
                if key in customized_config:
                    customized_config[key]["default"] = value
            print(f"Auto-configured factual CFSurv with overrides for {dataset_name.upper()}")
        else:
            print(f"Auto-configured factual CFSurv defaults for {dataset_name.upper()}")

        print(f"  - Path slots: {customized_config['path_slot_num']['default']}")
        print(f"  - Slot top-k ratio: {customized_config['path_slot_topk_ratio']['default']}")
        print(f"  - Slot dropout: {customized_config['path_slot_dropout']['default']}")

    return {
        "base": base_config,
        "customized": customized_config,
    }


custom_config = get_auto_config()


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def SNN_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.SELU(),
        nn.AlphaDropout(p=dropout, inplace=False),
    )


def conv1d_Block(dim1, dim2):
    return nn.Sequential(
        nn.Conv1d(dim1, dim2, 1),
        nn.InstanceNorm1d(dim2),
        nn.ReLU(),
    )


class PairFusionBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, g, p):
        x = torch.cat([g, p, g * p, torch.abs(g - p)], dim=-1)
        return self.net(x)


class FactorMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CFSurv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seed = args.seed
        self.n_classes = args.n_classes
        self.feat_dim = 256
        self.max_tokens = int(getattr(args, "max_tokens", 4096))
        self.min_tokens = int(getattr(args, "min_tokens", 50))
        self.adaptive_sampling = bool(getattr(args, "adaptive_sampling", True))
        self.token_keep_ratio = float(getattr(args, "token_keep_ratio", 1.0))

        if args.extractor in ("uni", "resnet50"):
            path_dim = 1024
        elif args.extractor in ("plip", "conch", "feather"):
            path_dim = 512
        elif args.extractor in ("hipt",):
            path_dim = 384
        elif args.extractor in ("titan", "conch_v15"):
            path_dim = 768
        else:
            path_dim = 512

        self.C_sp = path_dim
        self.path_token_proj = SNN_Block(self.C_sp, self.feat_dim)
        self.path_slot_num = int(getattr(args, "path_slot_num", 8))
        self.path_compressor = CFSurvPathSlotCompressor(
            dim=self.feat_dim,
            num_slots=self.path_slot_num,
            iters=int(getattr(args, "path_slot_iters", 10)),
            heads=int(getattr(args, "path_slot_heads", 8)),
            dim_head=int(getattr(args, "path_slot_dim_head", 64)),
            dropout=float(getattr(args, "path_slot_dropout", 0.0)),
            temperature=float(getattr(args, "path_slot_temperature", 0.01)),
            topk_ratio=float(getattr(args, "path_slot_topk_ratio", 0.25)),
            top_k_method=str(getattr(args, "path_slot_top_k_method", "parallel_topk_st")),
            train_init_noise=_as_bool(getattr(args, "path_slot_train_init_noise", True)),
        )

        self.genomics_idx = genomics_idx[args.dataset]
        sig_networks = []
        for idx in range(len(self.genomics_idx) - 1):
            d = self.genomics_idx[idx + 1] - self.genomics_idx[idx]
            sig_networks.append(nn.Sequential(SNN_Block(d, 256)))
        self.genomics_fc = nn.ModuleList(sig_networks)
        self.geno_fc = SNN_Block(256, self.feat_dim)
        self.geno_conv = conv1d_Block(6, 1)

        self.fusion_backbone = PairFusionBackbone(self.feat_dim)
        self.baseline_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.n_classes),
        )
        self.enable_tri_factor = _as_bool(getattr(args, "enable_tri_factor", True), True)
        self.tri_use_zb = _as_bool(getattr(args, "tri_use_zb", True), True)
        self.tri_use_zs = _as_bool(getattr(args, "tri_use_zs", True), True)
        self.tri_use_zu = _as_bool(getattr(args, "tri_use_zu", True), True)
        self.site_bg_stop_grad = _as_bool(getattr(args, "site_bg_stop_grad", True), True)
        self.film_zero_init = _as_bool(getattr(args, "film_zero_init", True), True)
        self.enable_bg_calibration = _as_bool(getattr(args, "enable_bg_calibration", False), False)
        self.bg_calibration_weight = float(getattr(args, "bg_calibration_weight", 1.0))
        self.bg_calibration_stop_grad = _as_bool(getattr(args, "bg_calibration_stop_grad", True), True)
        self.bg_calibration_zero_init = _as_bool(getattr(args, "bg_calibration_zero_init", True), True)
        if self.enable_tri_factor and not (self.tri_use_zs or self.tri_use_zu):
            raise ValueError("At least one of tri_use_zs or tri_use_zu must be true when enable_tri_factor is true.")
        if self.enable_bg_calibration and (not self.enable_tri_factor or not self.tri_use_zb):
            raise ValueError("enable_bg_calibration requires enable_tri_factor=true and tri_use_zb=true.")

        self.tri_dim = int(getattr(args, "tri_dim", self.feat_dim))
        self.prog_proto_num = int(getattr(args, "prog_proto_num", 4))
        self.proto_temperature = float(getattr(args, "proto_temperature", 0.2))
        self.risk_conditioning = str(getattr(args, "risk_conditioning", "film")).lower()
        if self.risk_conditioning not in {"film", "additive", "moe"}:
            raise ValueError(
                "risk_conditioning must be one of {'film', 'additive', 'moe'}, "
                f"got '{self.risk_conditioning}'."
            )
        if self.prog_proto_num < 1:
            raise ValueError("prog_proto_num must be >= 1.")

        self.num_sites = max(1, int(getattr(args, "num_sites", 1)))
        self.grl_lambda = float(getattr(args, "grl_lambda", 1.0))

        self.E_b = FactorMLP(self.feat_dim, self.tri_dim)
        self.E_s = FactorMLP(self.feat_dim, self.tri_dim)
        self.E_u = FactorMLP(self.feat_dim, self.tri_dim)

        self.prog_prototypes = nn.Parameter(torch.empty(self.prog_proto_num, self.tri_dim))
        nn.init.xavier_uniform_(self.prog_prototypes)

        self.film = nn.Sequential(
            nn.Linear(self.prog_proto_num, self.tri_dim * 2),
            nn.GELU(),
            nn.Linear(self.tri_dim * 2, self.tri_dim * 2),
        )
        if self.film_zero_init:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        self.risk_head = nn.Sequential(
            nn.Linear(self.tri_dim, self.tri_dim),
            nn.LayerNorm(self.tri_dim),
            nn.GELU(),
            nn.Linear(self.tri_dim, self.n_classes),
        )
        self.subcohort_bias = nn.Linear(self.prog_proto_num, self.n_classes)
        self.risk_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.tri_dim, self.tri_dim),
                nn.LayerNorm(self.tri_dim),
                nn.GELU(),
                nn.Linear(self.tri_dim, self.n_classes),
            )
            for _ in range(self.prog_proto_num)
        ])

        self.site_bg_head = nn.Sequential(
            nn.Linear(self.tri_dim, self.tri_dim),
            nn.LayerNorm(self.tri_dim),
            nn.GELU(),
            nn.Linear(self.tri_dim, self.num_sites),
        )
        self.site_adv_head = nn.Sequential(
            nn.Linear(self.tri_dim, self.tri_dim),
            nn.LayerNorm(self.tri_dim),
            nn.GELU(),
            nn.Linear(self.tri_dim, self.num_sites),
        )
        self.bg_calibration_head = nn.Sequential(
            nn.Linear(self.tri_dim, self.tri_dim),
            nn.LayerNorm(self.tri_dim),
            nn.GELU(),
            nn.Linear(self.tri_dim, self.n_classes),
        )
        if self.bg_calibration_zero_init:
            nn.init.zeros_(self.bg_calibration_head[-1].weight)
            nn.init.zeros_(self.bg_calibration_head[-1].bias)

        self.reconstructor = nn.Sequential(
            nn.Linear(self.tri_dim * 3, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def _encode_gene(self, x_omic):
        geno_feat = torch.stack([
            self.genomics_fc[idx](x_omic[..., length:self.genomics_idx[idx + 1]])
            for idx, length in enumerate(self.genomics_idx[:-1])
        ], dim=1)
        return self.geno_conv(self.geno_fc(geno_feat)).squeeze(1)

    def _adaptive_token_sampling(self, x_path):
        B, N, C = x_path.shape
        sampled = False
        if N == 0:
            x_path = torch.randn(B, self.min_tokens, C, device=x_path.device) * 1e-4
            return x_path, self.min_tokens, True
        if not self.adaptive_sampling:
            return x_path, N, sampled
        if N > self.max_tokens:
            idx = torch.randperm(N, device=x_path.device)[:self.max_tokens]
            x_path = x_path[:, idx, :]
            sampled, N = True, self.max_tokens
        elif N < self.min_tokens:
            repeat_times = math.ceil(self.min_tokens / N)
            x_rep = x_path.repeat(1, repeat_times, 1)[:, :self.min_tokens, :]
            noise = torch.randn_like(x_rep) * 1e-4
            x_path = x_rep + noise
            sampled, N = True, self.min_tokens
        return x_path, N, sampled

    def _encode_path(self, x_path):
        x_path, _, _ = self._adaptive_token_sampling(x_path)
        if self.training and self.token_keep_ratio < 1.0:
            N_ = x_path.shape[1]
            keep = max(int(N_ * self.token_keep_ratio), self.min_tokens)
            if keep < N_:
                idx = torch.randperm(N_, device=x_path.device)[:keep]
                x_path = x_path[:, idx, :]
        z_tokens = self.path_token_proj(x_path)
        path_vec, viz = self.path_compressor(z_tokens)
        return path_vec, viz

    def _prototype_assign(self, z_s):
        z = F.normalize(z_s, dim=-1)
        p = F.normalize(self.prog_prototypes, dim=-1)
        logits = torch.matmul(z, p.t()) / max(self.proto_temperature, 1e-6)
        assign = torch.softmax(logits, dim=-1)
        return assign, logits

    def _risk_from_factors(self, z_u, proto_assign):
        if self.risk_conditioning == "film":
            gamma_beta = self.film(proto_assign)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            z_u_cond = (1.0 + gamma) * z_u + beta
            return self.risk_head(z_u_cond), z_u_cond

        if self.risk_conditioning == "additive":
            z_u_cond = z_u
            return self.risk_head(z_u) + self.subcohort_bias(proto_assign), z_u_cond

        if self.risk_conditioning == "moe":
            z_u_cond = z_u
            expert_logits = torch.stack([expert(z_u) for expert in self.risk_experts], dim=1)
            return torch.sum(proto_assign.unsqueeze(-1) * expert_logits, dim=1), z_u_cond

        raise RuntimeError(f"Unhandled risk_conditioning: {self.risk_conditioning}")

    def _uniform_proto_assign(self, batch_size, device, dtype):
        return torch.full(
            (batch_size, self.prog_proto_num),
            1.0 / float(self.prog_proto_num),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _norm_or_zero(value, reference):
        if value is None:
            return torch.tensor(0.0, device=reference.device).cpu()
        return value.detach().norm(dim=-1).mean().cpu()

    def forward(
        self,
        x_path,
        x_omic,
        phase,
        label=None,
        c=None,
        patient_uid=None,
        **kwargs,
    ):
        gene_vec = self._encode_gene(x_omic)
        path_vec, viz_full = self._encode_path(x_path)
        h = self.fusion_backbone(gene_vec, path_vec)

        if not self.enable_tri_factor:
            final_logits = self.baseline_head(h)
            hazards = torch.sigmoid(final_logits).clamp(1e-6, 1 - 1e-6)
            S = torch.cumprod(1 - hazards, dim=1)
            return {
                "tri_enabled": False,
                "bg_calibration_enabled": False,
                "encoded_gene": gene_vec.detach(),
                "encoded_path": path_vec.detach(),
                "h": h,
                "hazards": [hazards],
                "S": [S],
                "viz": {
                    "path_compressor": {
                        "slot_usage": viz_full["slot_usage"].detach().cpu(),
                        "slot_token_counts": viz_full["slot_token_counts"].detach().cpu(),
                        "avg_tokens_per_slot": viz_full["avg_tokens_per_slot"].detach().cpu(),
                        "slot_confidence": viz_full["slot_confidence"],
                        "slot_gate": viz_full["slot_gate"].detach().cpu(),
                        "slot_keep_mask": viz_full["slot_keep_mask"].detach().cpu(),
                        "token_attn_entropy": viz_full["token_attn_entropy"],
                    }
                },
            }

        z_b = self.E_b(h) if self.tri_use_zb else None
        z_s = self.E_s(h) if self.tri_use_zs else None
        z_u = self.E_u(h) if self.tri_use_zu else None

        if z_s is not None:
            proto_assign, proto_logits = self._prototype_assign(z_s)
        else:
            reference = z_u if z_u is not None else h
            proto_assign = self._uniform_proto_assign(reference.size(0), reference.device, reference.dtype)
            proto_logits = torch.zeros_like(proto_assign)

        risk_base = z_u if z_u is not None else z_s
        final_logits, z_u_cond = self._risk_from_factors(risk_base, proto_assign)
        bg_calibration_logits = None
        if self.enable_bg_calibration:
            z_b_cal = z_b.detach() if self.bg_calibration_stop_grad else z_b
            bg_calibration_logits = self.bg_calibration_head(z_b_cal)
            final_logits = final_logits + self.bg_calibration_weight * bg_calibration_logits

        hazards = torch.sigmoid(final_logits).clamp(1e-6, 1 - 1e-6)
        S = torch.cumprod(1 - hazards, dim=1)

        site_bg_logits = None
        if self.tri_use_zb:
            z_b_site = self.E_b(h.detach()) if self.site_bg_stop_grad else z_b
            site_bg_logits = self.site_bg_head(z_b_site)

        site_adv_logits = None
        if self.tri_use_zu:
            site_adv_logits = self.site_adv_head(grad_reverse(z_u, self.grl_lambda))

        h_rec = None
        if self.tri_use_zb and self.tri_use_zs and self.tri_use_zu:
            h_rec = self.reconstructor(torch.cat([z_b, z_s, z_u], dim=-1))

        output = {
            "tri_enabled": True,
            "bg_calibration_enabled": bool(self.enable_bg_calibration),
            "tri_active_factors": {
                "z_b": bool(self.tri_use_zb),
                "z_s": bool(self.tri_use_zs),
                "z_u": bool(self.tri_use_zu),
            },
            "encoded_gene": gene_vec.detach(),
            "encoded_path": path_vec.detach(),
            "h": h,
            "z_u_cond": z_u_cond,
            "proto_assign": proto_assign,
            "proto_logits": proto_logits,
            "hazards": [hazards],
            "S": [S],
            "viz": {
                "path_compressor": {
                    "slot_usage": viz_full["slot_usage"].detach().cpu(),
                    "slot_token_counts": viz_full["slot_token_counts"].detach().cpu(),
                    "avg_tokens_per_slot": viz_full["avg_tokens_per_slot"].detach().cpu(),
                    "slot_confidence": viz_full["slot_confidence"],
                    "slot_gate": viz_full["slot_gate"].detach().cpu(),
                    "slot_keep_mask": viz_full["slot_keep_mask"].detach().cpu(),
                    "token_attn_entropy": viz_full["token_attn_entropy"],
                },
                "tri_factor": {
                    "proto_assign": proto_assign.detach().cpu(),
                    "z_b_norm": self._norm_or_zero(z_b, h),
                    "z_s_norm": self._norm_or_zero(z_s, h),
                    "z_u_norm": self._norm_or_zero(z_u, h),
                }
            },
        }
        if h_rec is not None:
            output["h_rec"] = h_rec
        if z_b is not None:
            output["z_b"] = z_b
        if z_s is not None:
            output["z_s"] = z_s
            output["prog_prototypes"] = self.prog_prototypes
        if z_u is not None:
            output["z_u"] = z_u
        if site_bg_logits is not None:
            output["site_bg_logits"] = site_bg_logits
        if site_adv_logits is not None:
            output["site_adv_logits"] = site_adv_logits
        if bg_calibration_logits is not None:
            output["bg_calibration_logits"] = bg_calibration_logits
        return output
