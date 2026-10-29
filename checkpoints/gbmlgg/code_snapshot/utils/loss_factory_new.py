import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import as_bool


def nll_loss(hazards, S, Y, c, alpha=0.0, eps=1e-7):
    """
    Discrete-time survival negative log-likelihood.

    hazards: [B, K]
    S: [B, K], cumulative product of (1 - hazards)
    Y: [B], discrete time-bin index in [0, K - 1]
    c: [B], censorship indicator, where 1 means censored and 0 means observed event
    """
    B = Y.numel()
    Y = Y.view(B, 1).long()
    c = c.view(B, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], dim=1)

    uncensored = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp_min(eps)) +
        torch.log(torch.gather(hazards, 1, Y).clamp_min(eps))
    )
    censored = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp_min(eps))

    neg_l = censored + uncensored
    loss = (1 - alpha) * neg_l + alpha * uncensored
    return loss.mean()


class NLLSurvLoss(object):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, out, gt):
        total = 0.0
        for hazards, survival in zip(out["hazards"], out["S"]):
            total += nll_loss(hazards, survival, gt["label"], gt["c"], alpha=self.alpha)
        return total


class TriFactorLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lambda_site_bg = float(getattr(args, "lambda_site_bg", 0.1))
        self.lambda_site_adv = float(getattr(args, "lambda_site_adv", 0.05))
        self.lambda_orth = float(getattr(args, "lambda_orth", 0.01))
        self.lambda_rec = float(getattr(args, "lambda_rec", 0.001))
        self.lambda_proto = float(getattr(args, "lambda_proto", 0.01))
        self.lambda_proto_usage = float(getattr(args, "lambda_proto_usage", 0.01))
        self.proto_usage_queue_size = int(getattr(args, "proto_usage_queue_size", 256))
        self.site_loss_warmup_epochs = int(getattr(args, "site_loss_warmup_epochs", 5))
        self.enable_rec = as_bool(getattr(args, "enable_rec", True), True)
        self.epoch = 0
        self.phase = "train"
        self.proto_usage_queue = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_phase(self, phase):
        self.phase = str(phase).lower()

    def get_runtime_state(self):
        queue = self.proto_usage_queue
        return {
            "epoch": int(self.epoch),
            "phase": self.phase,
            "proto_usage_queue": None if queue is None else queue.detach().cpu(),
        }

    def load_runtime_state(self, state):
        if not state:
            return
        self.epoch = int(state.get("epoch", self.epoch))
        self.phase = str(state.get("phase", self.phase))
        queue = state.get("proto_usage_queue", None)
        self.proto_usage_queue = None if queue is None else queue.detach()

    def _site_scale(self):
        if self.site_loss_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.epoch + 1) / float(self.site_loss_warmup_epochs))

    @staticmethod
    def _zero_like(preds):
        for value in preds.values():
            if torch.is_tensor(value):
                return torch.tensor(0.0, dtype=value.dtype, device=value.device)
        return torch.tensor(0.0)

    @staticmethod
    def _cosine_square(a, b):
        return F.cosine_similarity(a, b, dim=-1, eps=1e-8).pow(2).mean()

    def forward(self, preds, target):
        total = self._zero_like(preds)
        logs = {}

        if preds.get("tri_enabled", True) is False:
            logs["tri_disabled"] = torch.tensor(1.0, device=total.device)
            return total, logs

        site = target.get("site", None)
        site_scale = self._site_scale()
        logs["site_scale"] = torch.tensor(site_scale, device=total.device)
        if site is not None:
            site = site.view(-1).long().to(total.device)
            if "site_bg_logits" in preds and self.lambda_site_bg > 0:
                loss_site_bg = F.cross_entropy(preds["site_bg_logits"], site)
                total = total + site_scale * self.lambda_site_bg * loss_site_bg
                logs["site_bg"] = loss_site_bg.detach()
            if "site_adv_logits" in preds and self.lambda_site_adv > 0:
                loss_site_adv = F.cross_entropy(preds["site_adv_logits"], site)
                total = total + site_scale * self.lambda_site_adv * loss_site_adv
                logs["site_adv"] = loss_site_adv.detach()
        else:
            logs["site_missing"] = torch.tensor(1.0, device=total.device)

        orth_terms = []
        if "z_b" in preds and "z_u" in preds:
            orth_terms.append(self._cosine_square(preds["z_b"], preds["z_u"]))
        if "z_b" in preds and "z_s" in preds:
            orth_terms.append(0.5 * self._cosine_square(preds["z_b"], preds["z_s"]))
        if "z_s" in preds and "z_u" in preds:
            orth_terms.append(0.1 * self._cosine_square(preds["z_s"], preds["z_u"]))
        if orth_terms:
            loss_orth = torch.stack(orth_terms).sum()
            total = total + self.lambda_orth * loss_orth
            logs["orth"] = loss_orth.detach()

        if self.enable_rec and "h_rec" in preds and "h" in preds and self.lambda_rec > 0:
            loss_rec = F.mse_loss(preds["h_rec"], preds["h"].detach())
            total = total + self.lambda_rec * loss_rec
            logs["rec"] = loss_rec.detach()

        if "prog_prototypes" in preds and self.lambda_proto > 0:
            prototypes = F.normalize(preds["prog_prototypes"], dim=-1)
            identity = torch.eye(
                prototypes.size(0),
                device=prototypes.device,
                dtype=prototypes.dtype,
            )
            loss_proto = ((prototypes @ prototypes.t() - identity) ** 2).mean()
            total = total + self.lambda_proto * loss_proto
            logs["proto_div"] = loss_proto.detach()

        if "proto_assign" in preds:
            assign = preds["proto_assign"]
            if self.lambda_proto_usage > 0 and self.proto_usage_queue_size > 0:
                queue = self.proto_usage_queue
                if queue is not None:
                    queue = queue.to(device=assign.device, dtype=assign.dtype)
                    usage_source = torch.cat([queue, assign], dim=0)
                else:
                    usage_source = assign
                mean_usage = usage_source.mean(dim=0)
                uniform = torch.full_like(mean_usage, 1.0 / float(mean_usage.numel()))
                loss_proto_usage = F.mse_loss(mean_usage, uniform)
                total = total + self.lambda_proto_usage * loss_proto_usage
                logs["proto_usage"] = loss_proto_usage.detach()

            with torch.no_grad():
                logs["proto_max_prob"] = assign.max(dim=-1).values.mean().detach()
                sample_entropy = -(assign * assign.clamp_min(1e-8).log()).sum(dim=-1).mean()
                logs["proto_entropy"] = sample_entropy.detach()
                current_usage = assign.mean(dim=0)
                logs["proto_usage_min"] = current_usage.min().detach()
                logs["proto_usage_max"] = current_usage.max().detach()
                logs["proto_usage_entropy"] = (
                    -(current_usage * current_usage.clamp_min(1e-8).log()).sum()
                ).detach()

                if self.phase == "train" and self.proto_usage_queue_size > 0:
                    new_queue = assign.detach()
                    if self.proto_usage_queue is not None:
                        old_queue = self.proto_usage_queue.to(device=assign.device, dtype=assign.dtype)
                        new_queue = torch.cat([old_queue, new_queue], dim=0)
                    if new_queue.size(0) > self.proto_usage_queue_size:
                        new_queue = new_queue[-self.proto_usage_queue_size:]
                    self.proto_usage_queue = new_queue.detach()

        return total, logs


class Loss_factory(nn.Module):
    """Build the active training loss for CFSurv."""

    def __init__(self, args):
        super().__init__()
        self.phase = None
        self.lambda_tri = float(getattr(args, "lambda_tri", 1.0))
        self.lambda_path_slot_capacity = float(
            getattr(args, "lambda_path_slot_capacity", 0.01)
        )
        self.loss_fns = {
            "nllsurv": NLLSurvLoss(
                alpha=float(getattr(args, "nll_alpha", 0.0))
            ),
            "tri": TriFactorLoss(args),
        }
        self.loss_collection = []
        for loss_item in str(args.loss).replace("+", ",").split(","):
            loss_item = loss_item.strip().lower()
            if not loss_item:
                continue

            if loss_item not in self.loss_fns:
                valid = "nllsurv or nllsurv,tri"
                raise ValueError(f"Unsupported loss '{loss_item}'. CFSurv supports: {valid}.")
            if loss_item not in self.loss_collection:
                self.loss_collection.append(loss_item)

        if not self.loss_collection:
            raise ValueError("At least one loss must be specified. CFSurv supports: nllsurv or nllsurv,tri.")

        active = set(self.loss_collection)
        if active not in ({"nllsurv"}, {"nllsurv", "tri"}):
            raise ValueError("CFSurv supports only 'nllsurv' or 'nllsurv,tri'.")

    def set_epoch(self, epoch):
        for loss_fn in self.loss_fns.values():
            if hasattr(loss_fn, "set_epoch"):
                loss_fn.set_epoch(epoch)

    def set_phase(self, phase):
        phase = str(phase).lower()
        if phase not in {"train", "eval"}:
            raise ValueError("Loss phase must be 'train' or 'eval'.")
        self.phase = phase
        for loss_fn in self.loss_fns.values():
            if hasattr(loss_fn, "set_phase"):
                loss_fn.set_phase(phase)

    def _resolve_phase(self, target):
        target_phase = target.get("phase", None)
        if target_phase is not None:
            target_phase = str(target_phase).lower()
            if target_phase not in {"train", "eval"}:
                raise ValueError(
                    "target['phase'] must be 'train' or 'eval'."
                )
        if (
            self.phase is not None
            and target_phase is not None
            and self.phase != target_phase
        ):
            raise ValueError(
                "Loss phase mismatch: "
                f"criterion={self.phase!r}, target={target_phase!r}."
            )
        return self.phase or target_phase or "train"

    def get_runtime_state(self):
        return {
            name: loss_fn.get_runtime_state()
            for name, loss_fn in self.loss_fns.items()
            if hasattr(loss_fn, "get_runtime_state")
        }

    def load_runtime_state(self, state):
        if not state:
            return
        for name, loss_state in state.items():
            loss_fn = self.loss_fns.get(name)
            if loss_fn is not None and hasattr(loss_fn, "load_runtime_state"):
                loss_fn.load_runtime_state(loss_state)

    def forward(self, preds, target):
        total = torch.tensor(0.0, device=target["label"].device)
        loss_dict = {}
        for name in self.loss_collection:
            val = self.loss_fns[name](preds, target)
            component_logs = {}
            if isinstance(val, tuple):
                val, component_logs = val
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32, device=total.device)
            loss_dict[name] = val.detach()
            for key, component in component_logs.items():
                if not isinstance(component, torch.Tensor):
                    component = torch.tensor(component, dtype=torch.float32, device=total.device)
                loss_dict[key] = component.detach()
            weight = self.lambda_tri if name == "tri" else 1.0
            weighted_val = val * weight
            loss_dict[f"{name}_weighted"] = weighted_val.detach()
            total = total + weighted_val

        slot_aux = preds.get("slot_aux")
        phase = self._resolve_phase(target)
        if slot_aux is not None and phase == "train":
            slot_logits = slot_aux["survival_logits"]
            slot_hazards = torch.sigmoid(slot_logits)
            slot_survival = torch.cumprod(1 - slot_hazards, dim=1)
            slot_aux_nll = nll_loss(
                slot_hazards,
                slot_survival,
                target["label"],
                target["c"],
                alpha=0.5,
            )

            expected_capacity_ratio = slot_aux[
                "expected_capacity_ratio"
            ].mean()
            capacity_penalty = (
                self.lambda_path_slot_capacity * expected_capacity_ratio
            )

            total = total + slot_aux_nll + capacity_penalty
            loss_dict["slot_aux_nll"] = slot_aux_nll.detach()
            loss_dict["slot_capacity_ratio"] = (
                expected_capacity_ratio.detach()
            )
            loss_dict["slot_capacity_penalty"] = capacity_penalty.detach()

        return total, loss_dict
