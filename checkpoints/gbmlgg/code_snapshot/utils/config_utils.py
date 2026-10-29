"""Shared, side-effect-free configuration helpers for CFSurv."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


EXTRACTOR_FEATURE_DIMS = {
    "uni": 1024,
    "resnet50": 1024,
    "plip": 512,
    "conch": 512,
    "feather": 512,
    "hipt": 384,
    "titan": 768,
    "conch_v15": 768,
}

EXPECTED_OMIC_DIMS = {
    "blca": 3392,
    "brca": 3523,
    "gbmlgg": 3135,
    "luad": 3420,
    "ucec": 150,
}

DEFAULT_DATA_ROOT_CANDIDATES = (
    Path(r"E:\Resnet50_Features_cyp"),
    Path("/home/user/Chensf/Dataset"),
)


def path_feature_dim(extractor: str, default: int = 512) -> int:
    """Return the expected patch-feature dimension for an extractor."""
    return int(EXTRACTOR_FEATURE_DIMS.get(str(extractor).strip().lower(), default))


def expected_omic_dim(dataset: str) -> int:
    """Return the signature-selected omic dimension for a supported dataset."""
    key = str(dataset).strip().lower()
    if key not in EXPECTED_OMIC_DIMS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of {sorted(EXPECTED_OMIC_DIMS)}."
        )
    return int(EXPECTED_OMIC_DIMS[key])


def resolve_data_root_dir(configured: Any = None) -> str:
    """Resolve an explicit data root or the first existing supported default."""
    if configured is not None and str(configured).strip():
        path = Path(str(configured).strip()).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Configured data root does not exist: {path}")
        return str(path)

    for candidate in DEFAULT_DATA_ROOT_CANDIDATES:
        if candidate.is_dir():
            return str(candidate)

    choices = ", ".join(str(path) for path in DEFAULT_DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "No default WSI feature root exists. Checked: "
        f"{choices}. Pass --data_root_dir explicitly if the data is elsewhere."
    )


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def argparse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def strip_slide_suffix(value: Any) -> str:
    """Remove one known slide suffix with exact suffix semantics."""
    name = str(value)
    lower = name.lower()
    for suffix in (".svs", ".tif", ".tiff", ".pt", ".pth"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def pure_config_dict(value: Any) -> Any:
    """Convert argparse/config values to JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): pure_config_dict(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return pure_config_dict(vars(value))
    if isinstance(value, (list, tuple)):
        return [pure_config_dict(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np
        import torch

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return value


def validate_runtime_args(args: Any) -> None:
    """Fail early for incompatible runtime options while preserving the model flow."""
    errors = []
    modal = str(getattr(args, "modal", "multi")).lower()
    if modal not in {"multi", "path", "omic"}:
        errors.append("modal must be one of: multi, path, omic")

    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size != 1:
        errors.append(
            "batch_size must be 1 because CFSurv uses variable-length WSI bags and the "
            "default collator cannot stack bags with different token counts"
        )

    min_tokens = int(getattr(args, "min_tokens", 0) or 0)
    max_tokens = int(getattr(args, "max_tokens", 0) or 0)
    adaptive = as_bool(getattr(args, "adaptive_sampling", False), False)
    if min_tokens < 0 or max_tokens < 0:
        errors.append("min_tokens and max_tokens must be non-negative")
    if adaptive and max_tokens <= 0:
        errors.append("adaptive_sampling requires max_tokens > 0")
    if adaptive and min_tokens > max_tokens:
        errors.append("min_tokens cannot exceed max_tokens when adaptive_sampling is enabled")
    path_slot_num = int(getattr(args, "path_slot_num", 1))
    if path_slot_num < 1:
        errors.append("path_slot_num must be >= 1")
    adaptive_slot_capacity = as_bool(
        getattr(args, "path_slot_adaptive_capacity", True),
        True,
    )
    if adaptive_slot_capacity and path_slot_num != 8:
        errors.append(
            "path_slot_adaptive_capacity requires path_slot_num=8"
        )
    capacity_temperature = float(
        getattr(args, "path_slot_capacity_temperature", 1.0)
    )
    if not math.isfinite(capacity_temperature) or capacity_temperature <= 0:
        errors.append("path_slot_capacity_temperature must be finite and > 0")
    capacity_lambda = float(
        getattr(args, "lambda_path_slot_capacity", 0.01)
    )
    if not math.isfinite(capacity_lambda) or capacity_lambda < 0:
        errors.append("lambda_path_slot_capacity must be finite and >= 0")
    topk_ratio = float(getattr(args, "path_slot_topk_ratio", 0.25))
    if not 0 < topk_ratio <= 1:
        errors.append("path_slot_topk_ratio must be in (0, 1]")
    if str(getattr(args, "risk_conditioning", "film")).lower() not in {
        "film",
        "additive",
        "moe",
    }:
        errors.append("risk_conditioning must be film, additive, or moe")
    active_losses = {
        token.strip().lower()
        for token in str(getattr(args, "loss", "")).replace("+", ",").split(",")
        if token.strip()
    }
    if active_losses not in ({"nllsurv"}, {"nllsurv", "tri"}):
        errors.append("loss must be nllsurv or nllsurv,tri")

    lambda_tri = float(getattr(args, "lambda_tri", 1.0))
    if not math.isfinite(lambda_tri) or lambda_tri < 0:
        errors.append("lambda_tri must be finite and >= 0")
    nll_alpha = float(getattr(args, "nll_alpha", 0.0))
    if not math.isfinite(nll_alpha) or not 0 <= nll_alpha < 1:
        errors.append("nll_alpha must be finite and in [0, 1)")
    gradient_accumulation_steps = int(
        getattr(args, "gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps < 1:
        errors.append("gradient_accumulation_steps must be >= 1")
    grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
    if not math.isfinite(grad_clip_norm) or grad_clip_norm < 0:
        errors.append("grad_clip_norm must be finite and >= 0")
    optimizer_warmup_epochs = int(
        getattr(args, "optimizer_warmup_epochs", 0)
    )
    num_epoch = int(getattr(args, "num_epoch", 1))
    if optimizer_warmup_epochs < 0:
        errors.append("optimizer_warmup_epochs must be >= 0")
    if optimizer_warmup_epochs > 0 and optimizer_warmup_epochs >= num_epoch:
        errors.append("optimizer_warmup_epochs must be smaller than num_epoch")
    if (
        optimizer_warmup_epochs > 0
        and str(getattr(args, "scheduler", "cosine")).lower() != "cosine"
    ):
        errors.append("optimizer_warmup_epochs > 0 requires scheduler=cosine")

    if int(getattr(args, "num_workers", 0)) < 0:
        errors.append("num_workers must be non-negative")
    if int(getattr(args, "prefetch_factor", 2)) < 1:
        errors.append("prefetch_factor must be >= 1")
    if int(getattr(args, "log_every_n_steps", 1)) < 1:
        errors.append("log_every_n_steps must be >= 1")
    if int(getattr(args, "tri_export_chunksize", 1000)) < 1:
        errors.append("tri_export_chunksize must be >= 1")
    if str(getattr(args, "tri_export_compression", "none")).lower() not in {"none", "gzip"}:
        errors.append("tri_export_compression must be 'none' or 'gzip'")
    if str(getattr(args, "amp_dtype", "fp16")).lower() not in {"fp16", "bf16"}:
        errors.append("amp_dtype must be 'fp16' or 'bf16'")
    if as_bool(getattr(args, "evaluate", False), False) and not str(
        getattr(args, "resume", "") or ""
    ).strip():
        errors.append("evaluate=true requires --resume with a checkpoint path")

    if as_bool(getattr(args, "enable_tri_factor", True), True):
        if not (
            as_bool(getattr(args, "tri_use_zs", True), True)
            or as_bool(getattr(args, "tri_use_zu", True), True)
        ):
            errors.append("enable_tri_factor requires at least one of tri_use_zs or tri_use_zu")
    if as_bool(getattr(args, "enable_bg_calibration", False), False):
        if not as_bool(getattr(args, "enable_tri_factor", True), True):
            errors.append("enable_bg_calibration requires enable_tri_factor=true")
        if not as_bool(getattr(args, "tri_use_zb", True), True):
            errors.append("enable_bg_calibration requires tri_use_zb=true")

    if errors:
        raise ValueError("Invalid CFSurv configuration:\n- " + "\n- ".join(errors))
