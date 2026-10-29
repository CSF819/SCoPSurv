"""Select complete runs and restore their original checkpoint architecture."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

DATASETS = ("blca", "brca", "gbmlgg", "luad", "ucec")


def discover_best_runs(results_root, datasets=DATASETS):
    candidates, excluded = [], []
    for run in sorted(Path(results_root).rglob("*_CFSurv_*")):
        if not run.is_dir():
            continue
        for summary_path in sorted(run.glob("*.csv")):
            table = pd.read_csv(summary_path)
            if "name" not in table:
                continue
            for dataset in datasets:
                records = table[table.name.eq(dataset)]
                if records.empty:
                    continue
                if len(records) != 1:
                    raise ValueError(f"Ambiguous summary row: {summary_path}, {dataset}")
                record = records.iloc[0]
                scores = np.asarray([record.get(f"fold {fold}", np.nan) for fold in range(5)], dtype=float)
                paths = [sorted((run / f"{dataset}_fold_{fold}").glob("model_best*.pth.tar")) for fold in range(5)]
                reason = "" if np.isfinite(scores).all() else "incomplete_five_fold_summary"
                if not reason and any(len(p) != 1 for p in paths):
                    reason = "missing_or_ambiguous_best_checkpoint"
                row = {"dataset": dataset, "run_dir": str(run.resolve()),
                       "summary_path": str(summary_path.resolve()), "fold_scores": scores.tolist()}
                if reason:
                    excluded.append({**row, "reason": reason})
                else:
                    candidates.append({**row, "mean_cindex": float(scores.mean()),
                                       "checkpoints": [str(p[0].resolve()) for p in paths]})
    selected = []
    for dataset in datasets:
        ranked = sorted([r for r in candidates if r["dataset"] == dataset],
                        key=lambda r: (-r["mean_cindex"], r["run_dir"]))
        if not ranked:
            raise ValueError(f"No complete five-fold checkpoint set for {dataset}.")
        selected.append(ranked[0])
    return selected, candidates, excluded


def checkpoint_dataset(run_dir):
    names = {p.name.rsplit("_fold_", 1)[0] for p in Path(run_dir).glob("*_fold_[0-4]") if p.is_dir()}
    if len(names) != 1 or not names <= set(DATASETS):
        raise ValueError(f"Cannot identify one dataset in {run_dir}: {sorted(names)}")
    return names.pop()


def load_probe_model(checkpoint_path, device):
    """Strictly load modern models or the checkpoint's saved legacy network.

    Early checkpoints predate adaptive capacity and sample Slot noise only in
    training. The archived compressor preserves that inference behavior.
    """
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_path = path.parent / "run_config.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.exists() else {}
    config = checkpoint.get("args") or saved.get("args")
    if not config:
        raise ValueError(f"Checkpoint configuration is missing: {path}")
    config = dict(config)
    match = re.search(r"_fold(\d+)\.pth\.tar$", path.name)
    if not match:
        raise ValueError(f"Checkpoint filename does not identify a fold: {path}")
    fold = int(checkpoint.get("fold", saved.get("fold", -1)))
    if fold != int(match.group(1)) or saved.get("fold", fold) != fold:
        raise ValueError(f"Checkpoint fold metadata disagrees with its filename: {path}")
    dataset = str(config["dataset"]).lower()
    if path.parent.name != f"{dataset}_fold_{fold}":
        raise ValueError(f"Checkpoint dataset/fold directory mismatch: {path}")
    for flag in ("enable_tri_factor", "tri_use_zb", "tri_use_zs", "tri_use_zu"):
        from utils.config_utils import as_bool
        if not as_bool(config.get(flag, True), True):
            raise ValueError(f"All three branches are required: {flag}=false in {path}")
    args = SimpleNamespace(**config)
    if int(checkpoint.get("checkpoint_version", 0)) >= 2:
        from models.CFSurv.network_20260414_CFSurv import CFSurv
        model = CFSurv(args)
        provenance = {"architecture": "current", "checkpoint_version": checkpoint["checkpoint_version"]}
    else:
        snapshot = path.parent / "network_20260414_CFSurv.py"
        if not snapshot.exists():
            raise ValueError(f"Original network snapshot is required for this legacy checkpoint: {snapshot}")
        spec = importlib.util.spec_from_file_location("_probe_legacy_network", snapshot)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from utils.legacy_probe_path_compressor import CFSurvPathSlotCompressor
        module.CFSurvPathSlotCompressor = CFSurvPathSlotCompressor
        model = module.CFSurv(args)
        provenance = {"architecture": "saved_legacy_network", "network_snapshot": str(snapshot),
                      "compressor_source": "repository commit d954198:models/CFSurv/path_compressor.py",
                      "slot_evaluation_initialization": "learned_mean_without_noise"}
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    provenance.update(best_score=float(checkpoint["best_score"]), epoch=int(checkpoint["epoch"]),
                      strict_state_loading=True, fold_source="checkpoint" if "fold" in checkpoint else "run_config.json")
    return model, args, fold, provenance
