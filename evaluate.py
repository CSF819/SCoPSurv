"""Load the distributed best checkpoints with their saved model configuration."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_survival_patch_only import Generic_MIL_Survival_Dataset
from utils.probe_checkpoints import DATASETS, load_probe_model


ROOT = Path(__file__).resolve().parent


def evaluate_validation(model, saved, fold, feature_root, device):
    """Use training-fold genomic scaling and the saved five-fold patient split."""
    saved.data_root_dir = str(feature_root)
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=str(ROOT / "csv" / f"tcga_{saved.dataset}_all_clean.csv"),
        modal=saved.modal,
        apply_sig=True,
        data_dir=str(feature_root / saved.dataset.upper()),
        shuffle=False,
        seed=saved.seed,
        patient_strat=False,
        n_bins=saved.n_classes,
        label_col="survival_months",
        args=saved,
    )
    _, validation = dataset.return_splits(
        from_id=False,
        csv_path=str(ROOT / "splits" / saved.which_splits / f"tcga_{saved.dataset}" / f"splits_{fold}.csv"),
        set_name=saved.dataset,
        extractor=saved.extractor,
    )
    if dataset.num_sites != saved.num_sites:
        raise ValueError("The gene table's site vocabulary differs from the checkpoint.")
    availability = validation.audit_wsi_files()
    if availability["missing"]:
        raise FileNotFoundError(f"Validation WSI features are missing: {availability}")
    risks, times, censorships = [], [], []
    loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{saved.dataset} fold {fold}"):
            path, omic, label, time, censorship, _, uid, site = batch
            output = model(
                x_path=path.float().to(device),
                x_omic=omic.float().to(device),
                phase="eval",
                label=label.float().to(device),
                c=censorship.float().to(device),
                patient_uid=uid.to(device),
                site=site.to(device),
            )
            risks.extend((-output["S"][-1].sum(dim=1)).cpu().tolist())
            times.extend(time.reshape(-1).tolist())
            censorships.extend(censorship.reshape(-1).tolist())
    if validation.wsi_load_stats["corrupt"] or validation.wsi_load_stats["dimension_errors"]:
        raise ValueError(f"Invalid WSI features: {validation.wsi_load_stats}")
    return float(concordance_index_censored(
        (1 - np.asarray(censorships)).astype(bool), np.asarray(times), np.asarray(risks), tied_tol=1e-8
    )[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", default=",".join(DATASETS), help="Comma-separated TCGA cohorts")
    parser.add_argument("--fold", default="0,1,2,3,4", help="Comma-separated fold indices")
    parser.add_argument("--data_root_dir", type=Path, help="Root containing DATASET/patch/slide.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check_only", action="store_true", help="Check strict loading and a synthetic forward pass; no WSI files needed")
    parser.add_argument("--output", type=Path, default=Path("results/checkpoint_evaluation.csv"))
    options = parser.parse_args()
    selected = [name.strip().lower() for name in options.sets.split(",")]
    try:
        folds = {int(value) for value in options.fold.split(",")}
    except ValueError:
        parser.error("--fold must contain integers from 0 through 4")
    if not selected or not set(selected) <= set(DATASETS) or not folds or not folds <= set(range(5)):
        parser.error("Choose supported cohorts and folds 0 through 4")
    if not options.check_only and (options.data_root_dir is None or not options.data_root_dir.is_dir()):
        parser.error("A valid --data_root_dir is required for validation")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    feature_root = options.data_root_dir.resolve() if options.data_root_dir else None
    output_path = options.output.resolve()
    os.chdir(ROOT)  # The original dataset loader reads splits/signatures.csv relative to the project.
    torch.set_num_threads(4)
    device = torch.device(options.device)
    sources = json.loads((ROOT / "metadata/checkpoint_sources.json").read_text(encoding="utf-8"))["checkpoints"]
    rows = []
    for record in sources:
        if record["dataset"] not in selected or record["fold"] not in folds:
            continue
        model, saved, fold, provenance = load_probe_model(ROOT / record["relative_path"], device)
        if abs(provenance["best_score"] - record["expected_c_index"]) > 1e-10:
            raise ValueError(f"Checkpoint score disagrees with the source summary: {record['relative_path']}")
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(100000 + int(saved.seed) * 1000 + fold)
            if options.check_only:
                with torch.no_grad():
                    result = model(
                        x_path=torch.randn(1, max(64, int(getattr(saved, "min_tokens", 50))), model.C_sp, device=device),
                        x_omic=torch.randn(1, model.genomics_idx[-1], device=device),
                        phase="eval",
                    )
                hazards = result["hazards"][-1]
                if hazards.shape != (1, saved.n_classes) or not torch.isfinite(hazards).all():
                    raise ValueError(f"Invalid forward output: {record['relative_path']}")
                measured = ""
            else:
                measured = evaluate_validation(model, saved, fold, feature_root, device)
        row = {
            "dataset": saved.dataset, "fold": fold, "best_epoch": provenance["epoch"],
            "saved_validation_c_index": provenance["best_score"],
            "evaluated_validation_c_index": measured, "strict_state_loading": True,
            "synthetic_forward_checked": options.check_only,
            "architecture": provenance["architecture"], "checkpoint": record["relative_path"],
        }
        rows.append(row)
        print(f"{saved.dataset} fold {fold}: epoch={row['best_epoch']}, saved C-index={row['saved_validation_c_index']:.6f}, checked")
        del model
    if len(rows) != len(set(selected)) * len(folds):
        raise ValueError("The requested checkpoint set is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} checkpoint records to {output_path}")


if __name__ == "__main__":
    main()
