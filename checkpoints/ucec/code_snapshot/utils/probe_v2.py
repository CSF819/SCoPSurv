"""Frozen background, INLP and site-conditional diagnostics (experiment 1 v2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from utils.independent_probes import (
    ProbeConfig, REPRESENTATIONS, SurvivalEvaluation, build_head, make_splits,
    predict_neural, site_scores, survival_nll_rows, validate_frames, write_json,
)
from utils.loss_factory_new import nll_loss
from utils.tri_factor_probe import _vector_columns

DATASETS = ("blca", "brca", "gbmlgg", "luad", "ucec")
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class V2Config(ProbeConfig):
    strengths: tuple = (0, 1, 2, 4, 8)
    direction_seeds: tuple = (0, 1, 2, 3, 4)
    thresholds: tuple = (5, 10)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def background_pipeline(n_fit, dimension, seed):
    return make_pipeline(StandardScaler(), PCA(
        n_components=min(128, dimension, n_fit - 1), whiten=False,
        svd_solver="randomized", random_state=seed),
        RidgeClassifier(alpha=1, class_weight="balanced"))


def row_basis(matrix):
    """Orthonormal row basis with the predeclared absolute/relative tolerance."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    _, values, vectors = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(1e-10, max(matrix.shape) * np.finfo(float).eps *
                    (float(values[0]) if len(values) else 0.0))
    return vectors[values > threshold], threshold


def inlp_path(x, y, strengths=(0, 1, 2, 4, 8)):
    """Always fit on training data; remove the union of learned row spaces."""
    x = np.asarray(x, dtype=np.float64)
    dimension = x.shape[1]
    basis = np.empty((0, dimension))
    projector = np.eye(dimension)
    projectors = {0: projector.copy()}
    records = [{"iteration": 0, "actual_iteration": 0, "removed_rank": 0,
                "new_rank": 0, "status": "identity"}]
    stopped, actual = False, 0
    for iteration in range(1, max(strengths) + 1):
        new_rank, tolerance = 0, 1e-10
        if not stopped:
            classifier = RidgeClassifier(alpha=1, class_weight="balanced").fit(x @ projector, y)
            # Mapping back to the retained subspace prevents numerical re-removal.
            learned, tolerance = row_basis(classifier.coef_ @ projector)
            new_rank = len(learned)
            if new_rank:
                basis, _ = row_basis(np.vstack([basis, learned]))
                projector = np.eye(dimension) - basis.T @ basis
                projector = (projector + projector.T) / 2
                actual = iteration
            else:
                stopped = True
        records.append({"iteration": iteration, "actual_iteration": actual,
                        "removed_rank": len(basis), "new_rank": new_rank,
                        "svd_tolerance": tolerance,
                        "status": "no_new_direction" if stopped else "ok"})
        if iteration in strengths:
            projectors[iteration] = projector.copy()
    return projectors, records


def random_path(dimension, ranks, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    return {step: np.eye(dimension) - q[:, :rank] @ q[:, :rank].T
            for step, rank in ranks.items()}


def _split_log(train, evaluation, splits, fold):
    result = {key: train.case_id.iloc[value].tolist() for key, value in splits.items()
              if key not in ("site_classes", "site_eval")}
    result.update(site_classes=splits["site_classes"],
                  site_eval=evaluation.case_id.iloc[splits["site_eval"]].tolist(),
                  inner_split_seed=2026 + fold)
    return result


def load_reference(reference, dataset, fold):
    reference = Path(reference).resolve()
    selected = [r for r in read_json(reference / "best_checkpoints.json")["selected"]
                if r["dataset"] == dataset]
    if len(selected) != 1 or len(selected[0]["checkpoints"]) != 5:
        raise ValueError(f"{dataset}: expected exactly one fixed five-fold checkpoint selection")
    folder = reference / dataset / f"fold_{fold}"
    metadata = read_json(folder / "embeddings_metadata.json")
    if metadata["dataset"] != dataset or int(metadata["fold"]) != fold:
        raise ValueError("Dataset/fold metadata mismatch")
    checkpoint = Path(selected[0]["checkpoints"][fold]).resolve()
    if checkpoint != Path(metadata["checkpoint"]).resolve():
        raise ValueError("Frozen cache does not belong to the selected checkpoint")
    for name in ("checkpoint", "clinical_csv", "split_csv"):
        recorded = metadata["source"][name]
        if identity(recorded["path"]) != recorded:
            raise ValueError(f"Changed cache source: {name}")
    if not metadata.get("encoder_state_unchanged") or not metadata.get("repeated_forward_verified"):
        raise ValueError("Cache lacks frozen/repeated-forward verification")
    train = pd.read_csv(folder / "embeddings_train.csv.gz")
    evaluation = pd.read_csv(folder / "embeddings_eval.csv.gz")
    validate_frames(train, evaluation)
    if [len(train), len(evaluation)] != [metadata["train_n"], metadata["eval_n"]]:
        raise ValueError("Cache patient count disagrees with export metadata")
    cuts = np.asarray(metadata["finite_cuts"], dtype=float)
    if cuts.shape != (3,) or not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
        raise ValueError("Invalid original training survival cut points")
    raw_split = pd.read_csv(metadata["source"]["split_csv"]["path"])
    universe = set(train.case_id) | set(evaluation.case_id)
    for frame, name in ((train, "train"), (evaluation, "val")):
        if set(frame.case_id) != (set(raw_split[name].dropna().astype(str)) & universe):
            raise ValueError(f"Cache patient IDs disagree with original {name} partition")
        if not (frame.fold == fold).all() or not (frame.split == ("eval" if name == "val" else name)).all():
            raise ValueError("Row-level fold/split mismatch")
        if not np.array_equal(frame.label, np.searchsorted(cuts, frame.event_time, side="right")):
            raise ValueError("Outcome labels disagree with original training cuts")
        if not np.array_equal(frame.site.astype(str).map(metadata["site_vocabulary"]), frame.site_idx):
            raise ValueError("Site vocabulary mismatch")
    splits = make_splits(train, evaluation, fold, 5)
    saved = read_json(folder / "patient_splits.json")
    generated = _split_log(train, evaluation, splits, fold)
    for key in ("site_classes", "site_train", "site_eval", "site_fit", "site_val", "inner_split_seed"):
        if saved[key] != generated[key]:
            raise ValueError(f"Original probe split mismatch: {key}")
    provenance = {"reference_dir": str(reference), "dataset": dataset, "fold": fold,
                  "checkpoint": identity(checkpoint), "cache_sources": metadata["source"],
                  "finite_cuts": cuts.tolist(), "frozen_verification": True,
                  "files": {name: identity(folder / name) for name in (
                      "embeddings_train.csv.gz", "embeddings_eval.csv.gz", "embeddings_metadata.json",
                      "patient_splits.json", "predictions.csv.gz", "selection.json", "complete.json")}}
    return train, evaluation, metadata, splits, provenance


def prepare_fold(reference, output, dataset, fold, config):
    train, evaluation, metadata, splits, provenance = load_reference(reference, dataset, fold)
    out = Path(output) / dataset / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"version": PROTOCOL_VERSION, "config": asdict(config), "provenance": provenance}
    # JSON normalizes tuple-valued defaults for reproducible resume comparisons.
    normalized = json.loads(json.dumps(protocol))
    if (out / "prepared.json").exists() and read_json(out / "prepared.json") != normalized:
        raise ValueError("Output has a different protocol/source; choose a new output directory")
    write_json(out / "prepared.json", protocol)
    write_json(out / "patient_splits.json", _split_log(train, evaluation, splits, fold))
    write_json(out / "embeddings_metadata.json", metadata)
    exclusions = []
    for threshold in config.thresholds:
        local = make_splits(train, evaluation, fold, threshold)
        write_json(out / f"patient_splits_n{threshold}.json", _split_log(train, evaluation, local, fold))
        counts = train.site_idx.value_counts()
        for frame, name in ((train, "train"), (evaluation, "eval")):
            rows = frame[["case_id", "site_idx"]].copy()
            rows["partition"], rows["threshold"] = name, threshold
            rows["training_site_n"] = rows.site_idx.map(counts).fillna(0).astype(int)
            rows["eligible"] = rows.site_idx.isin(local["site_classes"])
            rows["reason"] = np.where(rows.eligible, "included", np.where(
                rows.training_site_n == 0, "unseen_or_unknown_site", "training_site_below_threshold"))
            exclusions.append(rows)
    pd.concat(exclusions, ignore_index=True).to_csv(out / "coverage.csv", index=False)
    return train, evaluation, metadata, splits, out


def neural_fit(x, y, task, family, outputs, config, decay, seed, validation=None, epochs=None):
    """Original objective/optimizer, with selected-state restoration and fast BA."""
    torch.manual_seed(seed)
    model = build_head(x.shape[1], outputs, family, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=decay)
    features = torch.as_tensor(np.asarray(x, dtype=np.float32))
    target = torch.as_tensor(np.asarray(y), dtype=torch.long if task == "site" else torch.float32)
    generator = torch.Generator().manual_seed(seed)
    weights = None
    if task == "site":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=outputs)
        if (counts == 0).any():
            raise ValueError("All eligible classes must occur in fitting data")
        weights = torch.tensor(len(y) / (outputs * counts), dtype=torch.float32)
    best_value, best_epoch, stale, best_state = -np.inf, 1, 0, None
    history = []
    if validation is not None and task == "site":
        val_y = np.asarray(validation[1])
        val_weight = np.array([1 / (len(np.unique(val_y)) * np.sum(val_y == c)) for c in val_y])
    for epoch in range(1, (epochs if epochs is not None else config.max_epochs) + 1):
        model.train()
        for batch in torch.randperm(len(features), generator=generator).split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[batch])
            if task == "site":
                loss = nn.functional.cross_entropy(logits, target[batch], weight=weights)
            else:
                hazards = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
                loss = nll_loss(hazards, None, target[batch, 0].long(), target[batch, 1], alpha=0)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        if validation is not None:
            pred = predict_neural(model, validation[0], task)
            value = (float(np.dot(pred.argmax(1) == val_y, val_weight)) if task == "site" else
                     -float(survival_nll_rows(pred, validation[1][:, 0], validation[1][:, 1]).mean()))
            history.append({"epoch": epoch, "selection_value": value})
            if value > best_value + 1e-12:
                best_value, best_epoch, stale = value, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
    if validation is not None:
        if best_state is None:
            raise ValueError("No finite internal validation objective")
        model.load_state_dict(best_state)
    return model.eval(), {"epoch": best_epoch if validation is not None else epochs,
                          "selection_value": best_value, "history": history}


def write_predictions(folder, frame, predictions, task, classes, evaluator, ident):
    # Pandas serializes float32 columns with fewer digits. Near-one hazards then
    # change survival likelihoods substantially, so preserve exact float32 values
    # by widening to float64 before CSV serialization (no prediction rounding).
    predictions = np.asarray(predictions, dtype=np.float64)
    rows = frame[["case_id", "site_idx", "event_time", "censorship", "label"]].copy()
    for key, value in ident.items():
        rows[key] = value
    if task == "site":
        rows["predicted_site"] = np.asarray(classes)[predictions.argmax(1)]
        for j, site in enumerate(classes):
            rows[f"site_prob_{site}"] = predictions[:, j]
    else:
        for j in range(4):
            rows[f"hazard_{j}"] = predictions[:, j]
        rows["risk"] = -np.cumprod(1 - predictions.astype(float), axis=1).sum(1)
        rows["evaluation_time"], rows["evaluation_event"] = evaluator.time, evaluator.event.astype(int)
        rows["evaluation_label"] = evaluator.labels
    return rows


def refresh_saved_predictions(reference, output, dataset, fold, config):
    """Re-export existing head predictions at full precision; never fit a model."""
    train, evaluation, metadata, splits, provenance = load_reference(reference, dataset, fold)
    out = Path(output) / dataset / f"fold_{fold}"
    if read_json(out / "prepared.json")["provenance"] != provenance:
        raise ValueError("Saved head/cache source mismatch")
    classes = splits["site_classes"]
    cohort = evaluation.iloc[splits["site_eval"]].reset_index(drop=True)
    evaluator = SurvivalEvaluation(train.iloc[splits["site_train"]], cohort, metadata["finite_cuts"])
    mapping = {site: i for i, site in enumerate(classes)}
    one_hot = np.eye(len(classes))[cohort.site_idx.map(mapping).to_numpy(int)]
    unit_count = 0
    for stage in ("erase", "conditional"):
        root = out / stage
        for condition in read_json(root / "conditions.json"):
            name, direction = condition["condition"], condition["direction"]
            if stage == "erase":
                rep = "z_u" if name == "z_u" else "h"
                scaler = joblib.load(root / f"{rep}_scalers.joblib")["final"]
                features = scaler.transform(cohort[_vector_columns(cohort, rep)].to_numpy(float))
                if name.startswith(("inlp_", "random_")):
                    filename = "inlp_projectors.npz" if name.startswith("inlp_") else f"random_projectors_seed{direction}.npz"
                    with np.load(root / filename) as matrices:
                        features = features @ matrices[f"final_{condition['strength']}"]
            elif name == "site_only":
                features = one_hot
            else:
                rep = name.removeprefix("site_plus_")
                scaler = joblib.load(root / f"{rep}_scalers.joblib")["final"]
                continuous = scaler.transform(cohort[_vector_columns(cohort, rep)].to_numpy(float))
                features = np.column_stack([one_hot, continuous])
            # The original arrays used row-indexed (C-contiguous) cache slices.
            # Preserve that layout so float32 GEMM reproduces the saved metrics.
            features = np.ascontiguousarray(features)
            for task in (("site", "survival") if stage == "erase" else ("survival",)):
                for family in ("linear", "mlp"):
                    folder = root / f"{name}_direction{direction}" / f"{task}_{family}"
                    if not (folder / "complete.json").exists():
                        raise ValueError(f"Cannot refresh incomplete head: {folder}")
                    ident = {"dataset": dataset, "fold": fold, "stage": stage, "condition": name,
                             "direction": direction, "strength": condition["strength"], "threshold": 5,
                             "task": task, "family": family}
                    predictions = []
                    existing = pd.read_csv(folder / "metrics.csv").set_index("seed")
                    for seed in ((0,) if task == "site" and family == "linear" else config.seeds):
                        if task == "site" and family == "linear":
                            model = joblib.load(folder / f"head_seed{seed}.joblib")
                            pred = model.predict_proba(features)
                        else:
                            saved = torch.load(folder / f"head_seed{seed}.pt", map_location="cpu", weights_only=True)
                            model = build_head(saved["input_dim"], saved["outputs"], saved["family"], config)
                            model.load_state_dict(saved["state_dict"], strict=True)
                            pred = predict_neural(model, features, task)
                        # Check against metrics computed BEFORE the lossy export.
                        scores = (site_scores(cohort.site_idx, np.asarray(classes)[pred.argmax(1)]) if task == "site"
                                  else evaluator.score(pred))
                        for metric in (("balanced_accuracy", "macro_f1") if task == "site" else ("ipcw_cindex", "nll")):
                            if not np.isclose(scores[metric], existing.loc[seed, metric], atol=1e-10, rtol=0, equal_nan=True):
                                raise ValueError(f"Reconstructed prediction does not reproduce {folder}/{seed}/{metric}")
                        predictions.append(write_predictions(folder, cohort, pred, task, classes, evaluator, {**ident, "seed": seed}))
                    pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv.gz", index=False)
                    unit_count += 1
    write_json(out / "prediction_precision.json", {"status": "complete", "units": unit_count,
                                                  "serialization": "float64_exact_values_of_model_outputs",
                                                  "original_head_metrics_reproduced": True,
                                                  "head_weights_retrained": False, "code": identity(__file__)})
    return unit_count


def run_head(folder, arrays, train_y, val_y, full_y, evaluation, classes, evaluator,
             task, family, config, ident):
    folder = Path(folder)
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    fit_x, val_x, full_x, eval_x = arrays
    logistic = task == "site" and family == "linear"
    outputs = len(classes) if task == "site" else 4
    candidates, best_model = [], None
    for strength in config.c_grid if logistic else config.weight_decay_grid:
        if logistic:
            model = LogisticRegression(C=strength, class_weight="balanced", max_iter=5000,
                                       solver="lbfgs", random_state=0).fit(fit_x, train_y)
            detail = {"C": strength, "epoch": None,
                      "selection_value": site_scores(val_y, model.predict(val_x))["balanced_accuracy"]}
        else:
            model, detail = neural_fit(fit_x, train_y, task, family, outputs, config, strength, 0,
                                       validation=(val_x, val_y))
            detail["weight_decay"] = strength
        candidates.append(detail)
        key = lambda d: (-d["selection_value"], d["C"] if logistic else -d["weight_decay"], d["epoch"] or 0)
        if detail is min(candidates, key=key):
            best_model = model
    selected = min(candidates, key=key)
    diagnostics = []
    for name, features, target in (("fit", fit_x, train_y), ("inner_validation", val_x, val_y)):
        pred = best_model.predict_proba(features) if logistic else predict_neural(best_model, features, task)
        score = site_scores(target, pred.argmax(1)) if task == "site" else {
            "nll": float(survival_nll_rows(pred, target[:, 0], target[:, 1]).mean())}
        diagnostics.append({"partition": name, "n": len(features), **score})
    write_json(folder / "selection.json", {"selected": selected, "candidates": candidates,
                                          "selected_epoch_diagnostics": diagnostics})
    metrics, all_predictions, recalls = [], [], []
    for seed in (0,) if logistic else config.seeds:
        if logistic:
            head = LogisticRegression(C=selected["C"], class_weight="balanced", max_iter=5000,
                                      solver="lbfgs", random_state=seed).fit(full_x, full_y)
            pred = head.predict_proba(eval_x)
            joblib.dump(head, folder / f"head_seed{seed}.joblib")
        else:
            head, _ = neural_fit(full_x, full_y, task, family, outputs, config,
                                 selected["weight_decay"], seed, epochs=selected["epoch"])
            pred = predict_neural(head, eval_x, task)
            torch.save({"state_dict": head.state_dict(), "input_dim": full_x.shape[1],
                        "outputs": outputs, "family": family, "config": asdict(config)},
                       folder / f"head_seed{seed}.pt")
        record = {**ident, "task": task, "family": family, "seed": seed}
        if task == "site":
            y = evaluation.site_idx.to_numpy()
            predicted = np.asarray(classes)[pred.argmax(1)]
            score = {"n": len(y), "train_n": len(full_y), "train_site_classes": len(classes),
                     "eval_site_classes": len(np.unique(y)), "uniform_chance": 1 / len(classes),
                     **site_scores(y, predicted)}
            recalls.extend({**record, "site_idx": int(c), "n": int(np.sum(y == c)),
                            "recall": float(np.mean(predicted[y == c] == c))} for c in np.unique(y))
        else:
            score = evaluator.score(pred)
        metrics.append({**record, **score})
        all_predictions.append(write_predictions(folder, evaluation, pred, task, classes, evaluator, record))
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(folder / "predictions.csv.gz", index=False)
    if recalls:
        pd.DataFrame(recalls).to_csv(folder / "site_recall.csv", index=False)
    write_json(folder / "complete.json", {"status": "complete", **ident, "task": task, "family": family})


def site_representation_folder(out, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported site representation: {representation}")
    return Path(out) / "background" if representation == "z_b" else Path(out) / "site_comparison" / representation


def run_background(reference, train, evaluation, metadata, out, config, representation="z_b"):
    fold, dataset = metadata["fold"], metadata["dataset"]
    root = site_representation_folder(out, representation)
    root.mkdir(parents=True, exist_ok=True)
    x = train[_vector_columns(train, representation)].to_numpy(float)
    ex = evaluation[_vector_columns(evaluation, representation)].to_numpy(float)
    for threshold in config.thresholds:
        folder = root / f"ridge_n{threshold}"
        if (folder / "complete.json").exists():
            continue
        folder.mkdir(exist_ok=True)
        splits = make_splits(train, evaluation, fold, threshold)
        fi, vi, ti, ei = [splits[key] for key in ("site_fit", "site_val", "site_train", "site_eval")]
        y = train.site_idx.to_numpy(int)
        inner = background_pipeline(len(fi), x.shape[1], 2026 + fold).fit(x[fi], y[fi])
        final = background_pipeline(len(ti), x.shape[1], 2026 + fold).fit(x[ti], y[ti])
        joblib.dump({"inner": inner, "final": final}, folder / "preprocessing_and_heads.joblib")
        diagnostics, predictions, recalls = [], [], []
        ident = {"dataset": dataset, "fold": fold, "stage": "background", "condition": f"{representation}_n{threshold}",
                 "threshold": threshold, "family": "ridge", "task": "site", "seed": 0, "direction": -1}
        for name, frame, features, head in (
            ("fit", train.iloc[fi], x[fi], inner), ("inner_validation", train.iloc[vi], x[vi], inner),
            ("eval", evaluation.iloc[ei], ex[ei], final)):
            pred = head.predict(features)
            score = {**ident, "partition": name, "n": len(frame), "train_n": len(ti),
                     "train_site_classes": len(splits["site_classes"]),
                     "eval_site_classes": frame.site_idx.nunique(), "uniform_chance": 1 / len(splits["site_classes"]),
                     **site_scores(frame.site_idx.to_numpy(), pred)}
            diagnostics.append(score)
            rows = frame[["case_id", "site_idx"]].copy()
            rows["predicted_site"] = pred
            for key, value in {**ident, "partition": name}.items():
                rows[key] = value
            predictions.append(rows)
            if name == "eval":
                recalls = [{**ident, "site_idx": int(c), "n": int(np.sum(frame.site_idx == c)),
                            "recall": float(np.mean(pred[frame.site_idx.to_numpy() == c] == c))}
                           for c in np.unique(frame.site_idx)]
        permutations, perm_predictions = [], []
        transformed, eval_transformed = final[:-1].transform(x[ti]), final[:-1].transform(ex[ei])
        perm_labels = []
        for repeat in range(config.permutations):
            seed = 30000 + 100 * fold + repeat
            shuffled = np.random.default_rng(seed).permutation(y[ti])
            perm_labels.append(shuffled)
            head = RidgeClassifier(alpha=1, class_weight="balanced").fit(transformed, shuffled)
            pred = head.predict(eval_transformed)
            permutations.append({**ident, "repeat": repeat, "permutation_seed": seed,
                                 **site_scores(evaluation.site_idx.iloc[ei].to_numpy(), pred)})
            perm_predictions.append(pred)
        np.savez_compressed(folder / "permutation_predictions.npz", case_id=evaluation.case_id.iloc[ei].to_numpy(str),
                            predictions=np.asarray(perm_predictions), training_case_id=train.case_id.iloc[ti].to_numpy(str),
                            permuted_training_labels=np.asarray(perm_labels))
        pd.DataFrame(permutations).to_csv(folder / "permutations.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(folder / "metrics.csv", index=False)
        pd.DataFrame(recalls).to_csv(folder / "site_recall.csv", index=False)
        pd.concat(predictions).to_csv(folder / "predictions.csv.gz", index=False)
        write_json(folder / "complete.json", {"status": "complete", "inner_pca_dim": inner[1].n_components_,
                                              "final_pca_dim": final[1].n_components_, **ident})
    # Reuse this representation's existing native-dimensional site probes.
    folder = root / "native_supplement"
    if not (folder / "complete.json").exists():
        source = Path(reference) / dataset / f"fold_{fold}"
        if read_json(source / "complete.json")["config"] != json.loads(json.dumps(asdict(ProbeConfig()))):
            raise ValueError("Existing supplementary probes do not use the declared v1 protocol")
        predictions = pd.read_csv(source / "predictions.csv.gz", float_precision="round_trip", low_memory=False)
        predictions = predictions[(predictions.task == "site") & (predictions.representation == representation)].copy()
        classes = make_splits(train, evaluation, fold, 5)
        expected = evaluation.iloc[classes["site_eval"]].set_index("case_id")
        for (family, seed), group in predictions.groupby(["family", "seed"]):
            if group.case_id.duplicated().any() or set(group.case_id) != set(expected.index):
                raise ValueError("Supplement patient predictions are incomplete")
            if not np.array_equal(group.set_index("case_id").loc[expected.index, "site_idx"], expected.site_idx):
                raise ValueError("Supplement site labels differ")
        if set(zip(predictions.family, predictions.seed)) != {("linear", 0), ("mlp", 0), ("mlp", 1), ("mlp", 2)}:
            raise ValueError("Supplement probe family/seed coverage is incomplete")
        folder.mkdir(exist_ok=True)
        for key, value in {"dataset": dataset, "stage": "background", "condition": f"{representation}_n5",
                           "threshold": 5, "direction": -1, "partition": "eval"}.items():
            predictions[key] = value
        predictions.to_csv(folder / "predictions.csv.gz", index=False)
        source_metrics = pd.read_csv(source / "metrics.csv")
        source_metrics = source_metrics[(source_metrics.task == "site") & (source_metrics.representation == representation)].copy()
        for key, value in {"dataset": dataset, "stage": "background", "condition": f"{representation}_n5", "threshold": 5,
                           "direction": -1, "partition": "eval"}.items():
            source_metrics[key] = value
        source_metrics.to_csv(folder / "metrics.csv", index=False)
        selections = [r for r in read_json(source / "selection.json") if r["task"] == "site" and r["representation"] == representation]
        write_json(folder / "selection.json", selections)
        write_json(folder / "complete.json", {"status": "complete", "source": str(source),
                                              "models": str(source / "models"), "reused": True})


def scaled_arrays(train, evaluation, splits, rep):
    x = train[_vector_columns(train, rep)].to_numpy(float)
    ex = evaluation[_vector_columns(evaluation, rep)].to_numpy(float)
    fi, vi, ti, ei = [splits[key] for key in ("site_fit", "site_val", "site_train", "site_eval")]
    inner, final = StandardScaler().fit(x[fi]), StandardScaler().fit(x[ti])
    return (inner.transform(x[fi]), inner.transform(x[vi]), final.transform(x[ti]), final.transform(ex[ei])), {
        "inner": inner, "final": final}


def run_decoding(train, evaluation, metadata, splits, out, stage, config, log=print):
    root = out / stage
    root.mkdir(exist_ok=True)
    fi, vi, ti, ei = [splits[key] for key in ("site_fit", "site_val", "site_train", "site_eval")]
    classes = splits["site_classes"]
    mapping = {site: j for j, site in enumerate(classes)}
    site_target = train.site_idx.map(mapping).fillna(-1).to_numpy(int)
    survival_target = train[["label", "censorship"]].to_numpy(np.float32)
    cohort = evaluation.iloc[ei].reset_index(drop=True)
    evaluator = SurvivalEvaluation(train.iloc[ti].reset_index(drop=True), cohort, metadata["finite_cuts"])
    fold = metadata["fold"]
    if stage == "erase":
        arrays, scalers = scaled_arrays(train, evaluation, splits, "h")
        inner_p, inner_records = inlp_path(arrays[0], site_target[fi], config.strengths)
        final_p, final_records = inlp_path(arrays[2], site_target[ti], config.strengths)
        ranks = [{r["iteration"]: r["removed_rank"] for r in records if r["iteration"] in config.strengths}
                 for records in (inner_records, final_records)]
        joblib.dump(scalers, root / "h_scalers.joblib")
        write_json(root / "projection_ranks.json", {"inner": inner_records, "final": final_records})
        np.savez_compressed(root / "inlp_projectors.npz", **{
            f"{name}_{step}": matrix for name, path in (("inner", inner_p), ("final", final_p)) for step, matrix in path.items()})
        conditions = [("h", -1, 0, arrays)]
        zu_arrays, zu_scalers = scaled_arrays(train, evaluation, splits, "z_u")
        joblib.dump(zu_scalers, root / "z_u_scalers.joblib")
        conditions.append(("z_u", -1, 0, zu_arrays))
        for step in config.strengths:
            if step:
                conditions.append((f"inlp_{step}", -1, step, tuple(
                    a @ (inner_p if j < 2 else final_p)[step] for j, a in enumerate(arrays))))
        for direction in config.direction_seeds:
            seed = 73000 + 100 * fold + direction
            ip, fp = (random_path(arrays[0].shape[1], rank, seed) for rank in ranks)
            np.savez_compressed(root / f"random_projectors_seed{direction}.npz", **{
                f"{name}_{step}": matrix for name, path in (("inner", ip), ("final", fp)) for step, matrix in path.items()})
            for step in config.strengths:
                if step:
                    conditions.append((f"random_{step}", direction, step, tuple(
                        a @ (ip if j < 2 else fp)[step] for j, a in enumerate(arrays))))
    else:
        eye = np.eye(len(classes))
        site_arrays = (eye[site_target[fi]], eye[site_target[vi]], eye[site_target[ti]],
                       eye[cohort.site_idx.map(mapping).to_numpy(int)])
        conditions = [("site_only", -1, 0, site_arrays)]
        for rep in REPRESENTATIONS:
            arrays, scalers = scaled_arrays(train, evaluation, splits, rep)
            joblib.dump(scalers, root / f"{rep}_scalers.joblib")
            conditions.append((f"site_plus_{rep}", -1, 0, tuple(
                np.column_stack([e, a]) for e, a in zip(site_arrays, arrays))))
        write_json(root / "site_vocabulary.json", {"classes": classes, "one_hot_standardized": False})
    write_json(root / "conditions.json", [{"condition": name, "direction": direction, "strength": strength,
                                           "input_dim": arrays[0].shape[1]} for name, direction, strength, arrays in conditions])
    for condition, direction, strength, arrays in conditions:
        for task in (("site", "survival") if stage == "erase" else ("survival",)):
            target = site_target if task == "site" else survival_target
            for family in ("linear", "mlp"):
                ident = {"dataset": metadata["dataset"], "fold": fold, "stage": stage, "condition": condition,
                         "direction": direction, "strength": strength, "threshold": 5}
                folder = root / f"{condition}_direction{direction}" / f"{task}_{family}"
                if not (folder / "complete.json").exists():
                    log(f"[{metadata['dataset']} fold {fold}] {stage}/{condition}/{direction}/{task}/{family}", flush=True)
                run_head(folder, arrays, target[fi], target[vi], target[ti], cohort, classes,
                         evaluator, task, family, config, ident)
    write_json(root / "complete.json", {"status": "complete", "conditions": len(conditions),
                                        "eligible_train_n": len(ti), "eligible_eval_n": len(ei),
                                        "censoring_distribution_training": "eligible_original_train",
                                        "administrative_cap": evaluator.cap})


def execute_fold(reference, output, dataset, fold, config, stages):
    started = time.time()
    out = Path(output) / dataset / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        train, evaluation, metadata, splits, out = prepare_fold(reference, output, dataset, fold, config)
        for stage in stages:
            if stage == "background":
                run_background(reference, train, evaluation, metadata, out, config)
                write_json(out / "background/complete.json", {"status": "complete"})
            elif stage in ("erase", "conditional"):
                run_decoding(train, evaluation, metadata, splits, out, stage, config)
            elif stage == "site":
                for representation in REPRESENTATIONS:
                    run_background(reference, train, evaluation, metadata, out, config, representation)
                write_json(out / "site_comparison/complete.json", {
                    "status": "complete", "representations": REPRESENTATIONS,
                    "thresholds": config.thresholds, "paired_patient_splits": True})
        result = {"dataset": dataset, "fold": fold, "status": "complete", "stages": list(stages),
                  "seconds": time.time() - started}
        write_json(out / "status.json", result)
        return result
    except Exception as exc:
        import traceback
        result = {"dataset": dataset, "fold": fold, "status": "failed", "error": str(exc),
                  "traceback": traceback.format_exc(), "seconds": time.time() - started}
        write_json(out / "status.json", result)
        return result
