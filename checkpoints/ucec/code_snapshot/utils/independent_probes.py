"""Frozen-representation probes for experiment 1 (no encoder optimization)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
from torch import nn

from utils.loss_factory_new import nll_loss
from utils.tri_factor_probe import _vector_columns

REPRESENTATIONS = ("h", "z_b", "z_s", "z_u")


@dataclass(frozen=True)
class ProbeConfig:
    min_site_count: int = 5
    hidden_dim: int = 64
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    c_grid: tuple = (0.01, 0.1, 1.0, 10.0)
    weight_decay_grid: tuple = (0.0001, 0.001, 0.01)
    seeds: tuple = (0, 1, 2)
    permutations: int = 20
    bootstrap: int = 2000


def write_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def validate_frames(train, evaluation):
    required = {"case_id", "site_idx", "event_time", "censorship", "label"}
    for frame in (train, evaluation):
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError("Missing patient metadata or empty frozen embeddings.")
        if frame.case_id.isna().any() or frame.case_id.duplicated().any():
            raise ValueError("Patient identifiers must be present and unique.")
        if not frame.censorship.isin([0, 1]).all():
            raise ValueError("censorship must be 0 (event) or 1 (censored).")
        if not frame.label.isin(range(4)).all():
            raise ValueError("Expected four discrete survival intervals.")
        if not np.isfinite(frame.event_time).all() or (frame.event_time < 0).any():
            raise ValueError("Invalid follow-up times.")
        if not np.isfinite(frame.site_idx).all() or not np.equal(frame.site_idx, frame.site_idx.astype(int)).all():
            raise ValueError("Site identifiers must be integers.")
    if set(train.case_id) & set(evaluation.case_id):
        raise ValueError("Patient leakage between training and evaluation.")
    for rep in REPRESENTATIONS:
        tc, ec = _vector_columns(train, rep), _vector_columns(evaluation, rep)
        if len(tc) != 256 or tc != ec or tc != [f"{rep}_{j:04d}" for j in range(256)]:
            raise ValueError(f"{rep}: expected matching, consecutive 256-dimensional columns.")
        for frame in (train, evaluation):
            if not np.isfinite(frame[tc].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite {rep} features; do not drop different patients per branch.")


def site_scores(y, pred):
    labels = np.unique(y)
    if not len(labels):
        return {"balanced_accuracy": np.nan, "macro_f1": np.nan}
    return {
        "balanced_accuracy": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }


def make_splits(train, evaluation, fold, min_site_count=5):
    counts = train.loc[train.site_idx >= 0, "site_idx"].value_counts()
    sites = sorted(int(s) for s in counts[counts >= min_site_count].index)
    if len(sites) < 2:
        raise ValueError("Fewer than two sufficiently represented training sites.")
    site_train = np.flatnonzero(train.site_idx.isin(sites))
    site_eval = np.flatnonzero(evaluation.site_idx.isin(sites))
    if not len(site_eval):
        raise ValueError("No eligible site evaluation patients.")
    rng = np.random.default_rng(2026 + fold)
    fit, val = [], []
    for site in sites:
        idx = np.flatnonzero(train.site_idx.to_numpy() == site)
        idx = rng.permutation(idx)
        n_val = min(len(idx) - 1, max(1, int(np.floor(len(idx) * 0.2 + 0.5))))
        val.extend(idx[:n_val])
        fit.extend(idx[n_val:])
    surv_fit, surv_val = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=2026 + fold,
        stratify=train.censorship.to_numpy(),
    )
    return {
        "site_classes": sites, "site_train": site_train, "site_eval": site_eval,
        "site_fit": np.sort(fit), "site_val": np.sort(val),
        "survival_fit": np.sort(surv_fit), "survival_val": np.sort(surv_val),
    }


def survival_nll_rows(hazards, labels, censorship):
    h = np.clip(np.asarray(hazards, dtype=float), 1e-7, 1 - 1e-7)
    log_survival = np.cumsum(np.log1p(-h), axis=1)
    before = np.column_stack([np.zeros(len(h)), log_survival])
    idx, y = np.arange(len(h)), np.asarray(labels, dtype=int)
    return -(1 - np.asarray(censorship)) * (before[idx, y] + np.log(h[idx, y])) - np.asarray(censorship) * before[idx, y + 1]


def intercept_hazards(labels, censorship):
    labels, censorship = np.asarray(labels), np.asarray(censorship)
    return np.array([np.sum((labels == j) & (censorship == 0)) / np.sum(labels >= j)
                     for j in range(4)]).clip(1e-7, 1 - 1e-7)


class SurvivalEvaluation:
    """Training-only IPCW weights, with reusable comparable-pair matrices."""

    def __init__(self, train, evaluation, cuts):
        self.cuts = np.asarray(cuts, dtype=float)
        if self.cuts.shape != (3,) or not np.isfinite(self.cuts).all() or not (np.diff(self.cuts) > 0).all():
            raise ValueError("Expected three increasing finite survival cut points.")
        self.train = train
        self.evaluation = evaluation
        self.censor = CensoringDistributionEstimator().fit(
            Surv.from_arrays(train.censorship.to_numpy() == 0, train.event_time.to_numpy()))
        self.cap = float(np.nextafter(train.event_time.max(), -np.inf))
        self.time = np.minimum(evaluation.event_time.to_numpy(dtype=float), self.cap)
        self.event = (evaluation.censorship.to_numpy() == 0) & (evaluation.event_time.to_numpy() <= self.cap)
        self.labels = np.searchsorted(self.cuts, self.time, side="right")
        self.admin_censored = int(np.sum(evaluation.event_time.to_numpy() > self.cap))
        self.tau = float(self.cuts[-1])
        self.pair_error = ""
        self.pair_weights = np.zeros((len(evaluation), len(evaluation)))
        try:
            if self.censor.predict_proba([self.tau])[0] <= 0:
                raise ValueError("Zero censoring survival at tau.")
            eligible = self.event & (self.time < self.tau)
            weights = np.zeros(len(evaluation))
            if eligible.any():
                g = self.censor.predict_proba(self.time[eligible])
                if (g <= 0).any():
                    raise ValueError("Zero censoring survival at an event time.")
                weights[eligible] = 1 / g ** 2
            # Events are comparable to later observations and same-time censoring.
            comparable = (self.time[:, None] < self.time[None, :]) | (
                (self.time[:, None] == self.time[None, :]) & ~self.event[None, :])
            self.pair_weights = comparable * weights[:, None]
            if self.pair_weights.sum() == 0:
                self.pair_error = "no_comparable_pairs_before_tau"
        except ValueError as exc:
            self.pair_error = str(exc)

    def numerator(self, risk):
        risk = np.asarray(risk)
        diff = risk[:, None] - risk[None, :]
        credit = (diff > 1e-8).astype(float) + 0.5 * (np.abs(diff) <= 1e-8)
        return self.pair_weights * credit

    def score(self, hazards, indices=None):
        survival = np.cumprod(1 - np.asarray(hazards, dtype=float), axis=1)
        risk = -survival.sum(axis=1)
        idx = np.arange(len(risk)) if indices is None else np.asarray(indices, dtype=int)
        if not len(idx):
            return {"n": 0, "ipcw_cindex": np.nan, "nll": np.nan, "status": "empty_cohort"}
        weights = self.pair_weights[np.ix_(idx, idx)]
        numerator = self.numerator(risk)[np.ix_(idx, idx)]
        result = {
            "n": len(idx), "events": int(self.event[idx].sum()),
            "ipcw_cindex": float(numerator.sum() / weights.sum()) if weights.sum() > 0 else np.nan,
            "nll": float(survival_nll_rows(hazards, self.labels, ~self.event)[idx].mean()),
            "tau": self.tau, "administrative_cap": self.cap,
            "admin_censored_n": int(np.sum(self.evaluation.event_time.to_numpy()[idx] > self.cap)),
            "comparable_pairs": int(np.count_nonzero(weights)),
            "cindex_status": self.pair_error or ("ok" if weights.sum() > 0 else "no_comparable_pairs"),
        }
        time, event = self.time[idx], self.event[idx]
        for j, endpoint in enumerate(self.cuts):
            name = f"brier_{j + 1}"
            result[f"brier_time_{j + 1}"] = endpoint
            try:
                if not (time.min() <= endpoint < time.max()) or endpoint >= self.cap:
                    raise ValueError("endpoint_outside_followup_support")
                cases, controls = (time <= endpoint) & event, time > endpoint
                g_t = self.censor.predict_proba([endpoint])[0]
                g_case = self.censor.predict_proba(time[cases]) if cases.any() else np.array([])
                if g_t <= 0 or (g_case <= 0).any():
                    raise ValueError("zero_censoring_probability")
                terms = np.zeros(len(idx))
                estimate = survival[idx, j]
                terms[cases] = estimate[cases] ** 2 / g_case
                terms[controls] = (1 - estimate[controls]) ** 2 / g_t
                result[name] = float(terms.mean())
                result[name + "_status"] = "ok"
            except ValueError as exc:
                result[name] = np.nan
                result[name + "_status"] = str(exc)
        return result


def build_head(input_dim, outputs, family, config):
    if family == "linear":
        return nn.Linear(input_dim, outputs)
    return nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.ReLU(),
                         nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, outputs))


def fit_neural(x, y, task, family, outputs, config, decay, seed,
               validation=None, epochs=None):
    torch.manual_seed(seed)
    model = build_head(x.shape[1], outputs, family, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=decay)
    features = torch.as_tensor(np.asarray(x, dtype=np.float32))
    target = torch.as_tensor(y, dtype=torch.long if task == "site" else torch.float32)
    if task == "site":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=outputs)
        if (counts == 0).any():
            raise ValueError("All training classes must occur in probe fit data.")
        weight = torch.as_tensor(len(y) / (outputs * counts), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_value, best_epoch, stale = -np.inf, 1, 0
    history = []
    for epoch in range(1, (epochs or config.max_epochs) + 1):
        model.train()
        for batch in torch.randperm(len(features), generator=generator).split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[batch])
            if task == "site":
                loss = nn.functional.cross_entropy(logits, target[batch], weight=weight)
            else:
                hazards = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
                loss = nll_loss(hazards, None, target[batch, 0].long(), target[batch, 1], alpha=0)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite probe training loss.")
            loss.backward()
            optimizer.step()
        if validation is not None:
            val_x, val_y = validation
            pred = predict_neural(model, val_x, task)
            value = (site_scores(val_y, pred.argmax(axis=1))["balanced_accuracy"] if task == "site"
                     else -survival_nll_rows(pred, val_y[:, 0], val_y[:, 1]).mean())
            history.append({"epoch": epoch, "selection_value": float(value)})
            if value > best_value + 1e-12:
                best_value, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= config.patience:
                break
    return model.eval(), {"epoch": best_epoch if validation is not None else epochs,
                          "selection_value": float(best_value), "history": history}


def predict_neural(model, x, task):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
        return (logits.softmax(dim=-1) if task == "site" else logits.sigmoid().clamp(1e-7, 1 - 1e-7)).numpy()


def run_fold_probes(train, evaluation, metadata, output_dir, config, log=print):
    """Tune on inner data, refit on outer training data, evaluate without tuning."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "complete.json").unlink(missing_ok=True)
    train, evaluation = train.reset_index(drop=True), evaluation.reset_index(drop=True)
    validate_frames(train, evaluation)
    fold = int(metadata["fold"])
    splits = make_splits(train, evaluation, fold, config.min_site_count)
    classes = splits["site_classes"]
    encode_site = {s: i for i, s in enumerate(classes)}
    evaluator = SurvivalEvaluation(train, evaluation, metadata["finite_cuts"])
    for frame in (train, evaluation):
        expected_labels = np.searchsorted(evaluator.cuts, frame.event_time.to_numpy(), side="right")
        if not np.array_equal(expected_labels, frame.label.to_numpy(dtype=int)):
            raise ValueError("Survival labels do not match this fold's training-only cut points.")
    split_log = {k: train.case_id.iloc[v].tolist() for k, v in splits.items()
                 if k not in ("site_classes", "site_eval")}
    split_log.update(site_classes=classes, site_eval=evaluation.case_id.iloc[splits["site_eval"]].tolist(),
                     survival_eval=evaluation.case_id.tolist(), inner_split_seed=2026 + fold)
    write_json(out / "patient_splits.json", split_log)
    exclusions = evaluation[["case_id", "site_idx"]].copy()
    train_counts = train.site_idx.value_counts()
    exclusions["training_site_n"] = exclusions.site_idx.map(train_counts).fillna(0).astype(int)
    exclusions["site_eligible"] = exclusions.site_idx.isin(classes)
    exclusions["reason"] = np.where(exclusions.site_eligible, "included", np.where(
        exclusions.training_site_n == 0, "unseen_or_unknown_site", "training_site_below_threshold"))
    exclusions.to_csv(out / "site_coverage.csv", index=False)
    metrics, predictions, permutations, recalls, selections = [], [], [], [], []
    artifacts = out / "models"
    artifacts.mkdir(exist_ok=True)
    for task in ("site", "survival"):
        fit_idx, val_idx = splits[task + "_fit"], splits[task + "_val"]
        train_idx = splits["site_train"] if task == "site" else np.arange(len(train))
        eval_idx = splits["site_eval"] if task == "site" else np.arange(len(evaluation))
        target = (train.site_idx.map(encode_site).fillna(-1).to_numpy(dtype=int) if task == "site"
                  else train[["label", "censorship"]].to_numpy(dtype=np.float32))
        eval_y = (evaluation.site_idx.iloc[eval_idx].map(encode_site).to_numpy(dtype=int) if task == "site" else None)
        outputs = len(classes) if task == "site" else 4
        for rep in REPRESENTATIONS:
            columns = _vector_columns(train, rep)
            x, ex = train[columns].to_numpy(dtype=np.float32), evaluation[columns].to_numpy(dtype=np.float32)
            inner_scaler = StandardScaler().fit(x[fit_idx])
            final_scaler = StandardScaler().fit(x[train_idx])
            fit_x, val_x = inner_scaler.transform(x[fit_idx]), inner_scaler.transform(x[val_idx])
            full_x, eval_x = final_scaler.transform(x[train_idx]), final_scaler.transform(ex[eval_idx])
            for family in ("linear", "mlp"):
                stem = f"{task}_{family}_{rep}"
                log(f"[fold {fold}] {stem}: selection / refit", flush=True)
                candidates = []
                logistic = task == "site" and family == "linear"
                for strength in (config.c_grid if logistic else config.weight_decay_grid):
                    if logistic:
                        head = LogisticRegression(C=strength, class_weight="balanced", max_iter=5000,
                                                  solver="lbfgs", random_state=0).fit(fit_x, target[fit_idx])
                        value = site_scores(target[val_idx], head.predict(val_x))["balanced_accuracy"]
                        detail = {"C": strength, "selection_value": value, "epoch": None}
                    else:
                        _, detail = fit_neural(fit_x, target[fit_idx], task, family, outputs, config,
                                               strength, 0, validation=(val_x, target[val_idx]))
                        detail["weight_decay"] = strength
                    candidates.append(detail)
                selected = sorted(candidates, key=lambda d: (-d["selection_value"],
                                  d["C"] if logistic else -d["weight_decay"], d["epoch"] or 0))[0]
                selections.append({"task": task, "family": family, "representation": rep,
                                   "selected": selected, "candidates": candidates})
                np.savez_compressed(artifacts / (stem + "_scaler.npz"), mean=final_scaler.mean_,
                                    scale=final_scaler.scale_, classes=np.asarray(classes) if task == "site" else [])
                for seed in ((0,) if logistic else config.seeds):
                    if logistic:
                        head = LogisticRegression(C=selected["C"], class_weight="balanced", max_iter=5000,
                                                  solver="lbfgs", random_state=seed).fit(full_x, target[train_idx])
                        prob = head.predict_proba(eval_x)
                        np.savez_compressed(artifacts / (stem + f"_seed{seed}.npz"),
                                            coef=head.coef_, intercept=head.intercept_, classes=head.classes_)
                    else:
                        head, _ = fit_neural(full_x, target[train_idx], task, family, outputs, config,
                                             selected["weight_decay"], seed, epochs=selected["epoch"])
                        prob = predict_neural(head, eval_x, task)
                        torch.save({"state_dict": head.state_dict(), "family": family, "outputs": outputs,
                                    "config": asdict(config)}, artifacts / (stem + f"_seed{seed}.pt"))
                    ident = {"fold": fold, "task": task, "family": family, "representation": rep, "seed": seed}
                    if task == "site":
                        pred = prob.argmax(axis=1)
                        metrics.append({**ident, "cohort": "site_eligible", "n": len(eval_idx),
                                        "train_n": len(train_idx), "train_site_classes": outputs,
                                        "eval_site_classes": len(np.unique(eval_y)), "uniform_chance": 1 / outputs,
                                        **site_scores(eval_y, pred)})
                        for c in np.unique(eval_y):
                            recalls.append({**ident, "site_idx": classes[c], "n": int(np.sum(eval_y == c)),
                                            "recall": float(np.mean(pred[eval_y == c] == c))})
                    else:
                        for cohort, indices in (("all", None), ("site_eligible", splits["site_eval"])):
                            metrics.append({**ident, "cohort": cohort, **evaluator.score(prob, indices)})
                    for j, i in enumerate(eval_idx):
                        row = {**ident, **evaluation.iloc[i][["case_id", "site_idx", "event_time", "censorship", "label"]].to_dict()}
                        if task == "site":
                            row.update(predicted_site=classes[int(prob[j].argmax())])
                            row.update({f"site_prob_{s}": float(prob[j, c]) for c, s in enumerate(classes)})
                        else:
                            row.update({f"hazard_{k}": float(prob[j, k]) for k in range(4)})
                            row.update(risk=float(-np.cumprod(1 - np.asarray(prob[j], dtype=float)).sum()),
                                       evaluation_time=float(evaluator.time[i]), evaluation_event=int(evaluator.event[i]))
                        predictions.append(row)
                if task == "site":
                    for repeat in range(config.permutations):
                        shuffled = np.random.default_rng(30000 + 100 * fold + repeat).permutation(target[train_idx])
                        if logistic:
                            head = LogisticRegression(C=selected["C"], class_weight="balanced", max_iter=5000,
                                                      solver="lbfgs", random_state=0).fit(full_x, shuffled)
                            pred = head.predict(eval_x)
                        else:
                            head, _ = fit_neural(full_x, shuffled, task, family, outputs, config,
                                                 selected["weight_decay"], 0, epochs=selected["epoch"])
                            pred = predict_neural(head, eval_x, task).argmax(axis=1)
                        permutations.append({"fold": fold, "family": family, "representation": rep,
                                             "repeat": repeat, **site_scores(eval_y, pred)})
                        if (repeat + 1) % 5 == 0:
                            log(f"[fold {fold}] {stem}: label permutations {repeat + 1}/{config.permutations}", flush=True)
    constant = np.tile(intercept_hazards(train.label, train.censorship), (len(evaluation), 1))
    for cohort, indices in (("all", None), ("site_eligible", splits["site_eval"])):
        metrics.append({"fold": fold, "task": "survival", "family": "intercept", "representation": "intercept",
                        "seed": 0, "cohort": cohort, **evaluator.score(constant, indices)})
    pd.DataFrame(predictions).to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(permutations, columns=["fold", "family", "representation", "repeat", "balanced_accuracy", "macro_f1"]).to_csv(out / "permutation_controls.csv", index=False)
    pd.DataFrame(recalls).to_csv(out / "site_recall.csv", index=False)
    write_json(out / "selection.json", selections)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    write_json(out / "complete.json", {"fold": fold, "config": asdict(config), "status": "complete"})
    return pd.DataFrame(metrics)
