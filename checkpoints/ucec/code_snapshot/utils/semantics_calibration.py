"""Prespecified finite-sample checks; failed checks disable real-data GCM claims."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.special import softmax
from scipy.stats import beta

from utils.semantics_stats import seed_for
from utils.semantics_nuisance import patient_association

PROFILES = ((300, 75, .45), (400, 100, .15), (600, 150, .15))
SCENARIOS = ("site_only", "nonlinear_prototype", "near_deterministic_prototype",
             "residual_dependent_censoring", "linear_patient", "nonlinear_patient")


def inner_site_split(site, seed=2026):
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for level in np.unique(site):
        ids = rng.permutation(np.flatnonzero(site == level))
        if len(ids) < 2:
            raise ValueError("A site cannot be split into fitting and internal validation patients")
        n = min(len(ids)-1, max(1, int(np.floor(.2*len(ids)+.5))))
        validation.extend(ids[:n])
        fit.extend(ids[n:])
    return np.sort(fit), np.sort(validation)


def simulate(profile, scenario, seed):
    """Fixed DGP: 8 imbalanced sites, 4 continuous soft prototypes, 256D r.

    T and C use independent exponential innovations. C depends on patient q
    only in scenario 4. Target event fractions set the censoring rate using
    event/censoring hazards, without inspecting realized outcomes.
    """
    nt, ne, target = PROFILES[profile]
    n = nt+ne
    rng = np.random.default_rng(seed)
    probabilities = np.array([.30, .20, .14, .11, .08, .065, .055, .05])
    site = rng.choice(8, size=n, p=probabilities)
    # Ensure the shared site's training >=5 definition is met in all simulations.
    # This changes only site sampling, before a, representation and outcomes exist.
    while np.bincount(site[:nt], minlength=8).min() < 5:
        site[:nt] = rng.choice(8, size=nt, p=probabilities)
    z = rng.normal(size=(n, 4)) + .35*np.sin(site[:, None]+np.arange(4))
    a = softmax(.7*z, axis=1)
    q = rng.normal(size=n)
    group = .35*(site-3.5)
    if scenario != 0:
        group = group + 1.2*np.sin(7*a[:, 0]) + 1.5*(a[:, 1]-.3)**2 - .8*np.cos(5*a[:, 2])
    patient = np.zeros(n)
    if scenario == 4:
        patient = .9*q
    elif scenario == 5:
        patient = .85*(q*q-1)
    basis = np.column_stack([a, a*a, np.sin(7*a), np.cos(5*a), np.eye(8)[site]])
    projection = rng.normal(size=(basis.shape[1], 256)) / np.sqrt(basis.shape[1])
    noise = .025 if scenario == 2 else .5
    representation = basis @ projection + noise*rng.normal(size=(n, 256))
    if scenario != 2:
        # A fixed subset encodes q, including under the null: no supervised map.
        representation[:, :16] += q[:, None]
    event_rate = .04*np.exp(np.clip(group+patient, -5, 5))
    censor_relative = np.exp(.10*(site-3.5) + (1.0*q if scenario == 3 else 0))
    lo, hi = 1e-8, 1e3
    for _ in range(60):
        middle = np.sqrt(lo*hi)
        expected = np.mean(event_rate/(event_rate+middle*censor_relative))
        if expected > target:
            lo = middle
        else:
            hi = middle
    censor_rate = np.sqrt(lo*hi)*censor_relative
    event_time = rng.exponential(1/event_rate)
    censor_time = rng.exponential(1/censor_rate)
    observed = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    data = {"site": site, "a": a, "time": observed, "event": event}
    train = {key: value[:nt] for key, value in data.items()}
    evaluation = {key: value[nt:] for key, value in data.items()}
    train["representations"] = {"synthetic": representation[:nt]}
    evaluation["representations"] = {"synthetic": representation[nt:]}
    cuts = np.quantile(observed[:nt][event[:nt]], [.25, .5, .75])
    fit, validation = inner_site_split(site[:nt])
    return train, evaluation, cuts, fit, validation


def calibration_job(arguments):
    from threadpoolctl import threadpool_limits
    from utils.independent_probes import write_json
    output, profile, scenario, replicate, config = arguments
    path = Path(output)/"calibration"/"replicates"/f"p{profile}_s{scenario}_r{replicate:04d}.json"
    seed = seed_for(config.seed, 901, profile, scenario, replicate)
    signature = {"config": json.loads(json.dumps(asdict(config))), "profile": profile,
                 "scenario": scenario, "replicate": replicate, "seed": seed}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError(f"Calibration resume configuration mismatch: {path}")
        if saved["status"] == "complete":
            return saved
    start = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            train, evaluation, cuts, fit, validation = simulate(profile, scenario, seed)
            result = patient_association(train, evaluation, fit, validation, cuts, config,
                                         seed, draws=config.calibration_draws)
            r = result["representations"]["synthetic"]
            record = {"signature": signature, "status": "complete", "stat": r["stat"],
                      "survival_diagnostic": result["survival_diagnostic"],
                      "feature_diagnostic": r["diagnostic"], "cuts": cuts,
                      "train_event_fraction": float(train["event"].mean()),
                      "eval_event_fraction": float(result["event_after_admin"].mean())}
    except Exception as exc:
        record = {"signature": signature, "status": "failed", "error": repr(exc),
                  "traceback": traceback.format_exc()}
    record["seconds"] = time.monotonic()-start
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return record


def binomial_interval(successes, total, confidence=.99):
    if total == 0:
        return np.nan, np.nan
    alpha = (1-confidence)/2
    return (0. if successes == 0 else float(beta.ppf(alpha, successes, total-successes+1)),
            1. if successes == total else float(beta.ppf(1-alpha, successes+1, total-successes)))


def summarize_calibration(output, config):
    import pandas as pd
    from utils.independent_probes import write_json
    folder = Path(output)/"calibration"
    rows, all_records, incomplete = [], [], False
    for profile, size in enumerate(PROFILES):
        for scenario, name in enumerate(SCENARIOS):
            records = []
            errors = []
            for replicate in range(config.calibration_reps):
                path = folder/"replicates"/f"p{profile}_s{scenario}_r{replicate:04d}.json"
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record["signature"]["config"] != json.loads(json.dumps(asdict(config))):
                        raise ValueError("Calibration record configuration mismatch")
                    if record["status"] == "complete":
                        records.append(record)
                        all_records.append({"profile": profile, "scenario": name,
                                            "replicate": replicate, **record["stat"]})
                    else:
                        errors.append(record.get("error"))
            n = len(records)
            rejected = sum(r["stat"]["p"] <= .05 for r in records)
            low, high = binomial_interval(rejected, n)
            incomplete |= n != config.calibration_reps
            rows.append({"profile": profile, "n_train": size[0], "n_eval": size[1],
                         "target_event_fraction": size[2], "scenario": name, "null": scenario < 4,
                         "completed": n, "requested": config.calibration_reps, "failures": len(errors),
                         "rejected": rejected, "rejection_rate": rejected/n if n else np.nan,
                         "ci99_low": low, "ci99_high": high,
                         "inflated": bool(scenario < 4 and n == config.calibration_reps and low > .05),
                         "errors": " | ".join(sorted(set(errors)))})
    table = pd.DataFrame(rows)
    inflated = table[table.inflated]
    qualified = config.calibration_reps >= 200 and config.calibration_draws >= 999 and config.trees == 200
    status = ("failed" if len(inflated) else "incomplete" if incomplete else
              "passed" if qualified else "smoke_only")
    summary = {"status": status, "allow_real_inference": status == "passed", "config": asdict(config),
               "completed": len(all_records), "required": 18*config.calibration_reps,
               "gate": "For any of 12 null profile/scenario cells, lower exact 99% binomial CI > 0.05",
               "inflated_cells": inflated.to_dict("records"),
               "interpretation": "Finite-sample simulation check, not a universal validity guarantee"}
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder/"calibration_summary.csv", index=False)
    pd.DataFrame(all_records).to_csv(folder/"calibration_statistics.csv", index=False)
    write_json(folder/"calibration_status.json", summary)
    return summary
