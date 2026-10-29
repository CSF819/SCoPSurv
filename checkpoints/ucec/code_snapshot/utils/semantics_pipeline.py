"""Source-checked fold stages for the soft-prototype semantics experiment."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.independent_probes import write_json, site_scores
from utils.probe_v2 import load_reference, identity, read_json, site_representation_folder
from utils.tri_factor_probe import _vector_columns
from utils.semantics_stats import (
    REPRESENTATIONS, seed_for, permanova_many, site_permutations, p_upper,
    bootstrap_counts, administrative_censor, MarginalCensoring, brier_rows,
    prototype_predictions, partial_explained,
)
from utils.semantics_nuisance import fit_forest, patient_association

ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    return json.loads(json.dumps(value))


def clinical_lookup(path):
    frame = pd.read_csv(path, usecols=["case_id", "oncotree_code"], dtype=str)
    # Missing vs observed subtype also counts as inconsistent duplicated metadata.
    conflict = frame.groupby("case_id", dropna=False).oncotree_code.nunique(dropna=False)
    if (conflict > 1).any():
        raise ValueError(f"Conflicting duplicated clinical subtype: {conflict[conflict > 1].index.tolist()}")
    return frame.drop_duplicates("case_id").set_index("case_id").oncotree_code


def prepare(reference, output, dataset, fold, config):
    train_frame, eval_frame, meta, split, provenance = load_reference(reference, dataset, fold)
    out = Path(output)/dataset/f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    protocol = normalized({"config": asdict(config), "provenance": provenance,
                           "preparation_code": identity(Path(__file__))})
    marker = out/"prepared.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["protocol"] != protocol:
            raise ValueError("Prepared data configuration or original source changed")
        for name, record in saved["artifacts"].items():
            if identity(out/name) != record:
                raise ValueError(f"Prepared artifact changed: {name}")
        return joblib.load(out/"prepared.joblib"), out
    import torch
    import torch.nn.functional as F
    from utils.probe_checkpoints import load_probe_model
    model, _, restored_fold, loading = load_probe_model(meta["checkpoint"], "cpu")
    if restored_fold != fold or model.prog_prototypes.shape != (4, 256):
        raise ValueError("Prototype dimensions or checkpoint fold mismatch")
    clinical = clinical_lookup(meta["source"]["clinical_csv"]["path"])
    datasets, rows, exclusions, objects = {}, [], [], {}
    for partition, frame, indices in (("train", train_frame, split["site_train"]),
                                       ("eval", eval_frame, split["site_eval"])):
        selected = frame.iloc[indices].reset_index(drop=True)
        features = {r: selected[_vector_columns(selected, r)].to_numpy(float) for r in REPRESENTATIONS}
        if any(x.shape[1] != 256 for x in features.values()):
            raise ValueError("Exact representation column matching did not yield 256 columns")
        zs = torch.tensor(features["z_s"], dtype=torch.float32)
        with torch.no_grad():
            assignments, logits = model._prototype_assign(zs)
            repeated, _ = model._prototype_assign(zs)
            expected = torch.softmax(F.normalize(zs, dim=-1) @ F.normalize(model.prog_prototypes, dim=-1).T /
                                     max(float(model.proto_temperature), 1e-6), dim=-1)
        if not torch.equal(assignments, repeated) or not torch.equal(assignments, expected):
            raise ValueError("Soft assignments do not reproduce the loaded checkpoint module")
        raw_a = assignments.cpu().numpy().astype(float)
        a = raw_a.copy()
        if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6):
            raise ValueError("Invalid checkpoint soft assignment")
        # Correct float32 sum-to-one rounding before convex survival mixtures.
        # Both original module outputs and float64 analysis weights are saved.
        a /= a.sum(1, keepdims=True)
        subtypes = selected.case_id.map(clinical)
        if subtypes.isna().any():
            raise ValueError("Missing clinical metadata for cached patient")
        datasets[partition] = {"case_id": selected.case_id.to_numpy(), "site": selected.site_idx.to_numpy(),
                               "site_name": selected.site.to_numpy(), "a": a,
                               "time": selected.event_time.to_numpy(float),
                               "event": (1-selected.censorship.to_numpy(int)).astype(bool),
                               "clinical": subtypes.to_numpy(str), "representations": features}
        listing = selected[["case_id", "site", "site_idx", "event_time", "censorship", "fold", "split"]].copy()
        listing["oncotree_code"] = subtypes
        for j in range(4):
            listing[f"a_{j}"] = a[:, j]
            listing[f"checkpoint_a_{j}"] = raw_a[:, j]
        rows.append(listing)
        excluded = frame.loc[~frame.index.isin(indices), ["case_id", "site", "site_idx"]].copy()
        excluded["partition"] = partition
        excluded["reason"] = "original_training_site_count_below_5"
        exclusions.append(excluded)
    train_ids = datasets["train"]["case_id"]
    lookup = {str(pid): i for i, pid in enumerate(train_ids)}
    inner = {key: np.array([lookup[str(pid)] for pid in train_frame.iloc[split[key]].case_id], int)
             for key in ("site_fit", "site_val")}
    for r in REPRESENTATIONS:
        objects[r] = StandardScaler().fit(datasets["train"]["representations"][r])
    objects["a"] = StandardScaler().fit(datasets["train"]["a"])
    prepared = {**datasets, "inner_fit": inner["site_fit"], "inner_val": inner["site_val"],
                "cuts": np.asarray(meta["finite_cuts"], float), "scalers": objects,
                "dataset": dataset, "fold": fold, "provenance": provenance,
                "coverage": {"original_train": len(train_frame), "original_eval": len(eval_frame),
                             "train": len(train_ids), "eval": len(datasets["eval"]["case_id"]),
                             "train_sites": len(np.unique(datasets["train"]["site"])),
                             "eval_sites": len(np.unique(datasets["eval"]["site"]))}}
    joblib.dump(prepared, out/"prepared.joblib", compress=2)
    joblib.dump(objects, out/"preprocessing.joblib")
    pd.concat(rows, ignore_index=True).to_csv(out/"patients_and_assignments.csv.gz", index=False)
    pd.concat(exclusions, ignore_index=True).to_csv(out/"excluded_patients.csv", index=False)
    write_json(out/"patient_splits.json", {"fit": train_ids[inner["site_fit"]],
                                          "inner_validation": train_ids[inner["site_val"]],
                                          "original_train": train_ids,
                                          "original_validation": datasets["eval"]["case_id"],
                                          "inner_seed": 2026+fold})
    write_json(out/"prototype_source.json", {"checkpoint": provenance["checkpoint"], "loading": loading,
                                            "temperature": float(model.proto_temperature),
                                            "prototype_shape": list(model.prog_prototypes.shape),
                                            "module_output_reproduced": True, "repeat_equal": True,
                                            "main_model_frozen": not any(p.requires_grad for p in model.parameters()),
                                            "forward_scope": "_prototype_assign(z_s) only; no slide/encoder inference"})
    write_json(marker, {"protocol": protocol, "artifacts": {name: identity(out/name) for name in
                            ("prepared.joblib", "preprocessing.joblib", "patients_and_assignments.csv.gz",
                             "patient_splits.json", "excluded_patients.csv", "prototype_source.json")}})
    return prepared, out


def supplementary_site(data, out):
    """Copy only existing probe/HSIC results whose sources and patients match."""
    provenance, dataset, fold = data["provenance"], data["dataset"], data["fold"]
    evaluation = data["eval"]
    expected = evaluation["case_id"]
    truth = evaluation["site"]
    records, status = [], []
    v2 = ROOT/"analysis_outputs/experiment1_motivation_v2"/dataset/f"fold_{fold}"
    try:
        if read_json(v2/"prepared.json")["provenance"] != provenance:
            raise ValueError("Existing site probe source mismatch")
        for rep in REPRESENTATIONS:
            parent = site_representation_folder(v2, rep)
            for unit in (parent/"ridge_n5", parent/"native_supplement"):
                if read_json(unit/"complete.json")["status"] != "complete":
                    raise ValueError(f"Incomplete existing site probe: {unit}")
                pred = pd.read_csv(unit/"predictions.csv.gz", float_precision="round_trip")
                pred = pred[pred.partition == "eval"]
                expected_units = {("ridge", 0)} if unit.name == "ridge_n5" else {
                    ("linear", 0), ("mlp", 0), ("mlp", 1), ("mlp", 2)}
                if set(zip(pred.family, pred.seed)) != expected_units:
                    raise ValueError("Supplementary site probe families or seeds are incomplete")
                for (family, seed), group in pred.groupby(["family", "seed"]):
                    if group.case_id.duplicated().any() or set(group.case_id) != set(expected):
                        raise ValueError("Supplementary site probe evaluation patient mismatch")
                    group = group.set_index("case_id").loc[expected]
                    if not np.array_equal(group.site_idx, truth):
                        raise ValueError("Supplementary site probe label mismatch")
                    score = site_scores(truth, group.predicted_site.to_numpy())
                    records.append({"representation": rep, "family": family, "seed": seed, **score,
                                    "source": str(unit), "source_prediction": identity(unit/"predictions.csv.gz")})
        status.append({"kind": "site_probes", "status": "verified", "source": str(v2)})
    except (ValueError, KeyError, FileNotFoundError) as exc:
        records = []
        status.append({"kind": "site_probes", "status": "unavailable", "reason": str(exc)})
    hsic = ROOT/"analysis_outputs/site_dependence_zb"/dataset/f"fold_{fold}"
    try:
        if read_json(hsic/"sources.json") != provenance:
            raise ValueError("Existing HSIC source mismatch")
        patients = pd.read_csv(hsic/"patients.csv").set_index("case_id")
        if set(patients.index) != set(expected) or not np.array_equal(patients.loc[expected].site_idx, truth):
            raise ValueError("Existing HSIC evaluation patient mismatch")
        training_patients = pd.read_csv(hsic/"training_patients.csv").set_index("case_id")
        if set(training_patients.index) != set(data["train"]["case_id"]) or not np.array_equal(
                training_patients.loc[data["train"]["case_id"]].site_idx, data["train"]["site"]):
            raise ValueError("Existing HSIC training patient mismatch")
        write_json(out/"supplement_hsic.json", read_json(hsic/"metrics.json"))
        status.append({"kind": "zb_hsic", "status": "verified", "source": identity(hsic/"metrics.json")})
    except (ValueError, KeyError, FileNotFoundError) as exc:
        status.append({"kind": "zb_hsic", "status": "unavailable", "reason": str(exc)})
    write_json(out/"supplement_sources.json", status)
    write_json(out/"supplement_site_probes.json", records)


def run_site(data, out, config):
    evaluation = data["eval"]
    x = np.stack([data["scalers"][r].transform(evaluation["representations"][r]) for r in REPRESENTATIONS], axis=1)
    labels = evaluation["site"]
    seed = seed_for(config.seed, 101, data["fold"])
    observed = permanova_many(x, labels)[0]
    null = site_permutations(x, labels, config.permutations, seed)
    counts = bootstrap_counts(labels, config.bootstrap, seed_for(seed, 1))
    samples = permanova_many(x, labels, counts)
    rows, dispersion = [], []
    for j, rep in enumerate(REPRESENTATIONS):
        rows.append({"representation": rep, "r2": observed[j, 0], "adjusted_r2": observed[j, 1],
                     "pseudo_f": observed[j, 2], "p": p_upper(observed[j, 2], null[:, j]),
                     **data["coverage"]})
        for site in np.unique(labels):
            values = x[labels == site, j]
            dispersion.append({"representation": rep, "site_idx": int(site), "n": len(values),
                               "covariance_trace": float(np.sum((values-values.mean(0))**2)/(len(values)-1))
                               if len(values) > 1 else np.nan,
                               "reason": "singleton_site" if len(values) == 1 else ""})
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(dispersion).to_csv(out/"site_dispersion.csv", index=False)
    np.savez_compressed(out/"resampling.npz", null=null, bootstrap=samples, counts=counts)
    supplementary_site(data, out)


def prototype_structure(data, out):
    summary, patients = [], []
    for partition in ("train", "eval"):
        a = data[partition]["a"]
        entropy = -np.sum(np.where(a > 0, a*np.log(np.maximum(a, 1e-300)), 0), 1)
        mean = a.mean(0)
        entropy_mean = -np.sum(np.where(mean > 0, mean*np.log(np.maximum(mean, 1e-300)), 0))
        hard = np.bincount(a.argmax(1), minlength=4)
        for k in range(4):
            summary.append({"partition": partition, "prototype": k, "n": len(a), "mean_weight": mean[k],
                            "hard_n": hard[k], "ess": a[:, k].sum()**2 / np.sum(a[:, k]**2),
                            "mean_patient_entropy": entropy.mean(), "mean_assignment_entropy": entropy_mean,
                            "assignment_information": max(0., entropy_mean-entropy.mean()),
                            "weight_min": a[:, k].min(), "weight_q25": np.quantile(a[:, k], .25),
                            "weight_median": np.median(a[:, k]), "weight_q75": np.quantile(a[:, k], .75),
                            "weight_max": a[:, k].max(), "weight_sd": a[:, k].std(ddof=1)})
        for i, case in enumerate(data[partition]["case_id"]):
            patients.append({"partition": partition, "case_id": case, "entropy": entropy[i],
                             "hard_prototype": int(a[i].argmax())})
    pd.DataFrame(summary).to_csv(out/"prototype_structure.csv", index=False)
    pd.DataFrame(patients).to_csv(out/"patient_assignment_entropy.csv", index=False)


def clinical_analysis(data, out, config):
    dataset = data["dataset"]
    if dataset not in ("gbmlgg", "ucec"):
        write_json(out/"clinical_status.json", {"status": "not_applicable", "reason": "single clinical subtype"})
        return
    evaluation = data["eval"]
    keep = np.ones(len(evaluation["site"]), bool) if dataset == "gbmlgg" else np.isin(evaluation["clinical"], ["UEC", "USC"])
    train_keep = np.ones(len(data["train"]["site"]), bool) if dataset == "gbmlgg" else np.isin(data["train"]["clinical"], ["UEC", "USC"])
    site, subtype = evaluation["site"][keep], evaluation["clinical"][keep]
    pd.concat([pd.DataFrame({"case_id": part["case_id"], "included": mask, "partition": partition,
                             "oncotree_code": part["clinical"],
                             "reason": np.where(mask, "included", "unspecified_UCEC")})
               for partition, part, mask in (("train", data["train"], train_keep), ("eval", evaluation, keep))],
              ignore_index=True).to_csv(out/"clinical_patients.csv", index=False)
    scalers = {r: StandardScaler().fit(data["train"]["representations"][r][train_keep]) for r in REPRESENTATIONS}
    scalers["a"] = StandardScaler().fit(data["train"]["a"][train_keep])
    joblib.dump(scalers, out/"clinical_preprocessing.joblib")
    arrays = [scalers[r].transform(evaluation["representations"][r])[keep] for r in REPRESENTATIONS]
    arrays += [scalers["a"].transform(evaluation["a"])[keep]]
    keys = [*REPRESENTATIONS, "a"]
    combined = np.concatenate(arrays, axis=1)
    boundaries = np.cumsum([0]+[x.shape[1] for x in arrays])
    site_design = np.equal.outer(site, np.unique(site)).astype(float)
    residual = combined - site_design @ np.linalg.lstsq(site_design, combined, rcond=None)[0]
    denominators = np.array([np.sum(x*x) for x in np.split(residual, boundaries[1:-1], axis=1)])
    levels = np.unique(subtype)
    def statistic(labels):
        extra = np.equal.outer(labels, levels).astype(float)
        extra -= site_design @ np.linalg.lstsq(site_design, extra, rcond=None)[0]
        u, singular, _ = np.linalg.svd(extra, full_matrices=False)
        rank = np.count_nonzero(singular > 1e-10)
        if rank == 0 or len(site) <= site_design.shape[1]+rank:
            return np.full(5, np.nan), rank
        projected = u[:, :rank].T @ residual
        numerator = np.array([np.sum(x*x) for x in np.split(projected, boundaries[1:-1], axis=1)])
        return np.divide(numerator, denominators, out=np.full(5, np.nan), where=denominators > 1e-14), rank
    observed, rank = statistic(subtype)
    rng = np.random.default_rng(seed_for(config.seed, 303, data["fold"]))
    null = np.empty((config.permutations, len(keys)))
    groups = [np.flatnonzero(site == s) for s in np.unique(site)]
    for draw in range(config.permutations):
        labels = subtype.copy()
        for ids in groups:
            labels[ids] = rng.permutation(labels[ids])
        null[draw], _ = statistic(labels)
    rows = [{"representation": key, "partial_r2": observed[j], "p": p_upper(observed[j], null[:, j]),
             "n": len(site), "sites": len(groups), "subtypes": len(levels), "additional_rank": rank}
            for j, key in enumerate(keys)]
    pd.DataFrame(rows).to_csv(out/"clinical_metrics.csv", index=False)
    np.savez_compressed(out/"clinical_null.npz", null=null)
    _, admin_event, _ = administrative_censor(data["train"]["time"], evaluation["time"], evaluation["event"])
    counts = bootstrap_counts(admin_event[keep], config.bootstrap, seed_for(config.seed, 304, data["fold"]))
    samples = np.full((config.bootstrap, len(keys)), np.nan)
    for draw, frequencies in enumerate(counts):
        indices = np.repeat(np.arange(len(site)), frequencies)
        ys, cs, xs = site[indices], subtype[indices], combined[indices]
        base = np.equal.outer(ys, np.unique(ys)).astype(float)
        extra = np.equal.outer(cs, np.unique(cs)).astype(float)
        xr = xs-base @ np.linalg.lstsq(base, xs, rcond=None)[0]
        extra -= base @ np.linalg.lstsq(base, extra, rcond=None)[0]
        u, singular, _ = np.linalg.svd(extra, full_matrices=False)
        local_rank = np.count_nonzero(singular > 1e-10)
        if local_rank == 0 or len(indices) <= base.shape[1]+local_rank:
            continue
        projection = u[:, :local_rank].T @ xr
        num = np.array([np.sum(v*v) for v in np.split(projection, boundaries[1:-1], axis=1)])
        den = np.array([np.sum(v*v) for v in np.split(xr, boundaries[1:-1], axis=1)])
        samples[draw] = np.divide(num, den, out=np.full(5, np.nan), where=den > 1e-14)
    np.savez_compressed(out/"clinical_bootstrap.npz", keys=np.array(keys), samples=samples)
    write_json(out/"clinical_status.json", {"status": "complete" if rank else "not_estimable",
                                             "reason": "" if rank else "subtype adds no estimable rank beyond site"})


def run_cohort(data, out, config):
    train, evaluation, cuts = data["train"], data["eval"], data["cuts"]
    prototype_structure(data, out)
    predictions, donors, errors = prototype_predictions(train["time"], train["event"], train["site"], train["a"],
                                                         evaluation["site"], evaluation["a"], cuts)
    joblib.dump(donors, out/"donor_weights_and_curves.joblib", compress=2)
    donor_rows, curve_rows = [], []
    for held_site, record in donors.items():
        ids = record["donor_indices"]
        for k, curve in enumerate(record["curves"]):
            donor_rows += [{"excluded_site_idx": held_site, "case_id": train["case_id"][idx],
                            "site_idx": train["site"][idx], "prototype": k, "weight": weight}
                           for idx, weight in zip(ids, record["weights"][:, k])]
            curve_rows += [{"excluded_site_idx": held_site, "prototype": k, "time": t,
                            "survival": s, "risk_weight": w, "event_weight": d,
                            "ess": record["ess"][k]} for t, s, w, d in zip(
                                curve["time"], curve["survival"], curve["risk_weight"], curve["event_weight"])]
    pd.DataFrame(donor_rows).to_csv(out/"donor_weights.csv.gz", index=False)
    pd.DataFrame(curve_rows).to_csv(out/"prototype_survival_curves.csv.gz", index=False)
    time, event, cap = administrative_censor(train["time"], evaluation["time"], evaluation["event"])
    marginal = MarginalCensoring().fit(train["time"], train["event"])
    joblib.dump(marginal, out/"marginal_censoring.joblib")
    gtime = marginal.predict(time)
    gcut = marginal.predict(cuts)
    # Fixed sensitivity model; it is not chosen using validation Brier results.
    conditional = None
    try:
        conditional = fit_forest(train["site"], train["a"], train["time"], ~train["event"],
                                 10, config, seed_for(config.seed, 401, data["fold"]))
        ctime = conditional.values(evaluation["site"], evaluation["a"], time)[:, 0]
        ccut = conditional.values(evaluation["site"], evaluation["a"], cuts[None, :])
        joblib.dump(conditional, out/"conditional_censoring.joblib", compress=2)
    except ValueError as exc:
        ctime, ccut = np.full(len(time), np.nan), np.full((len(time), 3), np.nan)
        errors.append({"stage": "conditional_censoring", "reason": str(exc)})
    permuted = []
    rng = np.random.default_rng(seed_for(config.seed, 402, data["fold"]))
    site_groups = [np.flatnonzero(train["site"] == s) for s in np.unique(train["site"])]
    permutation_indices = []
    for repeat in range(config.random_repeats):
        index = np.arange(len(train["site"]))
        for ids in site_groups:
            index[ids] = rng.permutation(ids)
        null_prediction, _, fail = prototype_predictions(train["time"], train["event"], train["site"], train["a"][index],
                                                           evaluation["site"], evaluation["a"], cuts)
        errors += [{"repeat": repeat, **failure} for failure in fail]
        permuted.append(null_prediction)
        permutation_indices.append(index)
    np.savez_compressed(out/"assignment_permutations.npz", indices=np.stack(permutation_indices),
                        proto=np.stack([p["proto"] for p in permuted]),
                        reference=np.stack([p["reference"] for p in permuted]))
    rows, patient_rows, boots, keys, excluded = [], [], [], [], []
    for j, cut in enumerate(cuts):
        endpoint_valid = bool(time.min() <= cut < time.max() and cut < cap)
        contributions = {}
        for model, (gt, gc) in {"marginal": (gtime, gcut[j]), "conditional": (ctime, ccut[:, j])}.items():
            for name, prediction in predictions.items():
                value, valid = brier_rows(time, event, prediction[:, j], cut, gt, gc)
                contributions[model, name] = value
        common = np.isfinite(np.stack([contributions["marginal", name] for name in predictions])).all(0)
        common &= endpoint_valid
        sensitivity_common = common & np.isfinite(np.stack([contributions["conditional", name] for name in predictions])).all(0)
        if conditional is None:
            sensitivity_common[:] = False
        for population, mask, models in (("primary", common, ("marginal",)),
                                          ("sensitivity_common", sensitivity_common, ("marginal", "conditional"))):
            count = bootstrap_counts(event[mask], config.bootstrap, seed_for(config.seed, 403, data["fold"], j,
                                                                             int(population != "primary"))) if mask.any() else None
            for model in models:
                for name in predictions:
                    values = contributions[model, name][mask]
                    estimate = float(values.mean()) if len(values) else np.nan
                    key = f"{population}|{model}|{name}|{j}"
                    rows.append({"population": population, "censoring_model": model, "prediction": name,
                                 "cut_index": j, "time": cut, "main_endpoint": j == 1, "n": int(mask.sum()),
                                 "brier": estimate, "reason": "" if len(values) else "no_common_estimable_patients"})
                    keys.append(key)
                    boots.append(count @ values/len(values) if len(values) else np.full(config.bootstrap, np.nan))
            excluded += [{"case_id": case, "cut_index": j, "population": population,
                          "reason": "endpoint_outside_evaluation_followup" if not endpoint_valid else
                          "donor_curve_or_required_censoring_weight_not_estimable"}
                         for case in evaluation["case_id"][~mask]]
        for i, case in enumerate(evaluation["case_id"]):
            patient_rows.append({"case_id": case, "site_idx": evaluation["site"][i], "cut_index": j,
                                 "cut": cut, "time_after_admin": time[i], "event_after_admin": int(event[i]),
                                 "admin_changed": bool(time[i] != evaluation["time"][i] or event[i] != evaluation["event"][i]),
                                 "primary_included": bool(common[i]), "sensitivity_included": bool(sensitivity_common[i]),
                                 "G_time_marginal": gtime[i], "G_cut_marginal": gcut[j],
                                 "G_time_conditional": ctime[i], "G_cut_conditional": ccut[i, j],
                                 **{f"S_{name}": predictions[name][i, j] for name in predictions},
                                 **{f"BS_{model}_{name}": contributions[model, name][i] for model, name in contributions}})
        # Negative controls evaluated on exactly the primary population whenever estimable.
        for repeat, prediction in enumerate(permuted):
            controls = {name: brier_rows(time, event, prediction[name][:, j], cut, gtime, gcut[j])[0]
                        for name in ("proto", "reference")}
            for name in controls:
                values = controls[name][common]
                rows.append({"population": "permutation_control", "censoring_model": "marginal", "prediction": name,
                             "cut_index": j, "time": cut, "main_endpoint": j == 1, "repeat": repeat,
                             "n": int(common.sum()), "brier": float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan,
                             "reason": "" if len(values) and np.isfinite(values).all() else "not_estimable_on_primary_patients"})
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(out/"patient_predictions.csv.gz", index=False)
    pd.DataFrame(excluded, columns=["case_id", "cut_index", "population", "reason"]).to_csv(out/"exclusions.csv", index=False)
    np.savez_compressed(out/"bootstrap.npz", keys=np.array(keys), samples=np.stack(boots))
    write_json(out/"estimation.json", {"cap": cap, "errors": errors,
                                       "conditional_censoring": {"model": "RandomSurvivalForest", "trees": config.trees,
                                                                  "depth": 5, "leaf": 10, "max_features": 1.,
                                                                  "selection": "fixed training-only sensitivity model"},
                                       "main_endpoint": float(cuts[1])})
    clinical_analysis(data, out, config)


def _subset(data, mask):
    return {key: ({r: x[mask] for r, x in value.items()} if key == "representations" else value[mask])
            for key, value in data.items()}


def run_patient(data, out, config, output):
    gate_file = Path(output)/"calibration/calibration_status.json"
    if not gate_file.exists():
        write_json(out/"inference_status.json", {"status": "blocked_calibration_missing", "allow_real_inference": False})
        return "blocked_calibration_missing"
    gate = read_json(gate_file)
    if gate["config"] != normalized(asdict(config)):
        raise ValueError("Calibration configuration differs from patient analysis")
    # Check the fitted implementation is the one whose simulations were calibrated.
    calibration_protocol = read_json(Path(output)/"calibration/protocol.json")
    for name, record in calibration_protocol["sources"].items():
        if identity(ROOT/"utils"/name) != record:
            raise ValueError("Patient inference code differs from calibrated implementation")
    if not gate["allow_real_inference"]:
        status = "blocked_calibration_"+gate["status"]
        write_json(out/"inference_status.json", {"status": status, "allow_real_inference": False,
                                                 "calibration_source": identity(gate_file),
                                                 "reason": "Real-data test statistics and p values are not generated"})
        return status
    variants = [("main", data["train"], data["eval"], data["inner_fit"], data["inner_val"], False)]
    if data["dataset"] in ("gbmlgg", "ucec"):
        trkeep = np.ones(len(data["train"]["site"]), bool)
        evkeep = np.ones(len(data["eval"]["site"]), bool)
        if data["dataset"] == "ucec":
            trkeep = np.isin(data["train"]["clinical"], ["UEC", "USC"])
            evkeep = np.isin(data["eval"]["clinical"], ["UEC", "USC"])
        # Preserve original shared inner split, removing only ineligible clinical rows.
        index = np.cumsum(trkeep)-1
        fit = index[data["inner_fit"][trkeep[data["inner_fit"]]]]
        val = index[data["inner_val"][trkeep[data["inner_val"]]]]
        if data["dataset"] == "ucec":
            variants.append(("clinical_matched_base", _subset(data["train"], trkeep),
                             _subset(data["eval"], evkeep), fit, val, False))
        variants.append(("clinical", _subset(data["train"], trkeep), _subset(data["eval"], evkeep), fit, val, True))
    for variant, train, evaluation, fit, val, clinical in variants:
        folder = out/variant
        folder.mkdir(parents=True, exist_ok=True)
        result = patient_association(train, evaluation, fit, val, data["cuts"], config,
                                     seed_for(config.seed, 501, data["fold"]), clinical=clinical, save_models=True)
        joblib.dump(result.pop("models"), folder/"nuisance_models.joblib", compress=2)
        columns = {"case_id": evaluation["case_id"], "site_idx": evaluation["site"],
                   "event_after_admin": result["event_after_admin"], "time_after_admin": result["time_after_admin"],
                   "martingale_residual": result["martingale"], "cumulative_hazard": result["hazard"]}
        pd.DataFrame(columns).to_csv(folder/"patients.csv", index=False)
        rows = []
        for rep, record in result["representations"].items():
            rows.append({"representation": rep, **record["stat"]})
            np.savez_compressed(folder/f"{rep}_residuals.npz", case_id=evaluation["case_id"].astype(str),
                                **{key: record[key] for key in ("null", "phi", "conditional_mean", "residual", "products")})
            write_json(folder/f"{rep}_diagnostics.json", record["diagnostic"])
        pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        write_json(folder/"survival_diagnostics.json", result["survival_diagnostic"])
    write_json(out/"inference_status.json", {"status": "complete", "allow_real_inference": True,
                                             "calibration_source": identity(gate_file)})
    return "complete"


def execute_fold(reference, output, dataset, fold, config, stages):
    result = {"dataset": dataset, "fold": fold, "status": "complete", "stages": {}}
    out = Path(output)/dataset/f"fold_{fold}"
    try:
        data, out = prepare(reference, output, dataset, fold, config)
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_json(out/"prepare_failure.json", result)
        return result
    for stage in stages:
        folder = out/stage
        folder.mkdir(parents=True, exist_ok=True)
        signature = {"config": normalized(asdict(config)), "prepared": identity(out/"prepared.json"),
                     "code": {name: identity(ROOT/"utils"/name) for name in
                              ("semantics_stats.py", "semantics_nuisance.py", "semantics_pipeline.py")}}
        if stage == "patient":
            gate = Path(output)/"calibration/calibration_status.json"
            signature["calibration_gate"] = read_json(gate) if gate.exists() else None
        marker = folder/"complete.json"
        try:
            if marker.exists():
                existing = read_json(marker)
                was_blocked = existing["status"].startswith("blocked_calibration_")
                without_gate = lambda value: {key: item for key, item in value.items() if key != "calibration_gate"}
                gate_updated = stage == "patient" and was_blocked and without_gate(existing["signature"]) == without_gate(signature)
                if existing["signature"] != signature and not gate_updated:
                    raise ValueError(f"Stage {stage} source/config differs from saved result; choose a new output directory")
                if existing["signature"] == signature and all((folder/name).exists() for name in existing["artifacts"]):
                    result["stages"][stage] = existing["status"]
                    continue
            if stage == "site":
                run_site(data, folder, config)
                status = "complete"
            elif stage == "cohort":
                run_cohort(data, folder, config)
                status = "complete"
            elif stage == "patient":
                status = run_patient(data, folder, config, output)
            else:
                raise ValueError(f"Unsupported fold stage {stage}")
            result["stages"][stage] = status
            write_json(marker, {"signature": signature, "status": status,
                                "artifacts": [str(p.relative_to(folder)) for p in folder.rglob("*")
                                              if p.is_file() and p.name not in ("complete.json", "failure.json")]})
        except Exception as exc:
            result["stages"][stage] = "failed"
            result["status"] = "failed"
            write_json(folder/"failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
    write_json(out/"execution.json", result)
    return result
