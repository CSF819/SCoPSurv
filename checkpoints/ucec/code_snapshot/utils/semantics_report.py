"""Inspectable fold aggregation and publication figures; no outcome-driven selection."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from utils.independent_probes import write_json
from utils.site_dependence import holm_adjust
from utils.semantics_stats import REPRESENTATIONS
from utils.probe_v2 import identity

DATASETS = ("blca", "brca", "gbmlgg", "luad", "ucec")
LABELS = {"h": "$h$", "z_b": "$z^b$", "z_s": "$z^s$", "z_u": "$z^u$", "a": "$a$"}
COLORS = ("#6c757d", "#0072B2", "#E69F00", "#009E73", "#CC79A7")


def interval(samples):
    samples = np.asarray(samples, float)
    finite = samples[np.isfinite(samples)]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return {"ci_low": low, "ci_high": high, "bootstrap_valid": len(finite)}


def resampling_diagnostic(samples, estimate):
    """Absolute R2 bootstrap can be biased with many sparse groups.

    Preserve its sampling distribution without labeling it an effect interval.
    Paired R2 contrasts and linear patient Brier averages retain their planned
    percentile intervals, with the full resampling arrays available for audit.
    """
    finite = np.asarray(samples, float)
    finite = finite[np.isfinite(finite)]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    mean = float(finite.mean()) if len(finite) else np.nan
    return {"resample_q025": low, "resample_q975": high, "resample_mean": mean,
            "resample_bias": mean-estimate, "bootstrap_valid": len(finite)}


def five_fold_p(pvalues, folds):
    values = np.asarray(pvalues, float)
    if set(folds) != set(range(5)) or len(folds) != 5 or not np.isfinite(values).all():
        return np.nan
    return float(min(1., 2*np.median(values)))


def holm_family(table, size):
    if not len(table):
        return table
    values = table.p_dataset.to_numpy(float)
    work = np.concatenate([np.where(np.isfinite(values), values, 1.), np.ones(max(0, size-len(values)))])
    corrected = holm_adjust(work)[:len(values)]
    table = table.copy()
    table["p_holm"] = np.where(np.isfinite(values), corrected, np.nan)
    table["holm_family_size"] = size
    return table


def collect(output, datasets, folds):
    tables = {name: [] for name in ("site", "cohort", "structure", "clinical", "patient", "supplement", "hsic", "coverage", "checkpoints", "censoring", "status")}
    resampling = {name: {} for name in ("site", "cohort", "clinical")}
    for dataset in datasets:
        for fold in folds:
            folder = Path(output)/dataset/f"fold_{fold}"
            tag = {"dataset": dataset, "fold": fold}
            prepare = folder/"prepared.json"
            if not prepare.exists():
                tables["status"].append({**tag, "stage": "prepare", "status": "missing"})
                continue
            try:
                prepared = json.loads(prepare.read_text(encoding="utf-8"))
                source = prepared["protocol"]["provenance"]
                records = [source["checkpoint"], *source["files"].values(),
                           *[source["cache_sources"][name] for name in ("clinical_csv", "split_csv")],
                           prepared["protocol"]["preparation_code"], *prepared["artifacts"].values()]
                for record in records:
                    if identity(record["path"]) != record:
                        raise ValueError(f"Prepared/source file changed: {record['path']}")
                for stage in ("site", "cohort", "patient"):
                    marker = folder/stage/"complete.json"
                    if marker.exists():
                        signed = json.loads(marker.read_text(encoding="utf-8"))["signature"]
                        for record in signed["code"].values():
                            if identity(record["path"]) != record:
                                raise ValueError(f"Stage implementation changed: {record['path']}")
            except (ValueError, OSError, KeyError) as exc:
                tables["status"].append({**tag, "stage": "source_verification", "status": "failed", "reason": str(exc)})
                continue
            meta = json.loads(Path(source["files"]["embeddings_metadata.json"]["path"]).read_text(encoding="utf-8"))
            tables["checkpoints"].append({**tag, "checkpoint": source["checkpoint"]["path"],
                                         "best_score": meta.get("model_loading", {}).get("best_score"),
                                         "epoch": meta.get("model_loading", {}).get("epoch"),
                                         "cache_train": source["files"]["embeddings_train.csv.gz"]["path"],
                                         "cache_eval": source["files"]["embeddings_eval.csv.gz"]["path"]})
            patients = pd.read_csv(folder/"patients_and_assignments.csv.gz", usecols=["case_id", "site_idx", "split"])
            tr = patients[patients.split == "train"]
            ev = patients[patients.split == "eval"]
            tables["coverage"].append({**tag, "original_train_n": meta["train_n"], "original_eval_n": meta["eval_n"],
                                      "train_n": len(tr), "eval_n": len(ev), "train_sites": tr.site_idx.nunique(),
                                      "eval_sites": ev.site_idx.nunique(), "eval_coverage": len(ev)/meta["eval_n"]})
            for stage in ("site", "cohort", "patient"):
                marker = folder/stage/"complete.json"
                failure = folder/stage/"failure.json"
                if marker.exists():
                    state = json.loads(marker.read_text(encoding="utf-8"))["status"]
                else:
                    state = "failed" if failure.exists() else "missing"
                tables["status"].append({**tag, "stage": stage, "status": state,
                                         "reason": json.loads(failure.read_text(encoding="utf-8")).get("error") if state == "failed" else ""})
            mapping = {"site": "site/metrics.csv", "cohort": "cohort/metrics.csv",
                       "structure": "cohort/prototype_structure.csv", "clinical": "cohort/clinical_metrics.csv"}
            for name, relative in mapping.items():
                path = folder/relative
                # Incomplete stages cannot contribute stale partial outputs.
                if path.exists() and (folder/relative.split("/")[0]/"complete.json").exists():
                    table = pd.read_csv(path)
                    for key, value in tag.items():
                        table[key] = value
                    tables[name].extend(table.to_dict("records"))
            for variant in ("main", "clinical", "clinical_matched_base"):
                path = folder/f"patient/{variant}/metrics.csv"
                if path.exists() and (folder/"patient/inference_status.json").exists():
                    status = json.loads((folder/"patient/inference_status.json").read_text(encoding="utf-8"))
                    if status["allow_real_inference"]:
                        table = pd.read_csv(path)
                        for key, value in {**tag, "variant": variant}.items():
                            table[key] = value
                        tables["patient"].extend(table.to_dict("records"))
            for name, relative in (("supplement", "site/supplement_site_probes.json"), ("hsic", "site/supplement_hsic.json")):
                path = folder/relative
                if path.exists():
                    values = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(values, list):
                        values = [values]
                    tables[name] += [{**record, **tag} for record in values]
            for name, relative in (("site", "site/resampling.npz"), ("cohort", "cohort/bootstrap.npz"),
                                    ("clinical", "cohort/clinical_bootstrap.npz")):
                path = folder/relative
                if path.exists() and (folder/relative.split("/")[0]/"complete.json").exists():
                    with np.load(path) as archive:
                        resampling[name][dataset, fold] = {key: archive[key] for key in archive.files if key != "counts"}
            pred_path = folder/"cohort/patient_predictions.csv.gz"
            if pred_path.exists() and (folder/"cohort/complete.json").exists():
                predictions = pd.read_csv(pred_path)
                for cut, group in predictions.groupby("cut_index"):
                    common = group.sensitivity_included.to_numpy(bool)
                    for model in ("marginal", "conditional"):
                        local = group.loc[common]
                        endpoint = local["cut"].to_numpy()
                        cases = local.event_after_admin.to_numpy(bool) & (local.time_after_admin.to_numpy() <= endpoint)
                        controls = local.time_after_admin.to_numpy() > endpoint
                        g = np.where(cases, local[f"G_time_{model}"], np.where(controls, local[f"G_cut_{model}"], np.nan))
                        use = (cases | controls) & np.isfinite(g) & (g > 0)
                        weights = 1/g[use]
                        tables["censoring"].append({**tag, "cut_index": cut, "censoring_model": model,
                                                    "common_n": len(local), "contributing_n": int(use.sum()),
                                                    "minimum_required_G": float(g[use].min()) if len(weights) else np.nan,
                                                    "max_ipcw": float(weights.max()) if len(weights) else np.nan,
                                                    "q95_ipcw": float(np.quantile(weights, .95)) if len(weights) else np.nan,
                                                    "ipcw_ess": float(weights.sum()**2/(weights@weights)) if len(weights) else np.nan})
    schemas = {"site": ["dataset", "fold", "representation", "adjusted_r2", "p"],
               "cohort": ["dataset", "fold", "population", "censoring_model", "prediction", "cut_index", "brier"],
               "structure": ["dataset", "fold", "partition", "prototype", "mean_weight"],
               "clinical": ["dataset", "fold", "representation", "partial_r2", "p"],
               "patient": ["dataset", "fold", "representation", "variant", "statistic", "p"],
               "supplement": ["dataset", "fold", "representation", "family", "seed", "balanced_accuracy"],
               "hsic": ["dataset", "fold", "hsic", "p_permutation_exploratory"],
               "coverage": ["dataset", "fold", "train_n", "eval_n", "eval_coverage"],
               "checkpoints": ["dataset", "fold", "checkpoint", "best_score", "epoch"],
               "censoring": ["dataset", "fold", "cut_index", "censoring_model", "max_ipcw"],
               "status": ["dataset", "fold", "stage", "status", "reason"]}
    return {key: pd.DataFrame(value) if value else pd.DataFrame(columns=schemas[key]) for key, value in tables.items()}, resampling


def summarize_site(table, samples):
    rows, contrasts = [], []
    if not len(table):
        return pd.DataFrame(), pd.DataFrame()
    for (dataset, rep), group in table.groupby(["dataset", "representation"], sort=False):
        group = group.sort_values("fold")
        group = group[np.isfinite(group.adjusted_r2)]
        index = REPRESENTATIONS.index(rep)
        complete = set(group.fold) == set(range(5)) and len(group) == 5 and np.isfinite(group.p).all()
        boot = [samples[dataset, f]["bootstrap"][:, index, 1] for f in group.fold]
        rows.append({"dataset": dataset, "representation": rep, "mean": group.adjusted_r2.mean(),
                     "r2_mean": group.r2.mean(), "folds": len(group), "complete_five_folds": complete,
                     "p_dataset": five_fold_p(group.p, group.fold),
                     **resampling_diagnostic(np.mean(boot, 0) if complete else [], group.adjusted_r2.mean())})
    for dataset, group in table.groupby("dataset"):
        pivot = group.pivot(index="fold", columns="representation", values="adjusted_r2")
        for right in ("h", "z_s", "z_u"):
            if not {"z_b", right} <= set(pivot.columns):
                continue
            local = pivot[["z_b", right]].dropna()
            complete = set(local.index) == set(range(5))
            boot = [samples[dataset, f]["bootstrap"][:, 1, 1] -
                    samples[dataset, f]["bootstrap"][:, REPRESENTATIONS.index(right), 1] for f in local.index]
            contrasts.append({"dataset": dataset, "left": "z_b", "right": right,
                              "mean": (local.z_b-local[right]).mean(), "folds": len(local),
                              **(interval(np.mean(boot, 0)) if complete else interval([]))})
    return holm_family(pd.DataFrame(rows), 20), pd.DataFrame(contrasts)


def summarize_cohort(table, samples):
    rows, comparisons, controls = [], [], []
    if not len(table):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    main = table[table.population != "permutation_control"]
    fields = ["dataset", "population", "censoring_model", "cut_index"]
    for key, group in main.groupby(fields, sort=False):
        dataset, population, censoring, cut = key
        pivot = group.pivot(index="fold", columns="prediction", values="brier")
        for name in ("proto", "reference", "km"):
            if name not in pivot:
                continue
            values = pivot[name].dropna()
            boot = []
            for fold in values.index:
                source = samples[dataset, fold]
                index = list(source["keys"]).index(f"{population}|{censoring}|{name}|{cut}")
                boot.append(source["samples"][index])
            rows.append(dict(zip(fields, key)) | {"prediction": name, "mean": values.mean(), "folds": len(values),
                        **(interval(np.mean(boot, 0)) if set(values.index) == set(range(5)) else interval([]))})
        if not {"proto", "reference"} <= set(pivot.columns):
            continue
        paired = pivot[["reference", "proto"]].dropna()
        boot = []
        for fold in paired.index:
            source = samples[dataset, fold]
            keys = list(source["keys"])
            boot.append(source["samples"][keys.index(f"{population}|{censoring}|reference|{cut}")] -
                        source["samples"][keys.index(f"{population}|{censoring}|proto|{cut}")])
        comparisons.append(dict(zip(fields, key)) | {"mean": (paired.reference-paired.proto).mean(),
                           "folds": len(paired), "main_endpoint": cut == 1,
                           **(interval(np.mean(boot, 0)) if set(paired.index) == set(range(5)) else interval([]))})
    null = table[table.population == "permutation_control"]
    if len(null):
        pivot = null.pivot(index=["dataset", "fold", "repeat", "cut_index"], columns="prediction", values="brier").reset_index()
        pivot["delta_brier"] = pivot.reference-pivot.proto
        for (dataset, cut, repeat), group in pivot.groupby(["dataset", "cut_index", "repeat"]):
            valid = group.dropna(subset=["delta_brier"])
            controls.append({"dataset": dataset, "cut_index": cut, "repeat": repeat, "folds": len(valid),
                             "delta_brier": valid.delta_brier.mean() if set(valid.fold) == set(range(5)) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame(controls)


def summarize_clinical(table, samples):
    rows = []
    if not len(table):
        return pd.DataFrame()
    for (dataset, rep), group in table.groupby(["dataset", "representation"], sort=False):
        group = group[np.isfinite(group.partial_r2)]
        boot = []
        for fold in group.fold:
            source = samples[dataset, fold]
            boot.append(source["samples"][:, list(source["keys"]).index(rep)])
        rows.append({"dataset": dataset, "representation": rep, "mean": group.partial_r2.mean(),
                     "folds": len(group), "p_dataset": five_fold_p(group.p, group.fold),
                     **resampling_diagnostic(np.mean(boot, 0) if set(group.fold) == set(range(5)) else [], group.partial_r2.mean())})
    return holm_family(pd.DataFrame(rows), 10)


def summarize_patient(table):
    rows = []
    if not len(table):
        return pd.DataFrame(columns=["dataset", "representation", "variant", "folds", "p_dataset", "p_holm"])
    for (dataset, rep, variant), group in table.groupby(["dataset", "representation", "variant"], sort=False):
        rows.append({"dataset": dataset, "representation": rep, "variant": variant,
                     "folds": len(group), "p_dataset": five_fold_p(group.p, group.fold),
                     "significant_folds_uncorrected": int((group.p <= .05).sum()),
                     "mean_residual_variance": group.representation_residual_variance.mean()})
    table = pd.DataFrame(rows)
    return pd.concat([holm_family(table[table.variant == "main"], 20),
                      holm_family(table[table.variant != "main"], 12)], ignore_index=True)


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": 10, "legend.fontsize": 8, "savefig.dpi": 220})
    return plt


def save_figure(figure, output, name):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(Path(output)/f"{name}.{extension}", bbox_inches="tight")


def forest_marks(ax, positions, group, color, marker, label=None):
    ax.scatter(positions, group["mean"], color=color, marker=marker, label=label, s=28, zorder=3)
    for x, (_, row) in zip(positions, group.iterrows()):
        if np.isfinite(row.get("ci_low", np.nan)):
            ax.vlines(x, row.ci_low, row.ci_high, color=color, lw=1.1)


def plot_site(summary, contrasts, output, datasets, fold_table=None):
    plt = figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), layout="constrained")
    for j, rep in enumerate(REPRESENTATIONS):
        if len(summary):
            group = summary[summary.representation == rep].set_index("dataset").reindex(datasets)
            forest_marks(axes[0], np.arange(len(datasets))+(j-1.5)*.16, group, COLORS[j], ("o", "s", "^", "D")[j], LABELS[rep])
            if fold_table is not None:
                for i, dataset in enumerate(datasets):
                    local = fold_table[(fold_table.dataset == dataset) & (fold_table.representation == rep)]
                    axes[0].scatter(i+(j-1.5)*.16+(local.fold.to_numpy()-2)*.012,
                                    local.adjusted_r2, color=COLORS[j], s=9, alpha=.3, zorder=2)
    for j, right in enumerate(("h", "z_s", "z_u")):
        if len(contrasts):
            group = contrasts[contrasts.right == right].set_index("dataset").reindex(datasets)
            forest_marks(axes[1], np.arange(len(datasets))+(j-1)*.18, group, COLORS[j+1], ("o", "s", "^")[j], "$z^b$ − "+LABELS[right])
    for ax in axes:
        ax.axhline(0, color="#777777", ls="--", lw=.8)
        ax.set_xticks(np.arange(len(datasets)), [d.upper() for d in datasets])
        ax.legend(frameon=False, ncol=2)
        ax.grid(axis="y", alpha=.15)
    axes[0].set(ylabel="Adjusted site $R^2$", title="1A · Site association (faint dots: folds)")
    axes[1].set(ylabel="Paired difference in adjusted $R^2$", title="Background relative to other representations")
    save_figure(fig, output, "1A_site_association")
    plt.close(fig)


def plot_cohort(structure, changes, clinical, controls, output, datasets, clinical_folds=None):
    plt = figure_style()
    if len(structure):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, max(4, .32*len(structure.groupby(["dataset", "fold"])))) ,
                                  layout="constrained", gridspec_kw={"width_ratios": [1, 1, .9]})
        ordered = [(d, f) for d in datasets for f in sorted(structure[structure.dataset == d].fold.unique())]
        for j, partition in enumerate(("train", "eval")):
            group = structure[structure.partition == partition]
            mat = group.pivot(index=["dataset", "fold"], columns="prototype", values="mean_weight").reindex(ordered)
            image = axes[j].imshow(mat.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="Blues")
            axes[j].set_xticks(range(4), ["0", "1", "2", "3"])
            axes[j].set_yticks(range(len(ordered)), [f"{d.upper()} / {f}" for d, f in ordered])
            axes[j].set(title=f"{partition.capitalize()} mean soft assignment", xlabel="Fold-local prototype ID")
            hard = group.pivot(index=["dataset", "fold"], columns="prototype", values="hard_n").reindex(ordered)
            for row in range(len(ordered)):
                for col in range(4):
                    axes[j].text(col, row, f"{mat.iloc[row, col]:.2f}\n({int(hard.iloc[row, col])})", ha="center", va="center",
                                 fontsize=6.2, color="white" if mat.iloc[row, col] > .55 else "#202020")
        for j, partition in enumerate(("train", "eval")):
            group = structure[(structure.partition == partition) & (structure.prototype == 0)].set_index(["dataset", "fold"]).reindex(ordered)
            axes[2].scatter(group.assignment_information, np.arange(len(ordered))+(j-.5)*.18,
                            color=COLORS[j+1], marker=("o", "s")[j], s=18, label=partition)
        axes[2].set_ylim(len(ordered)-.5, -.5)
        axes[2].set_yticks([])
        axes[2].set(xlabel="$I_a$ (nats)", title="Between-patient assignment variation")
        axes[2].legend(frameon=False)
        fig.colorbar(image, ax=axes[:2], shrink=.6, label="Mean weight; parentheses: hard assignment n")
        save_figure(fig, output, "1B_prototype_structure")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")
    for j, (population, censoring, label) in enumerate((
            ("primary", "marginal", "Marginal G (primary)"),
            ("sensitivity_common", "marginal", "Marginal G (common patients)"),
            ("sensitivity_common", "conditional", "Conditional G (common patients)"))):
        if len(changes):
            group = changes[(changes.population == population) & (changes.censoring_model == censoring) & (changes.cut_index == 1)]
            group = group.set_index("dataset").reindex(datasets)
            forest_marks(axes[0], np.arange(len(datasets))+(j-1)*.16, group, COLORS[j+1], ("o", "s", "^")[j], label)
    axes[0].axhline(0, color="#777777", ls="--", lw=.8)
    axes[0].set_xticks(np.arange(len(datasets)), [d.upper() for d in datasets])
    axes[0].set(ylabel=r"$\Delta$ Brier = reference − patient assignment", title="1B · Shared prognosis at the second cut")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(axis="y", alpha=.15)
    if len(clinical):
        ds = [d for d in datasets if d in set(clinical.dataset)]
        for j, rep in enumerate((*REPRESENTATIONS, "a")):
            group = clinical[clinical.representation == rep].set_index("dataset").reindex(ds)
            forest_marks(axes[1], np.arange(len(ds))+(j-2)*.13, group, COLORS[j], ("o", "s", "^", "D", "v")[j], LABELS[rep])
            if clinical_folds is not None:
                for i, dataset in enumerate(ds):
                    local = clinical_folds[(clinical_folds.dataset == dataset) & (clinical_folds.representation == rep)]
                    axes[1].scatter(i+(j-2)*.13+(local.fold.to_numpy()-2)*.01, local.partial_r2,
                                    color=COLORS[j], s=9, alpha=.3, zorder=2)
        axes[1].set_xticks(range(len(ds)), [d.upper() for d in ds])
        axes[1].legend(frameon=False, ncol=3)
    else:
        axes[1].text(.5, .5, "Clinical supplement unavailable\nfor the requested dataset/fold subset", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(ylabel="Subtype partial $R^2$ given site", title="Clinical association (faint dots: folds)")
    axes[1].grid(axis="y", alpha=.15)
    save_figure(fig, output, "1B_cohort_prognosis")
    plt.close(fig)
    if len(changes) and len(controls):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        for j, cut in enumerate((0, 1, 2)):
            group = changes[(changes.population == "primary") & (changes.censoring_model == "marginal") & (changes.cut_index == cut)]
            forest_marks(ax, np.arange(len(datasets))+(j-1)*.2, group.set_index("dataset").reindex(datasets),
                         COLORS[j+1], ("o", "s", "^")[j], f"Cut {cut+1}"+(" (primary)" if cut == 1 else ""))
            for i, dataset in enumerate(datasets):
                values = controls[(controls.dataset == dataset) & (controls.cut_index == cut)].delta_brier.dropna()
                ax.scatter(np.full(len(values), i+(j-1)*.2), values, s=7, color=COLORS[j+1], alpha=.25)
        ax.axhline(0, color="#777777", ls="--", lw=.8)
        ax.set_xticks(range(len(datasets)), [d.upper() for d in datasets])
        ax.set(ylabel=r"$\Delta$ Brier", title="All endpoints; faint dots: within-site assignment permutations")
        ax.legend(frameon=False, ncol=3)
        save_figure(fig, output, "1B_endpoints_and_controls")
        plt.close(fig)


def plot_patient(patient, calibration, gate, output, datasets):
    plt = figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained", gridspec_kw={"width_ratios": [1.35, 1]})
    names = ["site_only", "nonlinear_prototype", "near_deterministic_prototype", "residual_dependent_censoring", "linear_patient", "nonlinear_patient"]
    labels = ["Site only", "Nonlinear prototype", "Near-deterministic r(a)", "Censoring depends on r", "Linear patient effect", "Nonlinear patient effect"]
    if len(calibration):
        for j, profile in enumerate(sorted(calibration.profile.unique())):
            group = calibration[calibration.profile == profile].set_index("scenario").reindex(names)
            y = np.arange(6)+(j-1)*.19
            axes[0].scatter(group.rejection_rate, y, color=COLORS[j+1], marker=("o", "s", "^")[j], s=25,
                            label=f"{int(group.n_train.iloc[0])}/{int(group.n_eval.iloc[0])}, event {group.target_event_fraction.iloc[0]:.0%}")
            for position, (_, row) in zip(y, group.iterrows()):
                axes[0].hlines(position, row.ci99_low, row.ci99_high, color=COLORS[j+1], lw=1.)
    axes[0].axvline(.05, color="#666666", ls="--", lw=.8, label="Nominal 0.05")
    axes[0].axhline(3.5, color="#999999", lw=.7)
    axes[0].set_yticks(range(6), labels)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Rejection rate (99% binomial interval)", xlim=(-.015, 1.02), title="1C · Null rejection and alternative power")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
    if gate.get("allow_real_inference") and len(patient):
        table = patient[patient.variant == "main"].pivot(index="dataset", columns="representation", values="p_holm")
        table = table.reindex(index=datasets, columns=REPRESENTATIONS)
        values = table.to_numpy(float)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#EEEEEE")
        strength = -np.log10(np.maximum(values, 1e-12))
        peak = max(3., float(np.nanmax(strength))) if np.isfinite(strength).any() else 3.
        image = axes[1].imshow(strength, cmap=cmap, vmin=0, vmax=peak, aspect="auto")
        for i in range(len(datasets)):
            for j in range(4):
                axes[1].text(j, i, f"{values[i,j]:.3g}" if np.isfinite(values[i,j]) else "NA", ha="center", va="center", fontsize=8,
                             color="white" if strength[i, j] > .55*peak else "#202020")
        axes[1].set_xticks(range(4), [LABELS[r] for r in REPRESENTATIONS])
        axes[1].set_yticks(range(len(datasets)), [d.upper() for d in datasets])
        axes[1].set(title="Conditional association: Holm p values")
        fig.colorbar(image, ax=axes[1], shrink=.65, label="−log10(p); not an information ranking")
    else:
        axes[1].axis("off")
        axes[1].text(.5, .57, "Real-data 1C inference withheld", ha="center", va="center", fontsize=13)
        axes[1].text(.5, .40, f"Calibration status: {gate.get('status', 'not run')}\n"
                     f"Completed simulations: {gate.get('completed', 0)} / {gate.get('required', 3600)}\n"
                     "Site and cohort analyses remain available.", ha="center", va="center", linespacing=1.6)
    save_figure(fig, output, "1C_association_and_calibration")
    plt.close(fig)


def markdown_table(table, columns, digits=4):
    if not len(table):
        return "当前没有可汇总的记录。"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, np.nan)
            if isinstance(value, (float, np.floating)):
                value = f"{value:.{digits}g}" if np.isfinite(value) else "NA"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output, tables, site, contrasts, changes, clinical, patient, calibration, gate, config):
    lines = ["# 实验1：Site–Cohort–Patient 功能验证", "",
             "## 分析思路与证据范围", "",
             "本实验使用既定最优 checkpoint 的冻结表示，分别评价真实站点关联、软原型所承载的共享预后模式，以及控制 site 和软原型后的预后关联。三类统计量分别解释，不合并为总分。",
             "这是查看首轮结果后的探索性内部验证；原验证患者参与过 best checkpoint 选择。区间只反映固定 checkpoint、预处理和已拟合模型下的患者抽样不确定性。", "",
             "预设配对差值区间使用患者 bootstrap 的 2.5% 和 97.5% 分位数。各折等权汇总，不按患者数或显著性加权。绝对站点 R² 和临床额外 R² 的普通 bootstrap 分布存在明显有限样本偏移，图中展示逐折点和五折均值，其重采样分位数另存为诊断，不标成效应量置信区间。", "",
             f"执行协议：CPU；每进程 1 个计算线程；站点和临床置换 {config.permutations} 次；患者配对 bootstrap {config.bootstrap} 次；训练分配置换 {config.random_repeats} 次。Python 使用用户指定的 csfpytorch 环境。", "",
             "## 1A：Site-associated", "",
             "StandardScaler 仅拟合合格原训练患者，原验证集使用全部 256 维欧氏距离计算 PERMANOVA。下表给出五折平均调整后 R²，负值保留。数据集级检验用 min(1, 2×五折 p 值中位数) 聚合，主检验按 5 数据集×4 表示进行 Holm 校正。", "",
             markdown_table(site, ["dataset", "representation", "mean", "p_dataset", "p_holm", "folds"]), "",
             "背景表示相对其他表示的配对差值：", "",
             markdown_table(contrasts, ["dataset", "left", "right", "mean", "ci_low", "ci_high"]), "",
             f"![1A：站点解释率与配对差值]({(Path(output)/'figures/1A_site_association.png').resolve().as_posix()})", "",
             "站点组内协方差迹与样本量保存在各折 site/site_dispersion.csv。单例站点的分散度为缺失。PERMANOVA 同时可能受到分散程度和样本不平衡影响；站点分类准确率单独保存在已核对来源的 supplementary 表中。", ""]
    if len(site):
        background = site[site.representation == "z_b"]
        supported = background[background.p_holm <= .05].dataset.str.upper().tolist()
        lines += ["本轮 zᵇ 的站点关联通过主检验 Holm 校正的数据集："+("、".join(supported) if supported else "暂无")+"。相对信息集中程度须结合上述配对差异判断。", ""]
    lines += ["## 1B：Cohort", "",
              "四维软分配直接由 checkpoint 原型模块计算，温度保持原值。患者表同时保存 float32 模块输出和修正求和舍入误差后的 float64 权重。每折原型编号只在本折解释；图中的括号是硬分配人数，不能将跨折相同编号视为同一个群体。", "",
              "对每位验证患者，用其他站点的合格训练患者构造四条加权 KM 曲线。每个原型具有相同的供体站点总权重。患者分配预测与固定供体平均分配参照共享曲线、供体及删失权重，因此 ΔBrier>0 表示患者自身软分配改善了共享预后预测。", "",
              "主终点是原折第二个有限切点；下表比较边际删失模型及共同可估计人群上的条件删失敏感性结果。条件删失 RSF 固定 200 棵树、深度 5、叶节点至少 10 人，完全由原训练集拟合。", "",
              markdown_table(changes[changes.cut_index == 1] if len(changes) else changes,
                             ["dataset", "population", "censoring_model", "mean", "ci_low", "ci_high", "folds"]), ""]
    if len(changes):
        main = changes[(changes.cut_index == 1) & (changes.population == "primary")]
        positive = main[main.ci_low > 0].dataset.str.upper().tolist()
        negative = main[main.ci_high < 0].dataset.str.upper().tolist()
        lines += ["主终点区间完全高于 0 的数据集："+("、".join(positive) if positive else "暂无")+"；完全低于 0 的数据集："+("、".join(negative) if negative else "暂无")+"。区间包含 0 的结果不作等效或非劣效解释。", ""]
        marginal = changes[(changes.cut_index == 1) & (changes.population == "sensitivity_common") & (changes.censoring_model == "marginal")].set_index("dataset")
        conditional = changes[(changes.cut_index == 1) & (changes.population == "sensitivity_common") & (changes.censoring_model == "conditional")].set_index("dataset")
        unstable = [d.upper() for d in marginal.index.intersection(conditional.index)
                    if np.isfinite(marginal.loc[d, "mean"]) and np.isfinite(conditional.loc[d, "mean"]) and
                    np.sign(marginal.loc[d, "mean"]) != np.sign(conditional.loc[d, "mean"])]
        lines += ["在共同可估计患者上，ΔBrier 符号随删失模型变化的数据集："+("、".join(unstable) if unstable else "暂无")+"。对应结果应结合权重分布与区间判断稳定性。", ""]
    lines += ["软分配占用、熵、Iₐ、有效样本量及权重分布见 prototype_structure.csv。硬分配集中仍保留软分配分析；四条 KM 是原型加权患者的预后概要，不解释为已经识别出的真实潜在亚群分布。", "",
              f"![1B：各折原型占用与软分配差异]({(Path(output)/'figures/1B_prototype_structure.png').resolve().as_posix()})", "",
              f"![1B：共享预后预测与临床关联]({(Path(output)/'figures/1B_cohort_prognosis.png').resolve().as_posix()})", "",
              f"![1B：三个切点及分配置换参照]({(Path(output)/'figures/1B_endpoints_and_controls.png').resolve().as_posix()})", "",
              "临床补充使用 GBMLGG 六类和 UCEC 的 UEC/USC；未细分 UCEC 仅在临床补充中排除，其余三个数据集为单一亚型，不适用。临床额外解释率控制 site，标签在 site 内置换，单独组成 10 项 Holm 检验族：", "",
              markdown_table(clinical, ["dataset", "representation", "mean", "p_holm", "folds"]), "",
              "## 1C：Patient-specific", "",
              "条件变量 W=(site,a)。原训练集分别拟合随机生存森林和多输出 Kernel Ridge；内部验证负责选择参数，最终在完整合格训练集重新拟合。表示映射包含标准化原始 256 维和 64 维固定随机傅里叶特征。原验证集计算 m=event−Λ̂(time|W)、ξ=φ(r)−μ̂(W)，以其乘积的最大绝对标准化均值构造检验，使用中心化 Gaussian multiplier bootstrap 校准。", "",
              f"专项校准状态：**{gate.get('status', 'not_run')}**。已完成 {gate.get('completed', 0)}/{gate.get('required', 18*config.calibration_reps)} 次模拟。每个规模×场景要求 {config.calibration_reps} 次独立模拟；校准使用 {config.calibration_draws} 个乘子。", "",
              "预设门控规则：任一零假设场景的拒绝率 99% Clopper–Pearson 区间下限超过 0.05，判为明确的Ⅰ类错误膨胀；校准失败停止真实数据的 1C 推断。小规模运行不能作为通过校准的依据。", "",
              markdown_table(calibration, ["n_train", "n_eval", "scenario", "completed", "rejection_rate", "ci99_low", "ci99_high", "inflated"]), ""]
    if len(tables["censoring"]):
        extremes = tables["censoring"].sort_values("max_ipcw", ascending=False).head(5)
        position = lines.index("## 1C：Patient-specific")
        lines[position:position] = ["删失敏感性权重最大的五条折/终点记录（cut_index 从 0 开始；完整记录见 summary/censoring_fold_records.csv）：", "",
                                    markdown_table(extremes, ["dataset", "fold", "cut_index", "censoring_model", "minimum_required_G", "max_ipcw", "ipcw_ess"]), "",
                                    "有限但很大的 IPCW 权重保留原值，并在结果解释中说明其不稳定性；未通过事后截断权重改善指标。", ""]
    lines += [f"![1C：条件关联与校准诊断]({(Path(output)/'figures/1C_association_and_calibration.png').resolve().as_posix()})", ""]
    if gate.get("allow_real_inference"):
        lines += [markdown_table(patient, ["dataset", "representation", "variant", "p_dataset", "p_holm", "folds"]), "",
                  "阳性结果支持表示在当前 site 与软原型条件之外仍有预后关联。该检验依赖非信息删失及干扰函数估计质量；通过本轮模拟不保证普遍有效。不同表示的 p 值和最大统计量不用于排列信息量。", ""]
        if len(patient):
            main = patient[patient.variant == "main"]
            positives = main[(main.representation == "z_u") & (main.p_holm <= .05)].dataset.str.upper().tolist()
            lines += ["zᵘ 在数据集级主检验通过 Holm 校正的数据集："+("、".join(positives) if positives else "暂无")+"。"+
                      ("当前结果未提供 zᵘ 在 site 和软原型条件之外仍有预后关联的显著证据。" if not positives else "对应结论限于上述条件变量及当前检验。"), ""]
        if len(calibration):
            worst = calibration[calibration["null"]].sort_values("rejection_rate", ascending=False).iloc[0]
            nonlinear = calibration[calibration.scenario == "nonlinear_patient"].rejection_rate
            lines += [f"本轮校准的最大零假设拒绝率为 {worst.rejection_rate:.1%}，其 99% 区间为 [{worst.ci99_low:.4f}, {worst.ci99_high:.4f}]。虽然未触发预设门控，部分场景已接近失败边界；非线性患者效应的效能为 {nonlinear.min():.1%}–{nonlinear.max():.1%}。因此，阳性结果按探索性关联报告，阴性结果不等于表示中没有患者预后信息。", ""]
        lines += ["UCEC 的 clinical_matched_base 与 clinical 使用相同临床可用患者、原训练人群及内部划分，分别控制 (site,a) 和 (site,a,subtype)。GBMLGG 的 main 已使用完整临床可用患者，可直接与 clinical 对照。临床敏感性补充共 12 项单独 Holm 校正，避免将 UCEC 排除未细分亚型造成的人群变化归因于条件变量。", ""]
    else:
        lines += ["本轮不生成真实数据的 1C 推断结论。校准诊断及 1A、1B 结果完整保留；不能据此认定 zᵘ 已通过 patient-specific 功能验证。", ""]
    lines += ["## 可复现记录与结果解释", "",
              "逐折模型、患者名单、软分配、供体权重、曲线、预测、重采样、排除记录和阶段状态均保存在各数据集 fold 目录。汇总只在五折完整时生成五折推断；缺失折保留原因。训练数据及拟合对象在患者 bootstrap 中保持固定。", "",
              "多个分支同时具有站点或预后信息时，报告信息重叠。zᵇ 的站点关联、原型的共享预后作用及 zᵘ 的条件预后关联分别构成证据，不能互相替代。", "",
              markdown_table(tables["status"], ["dataset", "fold", "stage", "status", "reason"]), "",
              "## 文献和项目设定", "",
              "- [Anderson 与 Walsh, 2013](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/12-2010.1)：PERMANOVA 与分散度、样本不平衡的解释。",
              "- [Chen, 2019](https://proceedings.mlr.press/v97/chen19a/chen19a.pdf)：生存估计方法背景；本实验原型站点权重及固定分配参照是项目设定。",
              "- [VaDeSC](https://rmarcinkevics.github.io/files/2022-04-25-vadesc.pdf)：以未用于训练的临床变量评价分组。",
              "- [Kook 生存 TRAM-GCM 说明](https://lucaskook.github.io/papers/kook-biometrics-26.pdf)、[COMETs](https://arxiv.org/html/2402.14416v3)：生存残差乘积与双侧干扰函数估计。这里的 RSF、固定随机特征、高维最大统计量组合是本项目实现，经过专项模拟检查。",
              "- [固定分位数 p 值聚合](https://people.math.ethz.ch/~nicolai/confidence.pdf)：本实验使用固定 γ=0.5，不搜索分位数。",
              "- [scikit-survival Brier 定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html)：训练删失分布与可估计随访支持。", ""]
    (Path(output)/"实验1_结果说明.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(output, datasets, folds, config):
    output = Path(output)
    folder = output/"summary"
    figures = output/"figures"
    folder.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables, samples = collect(output, datasets, folds)
    for key, table in tables.items():
        table.to_csv(folder/f"{key}_fold_records.csv", index=False)
    site, contrasts = summarize_site(tables["site"], samples["site"])
    cohort, changes, controls = summarize_cohort(tables["cohort"], samples["cohort"])
    clinical = summarize_clinical(tables["clinical"], samples["clinical"])
    patient = summarize_patient(tables["patient"])
    for key, table in (("site", site), ("site_contrasts", contrasts), ("cohort_brier", cohort),
                       ("cohort_delta_brier", changes), ("cohort_permutation_controls", controls),
                       ("clinical", clinical), ("patient", patient)):
        table.to_csv(folder/f"{key}_summary.csv", index=False)
    status = output/"calibration/calibration_status.json"
    gate = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {"status": "not_run", "allow_real_inference": False}
    calibration_path = output/"calibration/calibration_summary.csv"
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    plot_site(site, contrasts, figures, datasets, tables["site"])
    plot_cohort(tables["structure"], changes, clinical, controls, figures, datasets, tables["clinical"])
    plot_patient(patient, calibration, gate, figures, datasets)
    write_report(output, tables, site, contrasts, changes, clinical, patient, calibration, gate, config)
    write_json(output/"report_status.json", {"status": "complete", "datasets": datasets, "folds": folds,
                                            "config": asdict(config), "calibration_status": gate.get("status"),
                                            "patient_inference_allowed": gate.get("allow_real_inference", False),
                                            "stage_counts": tables["status"].status.value_counts().to_dict() if len(tables["status"]) else {}})
    print(f"Report written: {output/'实验1_结果说明.md'}", flush=True)
