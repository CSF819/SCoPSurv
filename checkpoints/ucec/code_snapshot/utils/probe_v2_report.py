"""Paired fixed-model bootstrap and standalone scientific figures for v2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.independent_probes import SurvivalEvaluation, survival_nll_rows, write_json
from utils.probe_report import stratified_counts, weighted_cindex_samples
from utils.probe_v2 import identity, load_reference, read_json

KEYS = ["stage", "condition", "task", "family", "metric"]


def bounds(values):
    finite = np.asarray(values)[np.isfinite(values)]
    low, high = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
    return {"ci_low": low, "ci_high": high, "valid_repetitions": len(finite)}


def conditional_deltas(augmented_c, baseline_c, augmented_nll, baseline_nll):
    return augmented_c - baseline_c, baseline_nll - augmented_nll


def bootstrap_condition(group, evaluation, evaluator, counts, task, expected=None):
    ids = evaluation.case_id.to_numpy()
    numerators, nlls, correct = [], [], []
    if group.duplicated(["direction", "seed", "case_id"]).any():
        raise ValueError("Duplicate paired patient predictions")
    for _, pred in group.groupby(["direction", "seed"]):
        if set(pred.case_id) != set(ids):
            raise ValueError("Conditions have different paired evaluation patients")
        pred = pred.set_index("case_id").loc[ids]
        if not np.array_equal(pred.site_idx, evaluation.site_idx):
            raise ValueError("Prediction site labels differ from the frozen cache")
        if task == "site":
            correct.append((pred.predicted_site.to_numpy() == evaluation.site_idx.to_numpy()).astype(float))
        else:
            hazards = pred[[f"hazard_{j}" for j in range(4)]].to_numpy(float)
            if not np.isfinite(hazards).all() or (hazards < 0).any() or (hazards > 1).any():
                raise ValueError("Invalid patient hazards")
            numerators.append(evaluator.numerator(-np.cumprod(1 - hazards, axis=1).sum(1)))
            nlls.append(survival_nll_rows(hazards, evaluator.labels, ~evaluator.event))
    # Equal seed count per direction is required for these algebraic averages.
    if group.groupby("direction").seed.nunique().nunique() != 1:
        raise ValueError("Unequal probe seed coverage across random directions")
    if task == "site":
        y = evaluation.site_idx.to_numpy()
        weights = np.array([1 / (len(np.unique(y)) * np.sum(y == c)) for c in y])
        point = {"balanced_accuracy": float(np.dot(np.mean(correct, axis=0), weights))}
        samples = {"balanced_accuracy": counts @ (np.mean(correct, axis=0) * weights)}
    else:
        denom = evaluator.pair_weights.sum()
        point = {"ipcw_cindex": float(np.mean(numerators, axis=0).sum() / denom) if denom > 0 else np.nan,
                 "nll": float(np.mean(nlls))}
        samples = {"ipcw_cindex": weighted_cindex_samples(np.mean(numerators, axis=0), evaluator.pair_weights, counts),
                   "nll": counts @ np.mean(nlls, axis=0) / len(evaluation)}
    if expected is not None:
        for metric, value in point.items():
            if not np.isclose(value, expected[metric], rtol=0, atol=1e-10, equal_nan=True):
                raise ValueError(f"CSV predictions do not reproduce recorded {metric}; check export precision/source")
    return samples


def collect_dataset(reference, output, dataset, folds, config):
    all_metrics, all_predictions, coverage, permutations, ranks = [], [], [], [], []
    distributions = {}
    for fold in folds:
        root = Path(output) / dataset / f"fold_{fold}"
        train, evaluation, metadata, splits, provenance = load_reference(reference, dataset, fold)
        if read_json(root / "prepared.json")["provenance"] != provenance:
            raise ValueError("Report source differs from prepared training provenance")
        frames, metric_frames = [], []
        for stage in ("background", "erase", "conditional"):
            if not (root / stage / "complete.json").exists():
                raise ValueError(f"Incomplete {dataset} fold {fold} stage {stage}")
            for path in sorted((root / stage).rglob("predictions.csv.gz")):
                if not (path.parent / "complete.json").exists():
                    raise ValueError(f"Incomplete prediction unit {path.parent}")
                pred = pd.read_csv(path, float_precision="round_trip", low_memory=False)
                if "partition" in pred:
                    pred = pred[pred.partition == "eval"]
                frames.append(pred)
                metric_frames.append(pd.read_csv(path.parent / "metrics.csv"))
        predictions = pd.concat(frames, ignore_index=True)
        metrics = pd.concat(metric_frames, ignore_index=True)
        # Validate every declared head, including all seeds/directions; no silent partial report.
        for stage in ("erase", "conditional"):
            conditions = read_json(root / stage / "conditions.json")
            for c in conditions:
                for task in (("site", "survival") if stage == "erase" else ("survival",)):
                    for family in ("linear", "mlp"):
                        subset = predictions[(predictions.stage == stage) & (predictions.condition == c["condition"]) &
                                             (predictions.direction == c["direction"]) & (predictions.task == task) &
                                             (predictions.family == family)]
                        expected = {0} if task == "site" and family == "linear" else set(config.seeds)
                        if set(subset.seed) != expected:
                            raise ValueError(f"Missing seeds in {dataset}/{fold}/{stage}/{c}/{task}/{family}")
        for threshold in config.thresholds:
            source = root / "background" / f"ridge_n{threshold}"
            permutations.append(pd.read_csv(source / "permutations.csv"))
        rank_log = read_json(root / "erase/projection_ranks.json")
        ranks.extend({"dataset": dataset, "fold": fold, "partition": name, **r}
                     for name, rows in rank_log.items() for r in rows)
        cov = pd.read_csv(root / "coverage.csv")
        cov["dataset"], cov["fold"] = dataset, fold
        coverage.append(cov)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        eligible_eval = evaluation.iloc[splits["site_eval"]].reset_index(drop=True)
        eligible_train = train.iloc[splits["site_train"]].reset_index(drop=True)
        evaluator = SurvivalEvaluation(eligible_train, eligible_eval, metadata["finite_cuts"])
        site_counts = stratified_counts(eligible_eval.site_idx, config.bootstrap, np.random.default_rng(51000 + fold))
        surv_counts = stratified_counts(evaluator.event, config.bootstrap, np.random.default_rng(52000 + fold))
        for key, group in predictions.groupby(["stage", "condition", "task", "family"]):
            stage, condition, task, family = key
            if stage == "background" and condition == "z_b_n10":
                ids = read_json(root / "patient_splits_n10.json")["site_eval"]
                sample = evaluation.set_index("case_id").loc[ids].reset_index()
                counts = stratified_counts(sample.site_idx, config.bootstrap, np.random.default_rng(61000 + fold))
            else:
                sample, counts = eligible_eval, site_counts if task == "site" else surv_counts
            chosen_metrics = metrics
            for column, value in zip(("stage", "condition", "task", "family"), key):
                chosen_metrics = chosen_metrics[chosen_metrics[column] == value]
            chosen_metrics = chosen_metrics[chosen_metrics.partition.isna() | (chosen_metrics.partition == "eval")]
            metric_names = ["balanced_accuracy"] if task == "site" else ["ipcw_cindex", "nll"]
            expected = chosen_metrics[metric_names].mean().to_dict()
            for metric, values in bootstrap_condition(group, sample, evaluator, counts, task, expected).items():
                distributions.setdefault((*key, metric), []).append(values)
        print(f"[{dataset}] paired bootstrap fold {fold} complete", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    if any(len(values) != len(folds) for values in distributions.values()):
        raise ValueError("A condition is missing an outer fold")
    bootstrap = {key: np.mean(values, axis=0) for key, values in distributions.items()}
    outer = metrics[(metrics.partition.isna()) | (metrics.partition == "eval")].copy()
    value_columns = [c for c in ("balanced_accuracy", "macro_f1", "ipcw_cindex", "nll", "brier_1", "brier_2", "brier_3") if c in outer]
    grouping = ["dataset", "fold", "stage", "condition", "task", "family", "direction"]
    seed_mean = outer.groupby(grouping)[value_columns].mean().reset_index()
    seed_sd = outer.groupby(grouping)[value_columns].std(ddof=1).reset_index()
    fold_group = grouping[:-1]
    direction_sd = seed_mean.groupby(fold_group)[value_columns].std(ddof=1).reset_index()
    fold_mean = seed_mean.groupby(fold_group)[value_columns].mean().reset_index()
    summary = []
    for key, frame in fold_mean.groupby(["stage", "condition", "task", "family"]):
        for metric in value_columns:
            values = frame[metric]
            if values.notna().any() or (*key, metric) in bootstrap:
                summary.append({"dataset": dataset, **dict(zip(KEYS, (*key, metric))),
                                "mean": float(values.mean()) if values.notna().all() else np.nan,
                                "fold_sd": float(values.std(ddof=1)), "valid_folds": int(values.notna().sum()),
                                "folds": len(folds), **bounds(bootstrap.get((*key, metric), []))})
    primary = pd.DataFrame(summary)
    estimates = primary.set_index(KEYS)["mean"].to_dict()
    contrasts = []

    def contrast(stage, family, metric, left, right, name, task):
        lk, rk = (stage, left, task, family, metric), (stage, right, task, family, metric)
        values = bootstrap[lk] - bootstrap[rk]
        contrasts.append({"dataset": dataset, "stage": stage, "family": family, "task": task,
                          "metric": metric, "contrast": name, "left": left, "right": right,
                          "estimate": estimates[lk] - estimates[rk], **bounds(values)})

    for family in ("linear", "mlp"):
        for step in config.strengths:
            if step:
                for metric, task in (("balanced_accuracy", "site"), ("ipcw_cindex", "survival"), ("nll", "survival")):
                    contrast("erase", family, metric, f"inlp_{step}", f"random_{step}", f"inlp_minus_random_{step}", task)
                    contrast("erase", family, metric, "z_u", f"inlp_{step}", f"zu_minus_inlp_{step}", task)
        for rep in ("h", "z_b", "z_s", "z_u"):
            contrast("conditional", family, "ipcw_cindex", f"site_plus_{rep}", "site_only", f"delta_C_{rep}", "survival")
            contrast("conditional", family, "nll", "site_only", f"site_plus_{rep}", f"delta_NLL_{rep}", "survival")
    out = Path(output) / dataset
    metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_csv(out / "patient_predictions.csv.gz", index=False)
    fold_mean.to_csv(out / "fold_seed_direction_means.csv", index=False)
    seed_sd.to_csv(out / "probe_seed_sd.csv", index=False)
    direction_sd.to_csv(out / "random_direction_sd.csv", index=False)
    primary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    pd.concat(coverage, ignore_index=True).to_csv(out / "coverage.csv", index=False)
    pd.concat(permutations, ignore_index=True).to_csv(out / "permutation_controls.csv", index=False)
    pd.DataFrame(ranks).to_csv(out / "projection_ranks.csv", index=False)
    names = ["|".join(key) for key in bootstrap]
    np.savez_compressed(out / "paired_bootstrap.npz", keys=np.asarray(names),
                        samples=np.stack(list(bootstrap.values())))
    return primary, pd.DataFrame(contrasts), pd.concat(permutations), pd.concat(coverage)


def plot_figures(output, primary, contrasts, permutations, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {"ridge": "#245C85", "linear": "#C78325", "mlp": "#A14B70",
              "inlp": "#245C85", "random": "#A6A29A", "h": "#292B2F", "z_u": "#C78325"}

    def save(fig, stem):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=220)
        plt.close(fig)

    def interval(ax, x, row, color, marker="o"):
        if np.isfinite(row["ci_low"]):
            ax.vlines(x, row["ci_low"], row["ci_high"], color=color, linewidth=1.6)
        ax.scatter(x, row["mean"], color=color, marker=marker, s=38, zorder=4)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.8 * len(datasets)), 4.6), layout="constrained", squeeze=False)
    for ax, threshold in zip(axes[0], (5, 10)):
        for x, dataset in enumerate(datasets):
            for offset, family in ((-.19, "linear"), (0, "ridge"), (.19, "mlp")) if threshold == 5 else ((0, "ridge"),):
                row = primary[(primary.dataset == dataset) & (primary.stage == "background") &
                              (primary.condition == f"z_b_n{threshold}") & (primary.family == family) &
                              (primary.metric == "balanced_accuracy")].iloc[0]
                interval(ax, x + offset, row, colors[family], {"ridge": "o", "linear": "s", "mlp": "^"}[family])
            perm = permutations[(permutations.dataset == dataset) & (permutations.threshold == threshold)]
            draws = perm.groupby("repeat").balanced_accuracy.mean()
            if len(draws):
                ax.vlines(x + .34, draws.min(), draws.max(), color="#777777", linewidth=3)
                ax.scatter(x + .34, draws.mean(), marker="x", color="#777777", s=30)
            # Equal fold mean of 1/K, not 1/(mean K).
            coverage_path = out / dataset / "coverage.csv"
            if not coverage_path.exists():
                coverage_path = out / "coverage.csv"
            coverage = pd.read_csv(coverage_path) if coverage_path.exists() else None
            if coverage is not None:
                eligible = coverage[(coverage.partition == "train") & (coverage.threshold == threshold) & coverage.eligible]
                chance = (1 / eligible.groupby("fold").site_idx.nunique()).mean()
                ax.plot([x - .32, x + .27], [chance, chance], color="#333333", ls=":", lw=1.6)
        ax.set_xticks(range(len(datasets)), [d.upper() for d in datasets])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Site balanced accuracy")
        ax.set_title(f"1A | Background $z_b$ | training site n >= {threshold}")
        ax.grid(axis="y", alpha=.18)
    fig.legend(handles=[Line2D([], [], color=colors[f], marker=m, ls="", label=l) for f, m, l in (
        ("ridge", "o", "PCA + Ridge"), ("linear", "s", "Native logistic"), ("mlp", "^", "Native MLP"))] + [
        Line2D([], [], color="#777777", marker="x", label="Permutation mean / range"),
        Line2D([], [], color="#333333", ls=":", label="Uniform 1/K")], loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Fold means; colored intervals: paired patient bootstrap 95% CI", fontsize=10)
    save(fig, "1A_background_site")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.0 * len(datasets) + .6), squeeze=False, layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, family in enumerate(("linear", "mlp")):
            ax = axes[i, j]
            data = primary[(primary.dataset == dataset) & (primary.stage == "erase") & (primary.family == family)]
            def point(condition):
                sub = data[data.condition == condition].set_index("metric")
                return sub.loc["balanced_accuracy"], sub.loc["ipcw_cindex"]
            for method in ("random", "inlp"):
                conditions = ["h"] + [f"{method}_{s}" for s in (1, 2, 4, 8)]
                pairs = [point(c) for c in conditions]
                ax.plot([a["mean"] for a, b in pairs], [b["mean"] for a, b in pairs],
                        color=colors[method], marker="o" if method == "inlp" else "s",
                        ls="-" if method == "inlp" else "--", ms=4)
                for step, (a, b) in zip((0, 1, 2, 4, 8), pairs):
                    ax.hlines(b["mean"], a.ci_low, a.ci_high, color=colors[method], alpha=.45, lw=.8)
                    ax.vlines(a["mean"], b.ci_low, b.ci_high, color=colors[method], alpha=.45, lw=.8)
                    if step == 8 and method == "inlp":
                        ax.annotate(str(step), (a["mean"], b["mean"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
            for condition, marker in (("h", "D"), ("z_u", "*")):
                a, b = point(condition)
                ax.scatter(a["mean"], b["mean"], color=colors[condition], marker=marker, s=70, zorder=5)
                ax.hlines(b["mean"], a.ci_low, a.ci_high, color=colors[condition], alpha=.6, lw=1)
                ax.vlines(a["mean"], b.ci_low, b.ci_high, color=colors[condition], alpha=.6, lw=1)
            ax.set_title(f"{dataset.upper()} | {'MLP' if family == 'mlp' else 'Linear'}")
            ax.set_xlabel("Site balanced accuracy")
            ax.set_ylabel("Survival IPCW C-index")
            ax.grid(alpha=.18)
    # Common axis limits across datasets and probe families, from all plotted intervals.
    for metric, setter in (("balanced_accuracy", "set_xlim"), ("ipcw_cindex", "set_ylim")):
        rows = primary[(primary.stage == "erase") & (primary.metric == metric)]
        low = max(0, min(rows.ci_low.min(), rows["mean"].min()) - .035)
        high = min(1, max(rows.ci_high.max(), rows["mean"].max()) + .035)
        for ax in axes.flat:
            getattr(ax, setter)((low, high))
    fig.suptitle("1B | Site-information removal and retained prognostic information\n"
                 "Crossbars: marginal paired-bootstrap 95% CI; random: mean of five direction seeds", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=colors[c], marker=m, ls=ls, label=label) for c, m, ls, label in (
        ("h", "D", "", "Original h / iteration 0"), ("z_u", "*", "", r"Individual $z_u$"),
        ("inlp", "o", "-", "INLP: 0, 1, 2, 4, 8"), ("random", "s", "--", "Rank-matched random"))],
        loc="outside lower center", ncol=4, frameon=False)
    save(fig, "1B_site_survival_tradeoff")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(9, 2.8 * len(datasets) + .6), squeeze=False, layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(("ipcw_cindex", "nll")):
            ax = axes[i, j]
            ax.axvline(0, color="#777777", ls="--", lw=1)
            for y, rep in enumerate(("h", "z_b", "z_s", "z_u")):
                for dy, family, marker in ((-.12, "linear", "s"), (.12, "mlp", "^")):
                    row = contrasts[(contrasts.dataset == dataset) & (contrasts.stage == "conditional") &
                                    (contrasts.family == family) & (contrasts.metric == metric) &
                                    (contrasts.contrast == f"delta_{'C' if metric == 'ipcw_cindex' else 'NLL'}_{rep}")].iloc[0]
                    ax.hlines(y + dy, row.ci_low, row.ci_high, color=colors[family], lw=1.6)
                    ax.scatter(row.estimate, y + dy, color=colors[family], marker=marker, s=30)
            ax.set_yticks(range(4), [r"$h$", r"$z_b$", r"$z_s$", r"$z_u$"])
            ax.invert_yaxis()
            ax.set_title(dataset.upper())
            ax.set_xlabel("C(site + representation) - C(site)" if metric == "ipcw_cindex" else "NLL(site) - NLL(site + representation)")
            ax.grid(axis="x", alpha=.18)
    for j, metric in enumerate(("ipcw_cindex", "nll")):
        subset = contrasts[(contrasts.stage == "conditional") & (contrasts.metric == metric)]
        low, high = min(0, subset.ci_low.min()), max(0, subset.ci_high.max())
        pad = max(.015, (high - low) * .06)
        for ax in axes[:, j]:
            ax.set_xlim(low - pad, high + pad)
    fig.suptitle("1C | Prognostic increment conditional on site\n"
                 "Positive values indicate improvement; intervals: paired patient bootstrap 95% CI", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=colors[f], marker=m, label="MLP" if f == "mlp" else "Linear", ls="")
                        for f, m in (("linear", "s"), ("mlp", "^"))], loc="outside lower center", ncol=2, frameon=False)
    save(fig, "1C_conditional_increment")


def write_report(out, dataset, primary, contrasts, permutations, coverage, folds, repetitions=2000):
    lines = [f"# {dataset.upper()} 实验1修订版结果", "",
             f"固定最优 checkpoint；{len(folds)} 折内部验证。1A 检验背景表示的站点可预测性，"
             "1B 检验站点信息去除后的预后保留，1C 检验控制站点后的预后增量。", "",
             f"区间采用{repetitions}次患者级配对 bootstrap，条件和表示共享重采样患者；模型保持固定。"
             "神经网络先平均三个种子，随机投影再平均五个方向种子，最后等权平均各折。", "",
             "## 1A：背景表示的站点预测", "",
             "| 探针 | 训练站点阈值 | BA [95% CI] | 置换均值 / 范围 |",
             "|---|---:|---|---|"]
    for threshold, families in ((5, ("ridge", "linear", "mlp")), (10, ("ridge",))):
        for family in families:
            row = primary[(primary.stage == "background") & (primary.condition == f"z_b_n{threshold}") &
                          (primary.family == family) & (primary.metric == "balanced_accuracy")].iloc[0]
            controls = permutations[permutations.threshold == threshold].groupby("repeat").balanced_accuracy.mean()
            control = f"{controls.mean():.3f} / [{controls.min():.3f}, {controls.max():.3f}]" if family == "ridge" else "—"
            lines.append(f"| {family} | {threshold} | {row['mean']:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | {control} |")
    main = primary[(primary.stage == "background") & (primary.condition == "z_b_n5") &
                   (primary.family == "ridge") & (primary.metric == "balanced_accuracy")].iloc[0]
    controls = permutations[permutations.threshold == 5].groupby("repeat").balanced_accuracy.mean()
    train_cov = coverage[(coverage.partition == "train") & (coverage.threshold == 5) & coverage.eligible]
    chance = (1 / train_cov.groupby("fold").site_idx.nunique()).mean()
    observed = coverage[(coverage.partition == "eval") & (coverage.threshold == 5)]
    diagnostic_metrics = pd.read_csv(Path(out) / "fold_metrics.csv")
    diagnostic_metrics = diagnostic_metrics[(diagnostic_metrics.stage == "background") &
                                            (diagnostic_metrics.family == "ridge")]
    diagnostic_mean = diagnostic_metrics.groupby(["condition", "partition"]).balanced_accuracy.mean()
    lines += ["", "| Ridge 阈值 | 拟合集 BA | 内部验证 BA | 原验证 BA |", "|---|---:|---:|---:|"]
    for threshold in (5, 10):
        values = diagnostic_mean.loc[f"z_b_n{threshold}"]
        lines.append(f"| {threshold} | {values['fit']:.3f} | {values['inner_validation']:.3f} | {values['eval']:.3f} |")
    lines += ["", f"主分析覆盖 {int(observed.eligible.sum())}/{len(observed)} 名验证患者；均匀随机参照为 {chance:.3f}。",
              f"Ridge BA 与置换均值的差为 {main['mean'] - controls.mean():+.3f}。"
              + ("观察值超过20次置换结果的最大值，支持背景表示包含可预测的站点信息。" if main['mean'] > controls.max()
                 else "观察值未超过20次置换结果的最大值，本轮对背景站点信息的证据有限。"),
              "置换作为描述性负对照，不进行正式显著性检验；1A 不评价站点信息在分支间的相对集中程度。", "",
              "## 1B：站点信息去除与预后保留", "",
              "| 条件 | 探针 | Site BA [95% CI] | IPCW C [95% CI] | NLL |", "|---|---|---|---|---|"]
    for condition in ("h", "z_u", "inlp_1", "random_1", "inlp_2", "random_2", "inlp_4", "random_4", "inlp_8", "random_8"):
        for family in ("linear", "mlp"):
            rows = primary[(primary.stage == "erase") & (primary.condition == condition) & (primary.family == family)].set_index("metric")
            a, c, n = rows.loc["balanced_accuracy"], rows.loc["ipcw_cindex"], rows.loc["nll"]
            lines.append(f"| {condition} | {family} | {a['mean']:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}] | "
                         f"{c['mean']:.3f} [{c.ci_low:.3f}, {c.ci_high:.3f}] | {n['mean']:.3f} |")
    lines += ["", "投影强度预先固定为0、1、2、4、8轮。随机对照按各折实际删除秩匹配，"
              "具体秩见 projection_ranks.csv。完整的 INLP−Random、z_u−INLP 差值及区间见 paired_contrasts.csv。"
              "BA 和 C 的差值为左条件减右条件；NLL 差值也为左减右，负值表示左条件损失较低。", "",
              "这些结果描述重新训练的探针能够恢复的信息，不直接说明原风险头的因果依赖。", "",
              "预定8轮终点的 INLP−Random 配对差值：", "",
              "| 探针 | ΔBA [95% CI] | ΔC [95% CI] |", "|---|---|---|"]
    for family in ("linear", "mlp"):
        rows = contrasts[(contrasts.stage == "erase") & (contrasts.family == family) &
                         (contrasts.contrast == "inlp_minus_random_8")].set_index("metric")
        a, c = rows.loc["balanced_accuracy"], rows.loc["ipcw_cindex"]
        lines.append(f"| {family} | {a.estimate:+.3f} [{a.ci_low:+.3f}, {a.ci_high:+.3f}] | "
                     f"{c.estimate:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] |")
    lines += ["",
              "## 1C：站点条件下的预后增量", "",
              "| 表示 | 探针 | ΔC [95% CI] | ΔNLL [95% CI] |", "|---|---|---|---|"]
    for rep in ("h", "z_b", "z_s", "z_u"):
        for family in ("linear", "mlp"):
            rows = contrasts[(contrasts.stage == "conditional") & (contrasts.family == family)].set_index("contrast")
            c, n = rows.loc[f"delta_C_{rep}"], rows.loc[f"delta_NLL_{rep}"]
            lines.append(f"| {rep} | {family} | {c.estimate:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] | "
                         f"{n.estimate:+.3f} [{n.ci_low:+.3f}, {n.ci_high:+.3f}] |")
    conditional = contrasts[contrasts.stage == "conditional"]
    c_rows = conditional[conditional.metric == "ipcw_cindex"]
    n_rows = conditional[conditional.metric == "nll"]
    lines += ["", "ΔC=C(site+表示)−C(site)；ΔNLL=NLL(site)−NLL(site+表示)。"
              "正值分别表示区分能力改善和损失降低。两项证据分别判断，区间包含0不能作为等效或非劣效结论。", "",
              f"预定的8组表示×探针比较中，ΔC 区间整体高于0的有 {int((c_rows.ci_low > 0).sum())} 组；"
              f"ΔNLL 区间整体高于0的有 {int((n_rows.ci_low > 0).sum())} 组，"
              f"整体低于0的有 {int((n_rows.ci_high < 0).sum())} 组。"
              "区分能力与概率预测损失分别反映不同方面，不能只根据 C-index 判断预后预测全面改善。", "",
              "## 数据与解释范围", "",
              "- 原模型及配置曾根据原验证集选择；这是首轮结果之后的探索性内部功能验证。",
              "- 1B、1C 的删失分布只用合格原训练患者估计；超过训练随访支持上限的评价记录作行政删失。",
              "- 区间不包含主模型重新训练的不确定性；种子波动和随机方向波动分别见对应 CSV。",
              "- 1A 的 native logistic/MLP 是既有 z_b 站点探针的复用结果，其模型来源记录在各折 native_supplement 中。",
              "- 所有辅助指标、状态及行政删失数见 fold_metrics.csv；排除患者及原因见 coverage.csv。",
              "- 原实验全患者生存结果和四表示比较保留在 reference-dir；这里的生存结果采用共同站点合格人群重新拟合。", "",
              "## 方法依据", "",
              "独立冻结探针：[DIVA 附录 A.4.6](https://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf)。"
              "PCA＋Ridge 配置沿用本项目 motivation。", "",
              "迭代零空间投影：[INLP](https://aclanthology.org/2020.acl-main.647/)；"
              "随机删除方向对照：[Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/)。"
              "本项目采用 Ridge 构造 INLP，并重新训练生存探针评价信息保留。", ""]
    Path(out, "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_reports(reference, output, datasets, folds, config):
    results = []
    output = Path(output)
    write_json(output / "report_provenance.json", {
        "report_code": identity(__file__), "reference_dir": str(Path(reference).resolve()),
        "datasets": datasets, "folds": folds, "bootstrap": config.bootstrap,
        "averaging": "probe seeds, then directions, then equal-weight folds",
        "bootstrap_strata": {"site": "site", "survival": "administratively_censored_event"},
    })
    for dataset in datasets:
        p, c, perms, coverage = collect_dataset(reference, output, dataset, folds, config)
        write_report(output / dataset, dataset, p, c, perms, coverage, folds, config.bootstrap)
        # Dataset figure generation uses the same reviewed rows as the combined panels.
        plot_figures(output / dataset, p, c, perms, [dataset])
        results.append((p, c, perms))
    primary = pd.concat([r[0] for r in results], ignore_index=True)
    contrasts = pd.concat([r[1] for r in results], ignore_index=True)
    permutations = pd.concat([r[2] for r in results], ignore_index=True)
    primary.to_csv(output / "summary.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    permutations.to_csv(output / "permutation_controls.csv", index=False)
    plot_figures(output, primary, contrasts, permutations, datasets)
    lines = ["# 实验1修订版：冻结探针结果", "",
             "思路：直接检验背景站点信息，比较去除站点信息后的预后保留，并估计控制站点后的预后增量。", "",
             "固定最优 checkpoint，未重新训练主模型。当前结果是探索性内部验证。"
             f"图中为等权折均值和{config.bootstrap}次患者配对 bootstrap 的95%区间；主模型训练不确定性不在区间内。", "",
             "| 数据集 | z_b Ridge BA [95% CI] | 报告 |", "|---|---|---|"]
    for dataset in datasets:
        row = primary[(primary.dataset == dataset) & (primary.stage == "background") &
                      (primary.condition == "z_b_n5") & (primary.family == "ridge") &
                      (primary.metric == "balanced_accuracy")].iloc[0]
        lines.append(f"| {dataset.upper()} | {row['mean']:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | [{dataset.upper()}]({dataset}/report.md) |")
    lines += ["", "## 1A：背景表示的站点预测", "", "![1A](1A_background_site.png)", "",
              "彩色区间为患者抽样置信区间；灰色线段为20次标签置换结果的范围，两者含义不同。", "",
              "## 1B：站点信息去除与预后保留", "", "![1B](1B_site_survival_tradeoff.png)", "",
              "投影后的探针重新拟合，反映可恢复的信息。每条曲线保留全部预定强度；随机对照匹配实际删除秩。", "",
              "## 1C：站点条件下的预后增量", "", "![1C](1C_conditional_increment.png)", "",
              "两列正值分别表示 C-index 提高与 NLL 降低。完整指标、配对差值、患者覆盖及缺失原因见各数据集报告和 CSV。", "",
              "配置、缓存来源和训练代码状态见 invocation_all.json；报告代码来源见 report_provenance.json。"
              "每折划分和运行状态保存在对应 fold 目录。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "complete.json", {"status": "complete", "datasets": datasets, "folds": folds,
                                          "bootstrap": config.bootstrap, "summary_rows": len(primary),
                                          "contrast_rows": len(contrasts)})
