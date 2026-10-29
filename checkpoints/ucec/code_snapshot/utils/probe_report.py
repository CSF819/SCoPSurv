"""Paired patient bootstrap and source-backed experiment 1 report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.independent_probes import REPRESENTATIONS, SurvivalEvaluation, write_json


def stratified_counts(strata, repetitions, rng):
    strata = np.asarray(strata)
    counts = np.zeros((repetitions, len(strata)), dtype=float)
    for group in np.unique(strata):
        idx = np.flatnonzero(strata == group)
        selected = rng.choice(idx, size=(repetitions, len(idx)), replace=True)
        np.add.at(counts, (np.arange(repetitions)[:, None], selected), 1)
    return counts


def weighted_cindex_samples(numerator, denominator, counts):
    numer = np.einsum("bi,ij,bj->b", counts, numerator, counts, optimize=True)
    denom = np.einsum("bi,ij,bj->b", counts, denominator, counts, optimize=True)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)


def paired_bootstrap(fold_inputs, repetitions):
    distributions = {}
    for fold, train, evaluation, metadata, predictions, coverage in fold_inputs:
        evaluator = SurvivalEvaluation(train, evaluation, metadata["finite_cuts"])
        eligible = coverage.set_index("case_id").loc[evaluation.case_id, "site_eligible"].to_numpy(dtype=bool)
        site_idx = np.flatnonzero(eligible)
        site_y = evaluation.site_idx.to_numpy()[site_idx]
        site_counts = stratified_counts(site_y, repetitions, np.random.default_rng(51000 + fold))
        survival_counts = stratified_counts(evaluator.event, repetitions, np.random.default_rng(52000 + fold))
        site_weight = np.array([1 / (len(np.unique(site_y)) * np.sum(site_y == s)) for s in site_y])
        for task in ("site", "survival"):
            ids = evaluation.case_id.iloc[site_idx] if task == "site" else evaluation.case_id
            for family in ("linear", "mlp"):
                for rep in REPRESENTATIONS:
                    chosen = predictions[(predictions.task == task) & (predictions.family == family)
                                         & (predictions.representation == rep)]
                    seed_values = []
                    for _, group in chosen.groupby("seed"):
                        if group.case_id.duplicated().any() or set(group.case_id) != set(ids):
                            raise ValueError("Missing, duplicated, or mismatched patient predictions.")
                        group = group.set_index("case_id").loc[ids]
                        if task == "site":
                            seed_values.append((group.predicted_site.to_numpy() == site_y).astype(float))
                        else:
                            hazards = group[[f"hazard_{j}" for j in range(4)]].to_numpy()
                            seed_values.append(evaluator.numerator(-np.cumprod(1 - hazards, axis=1).sum(axis=1)))
                    if not seed_values:
                        raise ValueError(f"No predictions for {task}/{family}/{rep}.")
                    mean_values = np.mean(seed_values, axis=0)
                    if task == "site":
                        samples = site_counts @ (mean_values * site_weight)
                    else:
                        samples = weighted_cindex_samples(mean_values, evaluator.pair_weights, survival_counts)
                    distributions.setdefault((task, family, rep), []).append(samples)
    means = {key: np.mean(values, axis=0) for key, values in distributions.items()}
    rows, contrasts = [], []
    for (task, family, rep), values in means.items():
        finite = values[np.isfinite(values)]
        bounds = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
        rows.append({"task": task, "family": family, "representation": rep,
                     "ci_low": bounds[0], "ci_high": bounds[1], "valid_repetitions": len(finite)})
    for family in ("linear", "mlp"):
        for name, task, left, right in (("background_site", "site", "z_b", "z_u"),
                                       ("fusion_site", "site", "h", "z_u"),
                                       ("survival_retention", "survival", "z_u", "h")):
            values = means[task, family, left] - means[task, family, right]
            finite = values[np.isfinite(values)]
            bounds = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            contrasts.append({"contrast": name, "task": task, "family": family,
                              "left": left, "right": right, "ci_low": bounds[0],
                              "ci_high": bounds[1], "valid_repetitions": len(finite)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def plot_results(primary, contrasts, out, n_folds, dataset="BLCA"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    labels = ["Fusion h", r"Background $z_b$", r"Cohort $z_s$", r"Individual $z_u$"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), layout="constrained")
    for ax, task, title, cmap in zip(axes, ("site", "survival"),
                                    ("Site decoding | balanced accuracy", "Survival decoding | IPCW C-index"),
                                    ("Blues", "Greens")):
        data = primary[primary.task == task].set_index(["representation", "family"])
        values = np.array([[data.loc[(rep, family), "mean"] for family in ("linear", "mlp")]
                           for rep in REPRESENTATIONS])
        im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1], ["Linear", "Two-layer MLP"])
        ax.set_yticks(range(4), labels)
        ax.set_title(title, fontsize=9, pad=10)
        ax.tick_params(labelsize=9)
        ax.tick_params(length=0)
        for i, rep in enumerate(REPRESENTATIONS):
            for j, family in enumerate(("linear", "mlp")):
                row = data.loc[(rep, family)]
                annotation = f"{row['mean']:.3f}" if np.isfinite(row["mean"]) else "NA"
                if np.isfinite(row.ci_low):
                    annotation += f"\n[{row.ci_low:.3f}, {row.ci_high:.3f}]"
                ax.text(j, i, annotation, ha="center", va="center", fontsize=8.5,
                        color="white" if values[i, j] > 0.65 else "#152432")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    fig.suptitle(f"{dataset} | frozen representations | {n_folds} folds | internal validation", fontsize=11)
    fig.supxlabel("Fold-mean performance [paired patient bootstrap 95% CI]", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"probe_matrix.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), layout="constrained")
    names = ("background_site", "fusion_site", "survival_retention")
    titles = (r"BA($z_b$) - BA($z_u$)", r"BA($h$) - BA($z_u$)", r"C($z_u$) - C($h$)")
    for ax, name, title in zip(axes, names, titles):
        rows = contrasts[contrasts.contrast == name].set_index("family")
        ax.axvline(0, color="#87929c", linestyle="--", linewidth=1)
        for y, family in enumerate(("linear", "mlp")):
            row = rows.loc[family]
            if np.isfinite(row.ci_low):
                ax.hlines(y, row.ci_low, row.ci_high, color="#326a9b", linewidth=2)
            ax.scatter(row.estimate, y, color="#143a57", s=38, zorder=3)
        ax.set_yticks([0, 1], ["Linear", "MLP"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Difference (95% CI)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Prespecified contrasts | fixed checkpoints | patient uncertainty", fontsize=10)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"probe_contrasts.{extension}", dpi=300)
    plt.close(fig)


def build_report(output_dir, folds, config):
    out = Path(output_dir)
    inputs, metrics, permutation_rows, coverages = [], [], [], []
    evaluation_ids = set()
    dataset_names = set()
    for fold in folds:
        folder = out / f"fold_{fold}"
        if not (folder / "complete.json").exists():
            raise ValueError(f"Fold {fold} probes are incomplete.")
        train = pd.read_csv(folder / "embeddings_train.csv.gz", float_precision="round_trip")
        evaluation = pd.read_csv(folder / "embeddings_eval.csv.gz", float_precision="round_trip")
        if evaluation_ids & set(evaluation.case_id):
            raise ValueError("Outer evaluation patients overlap between folds.")
        evaluation_ids.update(evaluation.case_id)
        metadata = json.loads((folder / "embeddings_metadata.json").read_text(encoding="utf-8"))
        dataset_names.add(metadata.get("dataset", metadata.get("checkpoint_args", {}).get("dataset", "blca")).upper())
        prediction = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
        coverage = pd.read_csv(folder / "site_coverage.csv")
        inputs.append((fold, train, evaluation, metadata, prediction, coverage))
        metrics.append(pd.read_csv(folder / "metrics.csv"))
        permutation_rows.append(pd.read_csv(folder / "permutation_controls.csv"))
        train_counts = train.loc[train.site_idx >= 0, "site_idx"].value_counts()
        coverages.append({"fold": fold, "train_n": len(train), "eval_n": len(evaluation),
                          "train_site_classes": int((train_counts >= config.min_site_count).sum()),
                          "eval_site_classes": int(coverage.loc[coverage.site_eligible, "site_idx"].nunique()),
                          "site_eval_n": int(coverage.site_eligible.sum()),
                          "site_coverage": float(coverage.site_eligible.mean()),
                          "below_threshold_n": int((coverage.reason == "training_site_below_threshold").sum()),
                          "unseen_or_unknown_n": int((coverage.reason == "unseen_or_unknown_site").sum())})
    if len(dataset_names) != 1:
        raise ValueError("Different datasets cannot be combined into one five-fold report.")
    dataset_name = dataset_names.pop()
    raw = pd.concat(metrics, ignore_index=True)
    raw.to_csv(out / "all_fold_metrics.csv", index=False)
    keys = ["fold", "task", "family", "representation", "cohort"]
    measures = ["balanced_accuracy", "macro_f1", "ipcw_cindex", "nll", "brier_1", "brier_2", "brier_3"]
    long = raw.melt(id_vars=keys + ["seed"], value_vars=measures, var_name="metric", value_name="value").dropna(subset=["value"])
    by_fold = long.groupby(keys + ["metric"]).value.agg(mean="mean", seed_sd="std", seeds="count").reset_index()
    by_fold.seed_sd = by_fold.seed_sd.fillna(0)
    by_fold.to_csv(out / "fold_seed_summary.csv", index=False)
    summary = by_fold.groupby(keys[1:] + ["metric"]).agg(
        mean=("mean", "mean"), fold_sd=("mean", "std"), mean_seed_sd=("seed_sd", "mean"), valid_folds=("fold", "count")).reset_index()
    summary["expected_folds"] = len(folds)
    summary.to_csv(out / "metric_summary.csv", index=False)
    primary = summary[((summary.task == "site") & (summary.metric == "balanced_accuracy")) |
                      ((summary.task == "survival") & (summary.metric == "ipcw_cindex") & (summary.cohort == "all"))]
    primary = primary[primary.representation.isin(REPRESENTATIONS)].copy()
    if len(primary) != 16 or (primary.valid_folds != len(folds)).any():
        raise ValueError("Primary metrics missing for one or more representation/fold combinations; inspect statuses.")
    print(f"[report] paired patient bootstrap: {config.bootstrap} repetitions", flush=True)
    intervals, contrasts = paired_bootstrap(inputs, config.bootstrap)
    primary = primary.merge(intervals, on=["task", "family", "representation"], validate="one_to_one")
    for idx, row in contrasts.iterrows():
        entries = primary[(primary.task == row.task) & (primary.family == row.family)].set_index("representation")
        contrasts.loc[idx, "estimate"] = entries.loc[row.left, "mean"] - entries.loc[row.right, "mean"]
    primary.to_csv(out / "primary_results.csv", index=False)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    pd.DataFrame(coverages).to_csv(out / "coverage_summary.csv", index=False)
    nulls = pd.concat(permutation_rows, ignore_index=True)
    nulls.to_csv(out / "all_permutation_controls.csv", index=False)
    if not nulls.empty:
        null_summary = nulls.groupby(["family", "representation"]).balanced_accuracy.agg(["mean", "std", "min", "max", "count"])
        null_summary.to_csv(out / "permutation_summary.csv")
    plot_results(primary, contrasts, out, len(folds), dataset_name)
    table = ["| 表示 | 站点 LR BA | 站点 MLP BA | 生存线性 C | 生存 MLP C |", "|---|---:|---:|---:|---:|"]
    for rep in REPRESENTATIONS:
        cells = []
        for task in ("site", "survival"):
            for family in ("linear", "mlp"):
                row = primary[(primary.task == task) & (primary.family == family) & (primary.representation == rep)].iloc[0]
                cells.append(f"{row['mean']:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]")
        table.append("| " + " | ".join([rep] + cells) + " |")
    lines = [f"# 实验 1：{dataset_name} 独立探针结果", "",
             "本结果为内部功能验证。原验证集曾参与主模型 best epoch 选择，不能据此报告独立测试性能。", "",
             "设计思路：冻结同一 checkpoint，用独立线性预测器和两层 MLP 比较四种表示中的站点信息与预后信息。", "",
             *( ["BRCA 沿用原数据加载器的 IDC 纳入规则，并按原始五折划分纳入患者。", ""] if dataset_name == "BRCA" else [] ),
             f"共 {len(evaluation_ids)} 名验证患者、{len(folds)} 折；站点主分析纳入 {sum(c['site_eval_n'] for c in coverages)} 名患者。每折只纳入原训练集至少 {config.min_site_count} 名患者的站点。", "",
             "表中为先平均探针种子、再平均各折的指标；括号为固定 checkpoint 下患者分层配对 bootstrap 的 95% 区间。", "", *table, "",
             "## 患者覆盖", "",
             "| Fold | 原训练患者 | 原验证患者 | 合格训练站点 | 实际评价站点 | 站点评价患者 | 覆盖率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in coverages:
        lines.append(f"| {row['fold']} | {row['train_n']} | {row['eval_n']} | {row['train_site_classes']} | "
                     f"{row['eval_site_classes']} | {row['site_eval_n']} | {row['site_coverage']:.1%} |")
    lines.extend(["", f"站点任务排除训练站点病例数不足的患者 {sum(c['below_threshold_n'] for c in coverages)} 名，"
                  f"训练中未出现或未知站点的患者 {sum(c['unseen_or_unknown_n'] for c in coverages)} 名；生存主分析包含所有患者。",
                  "", "## 参照结果", ""])
    chance = raw[raw.task == "site"].groupby("fold").uniform_chance.first().mean()
    baseline = summary[(summary.representation == "intercept") & (summary.cohort == "all")].set_index("metric")
    lines.append(f"均匀随机站点分类的折均值 BA 为 {chance:.4f}。仅截距生存模型的 C-index 为 {baseline.loc['ipcw_cindex', 'mean']:.4f}，NLL 为 {baseline.loc['nll', 'mean']:.4f}。")
    if not nulls.empty:
        lines.extend(["", "训练标签置换参照（跨折与置换组合的均值 ± SD，作为描述性负对照）：", "",
                      "| 表示 | LR BA | MLP BA |", "|---|---:|---:|"])
        for rep in REPRESENTATIONS:
            cells = [f"{null_summary.loc[(family, rep), 'mean']:.4f} ± {null_summary.loc[(family, rep), 'std']:.4f}"
                     for family in ("linear", "mlp")]
            lines.append("| " + " | ".join([rep] + cells) + " |")
    lines.extend(["", "## 生存辅助指标", "",
                  "C-index 越高越好；NLL 和 Brier 越低越好。下面的 NLL、Brier 使用全部验证患者，子集 C-index 使用站点主分析患者。Brier 1–3 对应各折自己的三个训练切点。",
                  "", "| 探针 | 表示 | 子集 C-index | NLL | Brier 1 | Brier 2 | Brier 3 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    survival_summary = summary[summary.task == "survival"].set_index(["family", "representation", "cohort", "metric"])
    for family, rep in [(f, r) for f in ("linear", "mlp") for r in REPRESENTATIONS] + [("intercept", "intercept")]:
        cells = []
        for cohort, metric in [("site_eligible", "ipcw_cindex")] + [("all", m) for m in ("nll", "brier_1", "brier_2", "brier_3")]:
            key = (family, rep, cohort, metric)
            cells.append(f"{survival_summary.loc[key, 'mean']:.4f}" if key in survival_summary.index else "NA")
        lines.append("| " + " | ".join([family, rep] + cells) + " |")
    lines.extend(["", "## 预设比较", ""])
    for _, row in contrasts.iterrows():
        metric = "站点 balanced accuracy" if row.task == "site" else "生存 IPCW C-index"
        lines.append(f"- {row.family}：{metric}，{row.left} − {row.right} = {row.estimate:+.4f}，95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]。")
    mlp = contrasts[contrasts.family == "mlp"].set_index("contrast")
    bg, site, survival = (mlp.loc[n] for n in ("background_site", "fusion_site", "survival_retention"))
    lines.extend(["", "## 结果解释", ""])
    if site.ci_low > 0:
        lines.append(f"MLP 探针下，z_u 相对 h 的站点可预测性降低；BA(h) − BA(z_u) 为 {site.estimate:.4f}，95% CI [{site.ci_low:.4f}, {site.ci_high:.4f}]。")
    if bg.ci_low > 0 and site.ci_low > 0:
        lines.append("MLP 探针中，z_u 的站点可预测性低于 z_b 和 h，区间均支持这一方向。")
    else:
        lines.append("MLP 探针的两个站点比较尚未同时支持 z_u 低于 z_b 和 h；应保留差值及区间，不能预设站点信息已被有效分离。")
    for rep in ("z_s", "z_u"):
        row = primary[(primary.task == "survival") & (primary.family == "mlp") & (primary.representation == rep)].iloc[0]
        if row.ci_low > 0.5:
            lines.append(f"{rep} 的 MLP 生存 C-index 为 {row['mean']:.4f}，95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}]，区间高于无特征参照 0.5，支持其包含可用于风险排序的预后信息。")
    if survival.ci_high < 0:
        lines.append("z_u 相对 h 的 MLP 生存预测能力下降，区间位于零以下，需要考虑预后信息损失。")
    elif survival.ci_low > 0:
        lines.append("z_u 相对 h 的 MLP 生存预测能力提高，区间位于零以上。")
    else:
        lines.append("z_u 与 h 的 MLP 生存预测差值区间跨零；本轮未设非劣效界值，不能将其解释为两者等效。")
    for rep in ("z_s", "z_u"):
        nll = survival_summary.loc[("mlp", rep, "all", "nll"), "mean"]
        if nll > baseline.loc["nll", "mean"]:
            lines.append(f"{rep} 的 MLP 生存 NLL 为 {nll:.4f}，高于仅截距参照 {baseline.loc['nll', 'mean']:.4f}；本配置下的概率预测质量未因风险排序能力而同步改善。Brier 结果见辅助指标表。")
    lines.extend(["z_s 的生存探针只检查预后信息是否可读出，群体条件的增量作用仍由实验 3 评价。", "",
                  "## 方法与文件", "",
                  "- 表示均为 256 维，模型在 eval / inference 模式下重新导出；每折已核验重复推理一致、参数及缓冲区未改变。",
                  "- 内部验证只来自原训练折；选择配置后在完整原训练数据上重拟合。探针种子不代表主模型重复训练。",
                  "- IPCW 删失分布来自原训练折；tau 为第三个有限切点。评价随访超出训练支持上限时行政删失；NLL 同样使用行政删失后的评价结局。Brier 的时间点随各折训练分位数变化，汇总按切点序号平均，未计算 IBS。",
                  "- 站点标签置换固定已选配置，作为负对照；未将其当作正式置换显著性检验。",
                  "- primary_results.csv：主指标及患者区间；paired_contrasts.csv：预设比较；metric_summary.csv：NLL、Brier 等全部指标。",
                  "- fold_seed_summary.csv：逐折种子变异；coverage_summary.csv：站点覆盖；各 fold 目录保留特征、内部划分、预测、模型与配置。", "",
                  "文献依据：[DIVA，附录 A.4.6](https://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf)；[DANN，§3.2、§5.1.5](https://jmlr.org/papers/volume17/15-239/15-239.pdf)。具体探针配置为本项目预设协议。", "",
                  f"![信息探针矩阵]({(out / 'probe_matrix.png').as_posix()})", "",
                  f"![预设配对比较]({(out / 'probe_contrasts.png').as_posix()})", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "report_complete.json", {"folds": folds, "patients": len(evaluation_ids),
                                               "bootstrap": config.bootstrap, "status": "complete"})
    print(f"[report] {out / 'report.md'}", flush=True)
