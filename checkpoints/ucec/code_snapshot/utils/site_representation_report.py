"""Matched site-probe comparison of h, z_b, z_s and z_u (extension of 1A)."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from utils.independent_probes import REPRESENTATIONS, make_splits, site_scores, write_json
from utils.probe_report import stratified_counts
from utils.probe_v2 import identity, load_reference, read_json, site_representation_folder

ORDER = tuple(REPRESENTATIONS)


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
    return {"ci_low": lo, "ci_high": hi, "valid_repetitions": len(finite)}


def paired_site_samples(predictions, evaluation, counts, expected_seeds):
    ids = evaluation.case_id.to_numpy()
    truth = evaluation.site_idx.to_numpy()
    if predictions.duplicated(["seed", "case_id"]).any() or set(predictions.seed) != set(expected_seeds):
        raise ValueError("Missing/duplicated site probe seeds or patients")
    correct = []
    for _, group in predictions.groupby("seed"):
        if set(group.case_id) != set(ids):
            raise ValueError("Site representations must use the same evaluation patients")
        group = group.set_index("case_id").loc[ids]
        if not np.array_equal(group.site_idx, truth):
            raise ValueError("Site labels differ from the frozen cache")
        correct.append((group.predicted_site.to_numpy() == truth).astype(float))
    weights = np.array([1 / (len(np.unique(truth)) * np.sum(truth == s)) for s in truth])
    average = np.mean(correct, axis=0) * weights
    return float(average.sum()), counts @ average


def collect_site_dataset(reference, output, dataset, folds, config):
    distribution, metrics, predictions, controls, coverage, recalls, sources = {}, [], [], [], [], [], []
    for fold in folds:
        train, evaluation, _, _, provenance = load_reference(reference, dataset, fold)
        folder = Path(output) / dataset / f"fold_{fold}"
        if read_json(folder / "prepared.json")["provenance"] != provenance:
            raise ValueError("Prepared/cache provenance differs")
        marker = read_json(folder / "site_comparison/complete.json")
        if marker["representations"] != list(ORDER):
            raise ValueError("Four-representation site run is incomplete")
        sources.append(provenance)
        for threshold in config.thresholds:
            split = make_splits(train, evaluation, fold, threshold)
            cohort = evaluation.iloc[split["site_eval"]].reset_index(drop=True)
            counts = stratified_counts(cohort.site_idx, config.bootstrap,
                                       np.random.default_rng((51000 if threshold == 5 else 61000) + fold))
            coverage.append({"dataset": dataset, "fold": fold, "threshold": threshold,
                             "train_n": len(split["site_train"]), "eval_n": len(cohort),
                             "all_train_n": len(train), "all_eval_n": len(evaluation),
                             "train_site_classes": len(split["site_classes"]),
                             "eval_site_classes": cohort.site_idx.nunique(),
                             "coverage": len(cohort) / len(evaluation), "uniform_chance": 1 / len(split["site_classes"])})
            for representation in ORDER:
                root = site_representation_folder(folder, representation)
                units = [root / f"ridge_n{threshold}"]
                if threshold == 5:
                    units.append(root / "native_supplement")
                for unit in units:
                    if read_json(unit / "complete.json")["status"] != "complete":
                        raise ValueError(f"Incomplete site probe: {unit}")
                    pred = pd.read_csv(unit / "predictions.csv.gz", float_precision="round_trip", low_memory=False)
                    logged = pd.read_csv(unit / "metrics.csv", float_precision="round_trip")
                    for frame in (pred, logged):
                        frame["representation"] = representation
                    predictions.append(pred)
                    metrics.append(logged)
                    outer = pred[pred.partition == "eval"]
                    families = {"ridge"} if unit.name.startswith("ridge") else {"linear", "mlp"}
                    if set(outer.family) != families:
                        raise ValueError("Missing site probe family")
                    for family, group in outer.groupby("family"):
                        seeds = config.seeds if family == "mlp" else (0,)
                        estimate, samples = paired_site_samples(group, cohort, counts, seeds)
                        expected = logged[(logged.partition == "eval") & (logged.family == family)].balanced_accuracy.mean()
                        if not np.isclose(estimate, expected, rtol=0, atol=1e-12):
                            raise ValueError("Site prediction metrics do not reproduce the recorded BA")
                        distribution.setdefault((representation, threshold, family), []).append(samples)
                        for seed, g in group.groupby("seed"):
                            for site, sg in g.groupby("site_idx"):
                                recalls.append({"dataset": dataset, "fold": fold, "representation": representation,
                                                "threshold": threshold, "family": family, "seed": seed,
                                                "site_idx": site, "n": len(sg),
                                                "recall": float(np.mean(sg.predicted_site == sg.site_idx))})
                perm = pd.read_csv(root / f"ridge_n{threshold}/permutations.csv")
                if len(perm) != config.permutations or perm.repeat.nunique() != config.permutations:
                    raise ValueError("Incomplete label permutations")
                perm["representation"] = representation
                controls.append(perm)
        print(f"[{dataset} fold {fold}] four site representations / paired bootstrap complete", flush=True)
    if any(len(v) != len(folds) for v in distribution.values()):
        raise ValueError("A representation is missing an outer fold")
    samples = {k: np.mean(v, axis=0) for k, v in distribution.items()}
    metrics = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    controls, coverage = pd.concat(controls, ignore_index=True), pd.DataFrame(coverage)
    keys = ["dataset", "fold", "representation", "threshold", "family", "partition"]
    fold_means = metrics.groupby(keys)[["balanced_accuracy", "macro_f1"]].mean().reset_index()
    seed_sd = metrics.groupby(keys)[["balanced_accuracy", "macro_f1"]].std(ddof=1).reset_index()
    summary = []
    for (rep, threshold, family), group in fold_means[fold_means.partition == "eval"].groupby(["representation", "threshold", "family"]):
        row = {"dataset": dataset, "representation": rep, "threshold": threshold, "family": family,
               "mean": group.balanced_accuracy.mean(), "fold_sd": group.balanced_accuracy.std(ddof=1),
               "macro_f1": group.macro_f1.mean(), "folds": len(folds), **interval(samples[rep, threshold, family])}
        diag = fold_means[(fold_means.representation == rep) & (fold_means.threshold == threshold) & (fold_means.family == family)]
        row.update(fit_ba=diag.loc[diag.partition == "fit", "balanced_accuracy"].mean(),
                   inner_validation_ba=diag.loc[diag.partition == "inner_validation", "balanced_accuracy"].mean(),
                   uniform_chance=coverage.loc[coverage.threshold == threshold, "uniform_chance"].mean())
        if family == "ridge":
            draws = controls[(controls.representation == rep) & (controls.threshold == threshold)].groupby("repeat").balanced_accuracy.mean()
            row.update(permutation_mean=draws.mean(), permutation_min=draws.min(), permutation_max=draws.max())
        summary.append(row)
    summary = pd.DataFrame(summary)
    means = summary.set_index(["representation", "threshold", "family"])["mean"].to_dict()
    contrasts = []
    for threshold in config.thresholds:
        for family in (("ridge", "linear", "mlp") if threshold == 5 else ("ridge",)):
            for left, right in combinations(ORDER, 2):
                lk, rk = (left, threshold, family), (right, threshold, family)
                contrasts.append({"dataset": dataset, "threshold": threshold, "family": family,
                                  "left": left, "right": right, "estimate": means[lk] - means[rk],
                                  **interval(samples[lk] - samples[rk])})
    out = Path(output) / "site_comparison" / dataset
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    metrics.to_csv(out / "fold_metrics.csv", index=False)
    fold_means.to_csv(out / "fold_seed_means.csv", index=False)
    seed_sd.to_csv(out / "probe_seed_sd.csv", index=False)
    predictions.to_csv(out / "patient_predictions.csv.gz", index=False)
    controls.to_csv(out / "permutation_controls.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(recalls).to_csv(out / "site_recall.csv", index=False)
    write_json(out / "cache_sources.json", sources)
    np.savez_compressed(out / "paired_bootstrap.npz", keys=np.asarray([f"{r}|{n}|{f}" for r, n, f in samples]),
                        samples=np.stack(list(samples.values())))
    return summary, contrasts, controls, coverage


def plot_site_comparison(out, summary, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#50565E", "#C78925", "#245C85", "#A14B70")
    labels = (r"$h$", r"$z_b$", r"$z_s$", r"$z_u$")
    for name, panels in (("1A_four_representation_ridge", ((5, "ridge"), (10, "ridge"))),
                         ("1A_four_representation_native", ((5, "linear"), (5, "mlp")))):
        fig, axes = plt.subplots(2, len(datasets), figsize=(max(4.5, 2.8 * len(datasets)), 6.7),
                                 squeeze=False, layout="constrained", sharey=True)
        selected = summary[summary.family.isin([f for _, f in panels])]
        ymax = min(1., max(.3, np.ceil(selected.ci_high.max() * 10) / 10))
        for i, (threshold, family) in enumerate(panels):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                rows = summary[(summary.dataset == dataset) & (summary.threshold == threshold) & (summary.family == family)].set_index("representation")
                ax.axhline(rows.uniform_chance.iloc[0], ls=":", color="#333333", lw=1.2)
                for x, (rep, color) in enumerate(zip(ORDER, colors)):
                    row = rows.loc[rep]
                    ax.vlines(x, row.ci_low, row.ci_high, color=color, lw=1.8)
                    ax.scatter(x, row["mean"], color=color, s=40, zorder=4)
                    if family == "ridge":
                        ax.vlines(x + .13, row.permutation_min, row.permutation_max, color="#888888", lw=1.1)
                        ax.scatter(x + .13, row.permutation_mean, facecolors="white", edgecolors="#888888", s=20, zorder=3)
                ax.set_xticks(range(4), labels)
                ax.set_xlim(-.5, 3.5)
                ax.set_ylim(0, ymax)
                ax.set_title(f"{dataset.upper()} | {'PCA + Ridge' if family == 'ridge' else ('MLP' if family == 'mlp' else 'Logistic')}\nTraining site n >= {threshold}")
                ax.grid(axis="y", alpha=.17)
                if j == 0:
                    ax.set_ylabel("Site balanced accuracy")
        fig.suptitle("1A | Site predictability of four frozen representations\nEqual fold means; paired patient bootstrap 95% CI", fontsize=12)
        legend = [Line2D([], [], color="#333333", ls=":", label="Uniform 1/K")]
        if name.endswith("ridge"):
            legend.append(Line2D([], [], color="#888888", marker="o", markerfacecolor="white", label="Permutation mean / range"))
        fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(Path(out) / f"{name}.{ext}", dpi=220)
        plt.close(fig)


def write_site_report(out, summary, contrasts, coverage, datasets, config):
    lines = ["# 实验1A扩展：四种表示的站点预测能力", "",
             "思路：用相同的患者、站点阈值、内部划分和独立探针，直接比较 h、z_b、z_s、z_u 的站点可预测性。"
             "保留现有 z_b 结果，为其余三种表示补做同配置的 PCA＋Ridge；原始256维 Logistic/MLP 作为补充。", "",
             f"每个数据集独立建模后等权平均各折；MLP 先平均三个种子。区间为{config.bootstrap}次按站点分层的患者级配对 bootstrap。"
             "所有表示共享重采样患者和20次训练标签置换种子。", "",
             "## 主分析：PCA＋Ridge，训练站点至少5例", "",
             "| 数据集 | h | z_b | z_s | z_u | 随机1/K | 验证患者覆盖 |",
             "|---|---|---|---|---|---|---|"]
    for dataset in datasets:
        rows = summary[(summary.dataset == dataset) & (summary.threshold == 5) & (summary.family == "ridge")].set_index("representation")
        values = [f"{rows.loc[r, 'mean']:.3f} [{rows.loc[r, 'ci_low']:.3f}, {rows.loc[r, 'ci_high']:.3f}]" for r in ORDER]
        cov = coverage[(coverage.dataset == dataset) & (coverage.threshold == 5)]
        lines.append(f"| {dataset.upper()} | " + " | ".join(values) + f" | {rows.uniform_chance.iloc[0]:.3f} | {cov.eval_n.sum()}/{cov.all_eval_n.sum()} |")
    lines += ["", "![PCA＋Ridge](1A_four_representation_ridge.png)", "",
              "灰色范围是20次置换结果的范围，彩色区间是患者抽样置信区间，两者含义不同。"
              "阈值10是敏感性分析，其类别数和患者覆盖发生变化，不直接把 BA 增加解释为表示改善。", "",
              "## 原始256维探针补充", "", "![原始维度](1A_four_representation_native.png)", "",
              "Logistic 和 MLP 复用此前同一 checkpoint、同一患者划分上的独立站点探针。"
              "各折对齐患者、真实标签和探针种子，并从预测重新验证 BA。", "",
              "## 两项配对比较（主分析 Ridge）", "",
              "| 数据集 | BA(z_b)−BA(z_u) [95% CI] | BA(h)−BA(z_u) [95% CI] |", "|---|---|---|"]
    for dataset in datasets:
        rows = contrasts[(contrasts.dataset == dataset) & (contrasts.threshold == 5) & (contrasts.family == "ridge")]
        values = []
        for left in ("z_b", "h"):
            row = rows[(rows.left == left) & (rows.right == "z_u")].iloc[0]
            values.append(f"{row.estimate:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]")
        lines.append(f"| {dataset.upper()} | " + " | ".join(values) + " |")
    lines += ["", "全部六对表示差值均保存在 paired_contrasts.csv，方向为左表示减右表示，未进行正式显著性检验。", "",
              "## 拟合与验证诊断（Ridge，站点至少5例）", "",
              "| 数据集 | 表示 | 拟合 BA | 内部验证 BA | 原验证 BA | 置换均值 / 范围 |", "|---|---|---|---|---|---|"]
    for dataset in datasets:
        for rep in ORDER:
            row = summary[(summary.dataset == dataset) & (summary.representation == rep) & (summary.threshold == 5) & (summary.family == "ridge")].iloc[0]
            lines.append(f"| {dataset.upper()} | {rep} | {row.fit_ba:.3f} | {row.inner_validation_ba:.3f} | {row['mean']:.3f} | "
                         f"{row.permutation_mean:.3f} / [{row.permutation_min:.3f}, {row.permutation_max:.3f}] |")
    lines += ["", "## 解释范围与复现", "",
              "这是固定最优 checkpoint 下的探索性内部验证；原验证集参与过主模型和配置选择。"
              "站点可预测性高表示能够恢复站点相关信息；仅凭站点预测不能判断预后贡献或因果机制。"
              "差值区间包含0不能解释为等效。", "",
              "训练配置保持：StandardScaler→PCA（最多128维，不白化）→balanced Ridge（alpha=1），"
              "内部划分与 PCA 种子为2026+fold。标准化和 PCA 在相应拟合患者上训练。"
              "站点阈值仅依据原训练集确定，各表示保留全部合格训练类别，macro 指标在评价集中实际出现的类别上计算。", "",
              "完整指标、患者预测、站点 recall、种子波动和覆盖率见各数据集子目录。"
              "固定缓存来源记录在 cache_sources.json；训练模型与变换位于原输出目录的各 fold/site_comparison/表示，"
              "z_b 继续引用原 fold/background。1B、1C 的模型和结果保持原样。", ""]
    Path(out, "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_site_report(reference, output, datasets, folds, config):
    root = Path(output) / "site_comparison"
    root.mkdir(parents=True, exist_ok=True)
    (root / "complete.json").unlink(missing_ok=True)
    all_results = []
    for dataset in datasets:
        result = collect_site_dataset(reference, output, dataset, folds, config)
        summary, contrasts, controls, coverage = result
        plot_site_comparison(root / dataset, summary, [dataset])
        write_site_report(root / dataset, summary, contrasts, coverage, [dataset], config)
        all_results.append(result)
    summary, contrasts, controls, coverage = [pd.concat([r[i] for r in all_results], ignore_index=True) for i in range(4)]
    for name, frame in (("summary", summary), ("paired_contrasts", contrasts), ("permutation_controls", controls), ("coverage", coverage)):
        frame.to_csv(root / f"{name}.csv", index=False)
    plot_site_comparison(root, summary, datasets)
    write_site_report(root, summary, contrasts, coverage, datasets, config)
    write_json(root / "complete.json", {"status": "complete", "datasets": datasets, "folds": folds,
                                       "representations": ORDER, "bootstrap": config.bootstrap,
                                       "summary_rows": len(summary), "contrasts": len(contrasts),
                                       "report_code": identity(__file__)})
