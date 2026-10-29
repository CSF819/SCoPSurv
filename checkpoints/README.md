# 最优轮次 checkpoint

目录结构为 `<dataset>/<dataset>_fold_<fold>/`，每折包含：

- `model_best_<C-index>_epoch<epoch>_fold<fold>.pth.tar`：该折验证 C-index 最优轮次的原始 checkpoint。
- `run_config.json`：保存的训练参数、癌种、fold 和 compressor 配置。
- `network_20260414_CFSurv.py`：该次运行保存的网络源码。

每个癌种选定一套完整五折配置，按各折原始 best checkpoint 收录。全部权重保留原始内容和文件名。

| 癌种 | Slot 模式 | 保存的核心配置 |
|---|---|---|
| BLCA | fixed，K=4 | BLCA v3 的 `orth0p01_rec0p0001_proto0p01_usage0p01` 配置 |
| BRCA | legacy fixed，K=2 | learning rate 2e-4 |
| GBMLGG | fixed，K=4 | `tri1_alpha0_orth0p5_rec0p001_proto1_usage0p1_acc1_clip0_warm0` |
| LUAD | legacy fixed，K=2 | `sitebg_20260708_153715_bg_nostop_adv005` |
| UCEC | adaptive，候选 K∈{1,2,4,8} | `cap0p001_ct1_pn4_pt0p1_use0p05_acc1_cl5_w3` |

BLCA、GBMLGG 和 UCEC 的 `code_snapshot/` 保存对应训练时的 `main.py`、网络及核心 `utils/`。BRCA、LUAD 的旧版 slot 推理由根目录中的 `utils/legacy_probe_path_compressor.py` 提供。

使用根目录 `evaluate.py` 自动读取每折配置、恢复对应结构并严格加载参数。当前 `main.py` 的默认超参数与部分归档配置不同。

早期 BRCA、LUAD checkpoint 保存了模型权重、轮次、分数和癌种，完整参数由配套 `run_config.json` 提供。其余三个癌种的 checkpoint 还包含训练状态。模型参数均以原文件形式保存。

逐折索引见 `metadata/checkpoint_sources.json` 与 `metadata/checkpoint_summary.csv`。
