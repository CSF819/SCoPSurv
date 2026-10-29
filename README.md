# SCoPSurv

Site-Cohort-Patient Representation Learning for Multimodal Cancer Survival Prediction.

本项目包含 SCoPSurv 的训练与评估代码、五个 TCGA 癌种的基因数据、患者级五折划分，以及每折验证 C-index 最优轮次的 checkpoint。

模型流程：WSI patch features 与基因特征 → pathology slot compression → multimodal fusion → site-associated、cohort-shared 和 patient-specific representations → prototype-conditioned FiLM → discrete-time survival prediction。

## 文件结构

```text
SCoPSurv-Code/
  main.py                    # 当前实现的训练入口
  evaluate.py                # 按保存的配置加载和评估归档权重
  models/CFSurv/             # 网络与 pathology slot compressor
  utils/                     # 数据加载、损失、优化器、训练引擎及权重加载
  csv/                       # 五份训练用基因与生存信息 CSV
  splits/                    # signatures.csv 与 25 份五折划分文件
  checkpoints/<dataset>/     # 每个癌种五个 best checkpoints、配置与网络快照
  metadata/                  # 权重索引、最优轮次及文件来源
  tests/                     # 核心回归测试
  requirements-lock.txt      # 已核对的 csfpytorch 环境版本
  UPLOAD_TO_GITHUB.md         # Git LFS 上传步骤
```

源代码沿用 `CFSurv` 作为内部类名和命令行模型参数；项目名称为 SCoPSurv。

## 环境

已核对环境：Python 3.11.9、PyTorch 2.4.1、NumPy 1.26.4，完整依赖见 `requirements-lock.txt`。

本机使用 Conda 环境 `csfpytorch`：

```powershell
& 'E:\anaconda3\envs\csfpytorch\python.exe' main.py CFSurv --help
```

其他机器请在 `csfpytorch` 环境中按 `requirements.txt` 配置依赖。PyTorch 的 CUDA 构建需匹配目标机器。

## 数据准备

`csv/tcga_<dataset>_all_clean.csv` 包含基因特征、TCGA 患者与切片标识、site、oncotree_code、survival_months 和 censorship。`censorship=0` 表示观察到事件，`censorship=1` 表示删失。患者划分位于 `splits/5foldcv/tcga_<dataset>/splits_<fold>.csv`。

WSI 特征需单独准备。各 checkpoint 使用 ResNet-50 的 1024 维 patch features：

```text
<data_root_dir>/
  BLCA/patch/<slide_id>.pt
  BRCA/patch/<slide_id>.pt
  GBMLGG/patch/<slide_id>.pt
  LUAD/patch/<slide_id>.pt
  UCEC/patch/<slide_id>.pt
```

每个 `.pt` 保存一个形状为 `[N_patches, 1024]` 的 Tensor，文件名移除原始切片的 `.svs` 等扩展名。多张切片按患者聚合。数据集名称目录使用大写。

`splits/signatures.csv` 定义六组基因集合。选择后的输入维度分别为 BLCA 3392、BRCA 3523、GBMLGG 3135、LUAD 3420、UCEC 150。UCEC 使用与归档权重一致的原始基因表。

数据说明见 [`csv/README.md`](csv/README.md)。

## 当前实现的训练

从项目根目录运行，例如训练 UCEC 的五折模型：

```powershell
& 'E:\anaconda3\envs\csfpytorch\python.exe' main.py CFSurv --sets ucec --fold 0,1,2,3,4 --lr 5e-5 --data_root_dir 'E:\Resnet50_Features_cyp'
```

上述命令使用当前实现的其他默认参数。归档最优权重的完整参数位于对应折的 `run_config.json`；各癌种采用的 slot 模式和损失系数存在差异。当前实现默认启用 adaptive slot capacity。训练输出保存在 `results/`，包含 best/last checkpoint、运行参数、数据检查和逐折指标。

训练时，生存时间分箱与基因特征标准化在训练折拟合，并应用于验证折。默认 batch size 为 1。

Linux 服务器在激活 `csfpytorch` 后可以使用相同参数：

```bash
conda activate csfpytorch
CUDA_VISIBLE_DEVICES=0 python main.py CFSurv --sets ucec --fold 0,1,2,3,4 --lr 5e-5 --data_root_dir /path/to/Resnet50_Features
```

## 归档最优 checkpoint

每个癌种保留同一配置下五折训练各自的最优轮次。下表是 checkpoint 内保存的验证分数，标准差按五折总体标准差计算。

| 癌种 | Checkpoints | 五折平均 C-index | 标准差 |
|---|---:|---:|---:|
| BLCA | 5 | 0.6701 | 0.0462 |
| BRCA | 5 | 0.6932 | 0.0349 |
| GBMLGG | 5 | 0.8562 | 0.0313 |
| LUAD | 5 | 0.6927 | 0.0262 |
| UCEC | 5 | 0.7080 | 0.0370 |

逐折轮次、精确分数和文件路径见 [`metadata/checkpoint_summary.csv`](metadata/checkpoint_summary.csv)。

BRCA、LUAD 的早期 checkpoint 配有原始网络快照和 legacy slot compressor；`evaluate.py` 自动选择相应结构并严格加载参数。BLCA、GBMLGG、UCEC 同时保留了训练时的核心代码快照。具体内容见 [`checkpoints/README.md`](checkpoints/README.md)。

检查全部 25 个权重的参数加载和合成输入前向传播：

```powershell
& 'E:\anaconda3\envs\csfpytorch\python.exe' evaluate.py --check_only --device cpu
```

使用真实 WSI 特征重新评估指定癌种和折：

```powershell
& 'E:\anaconda3\envs\csfpytorch\python.exe' evaluate.py --sets ucec --fold 0 --data_root_dir 'E:\Resnet50_Features_cyp'
```

评估默认输出到 `results/checkpoint_evaluation.csv`，同时记录 checkpoint 内的历史分数和本次重新计算的分数。加载与前向检查不计算真实患者的 C-index。

## 测试

```powershell
& 'E:\anaconda3\envs\csfpytorch\python.exe' -B -m unittest tests.test_cfsurv_quality -v
```

## GitHub 上传

基因表和 checkpoint 已配置 Git LFS。请按 [`UPLOAD_TO_GITHUB.md`](UPLOAD_TO_GITHUB.md) 使用 Git 命令上传，克隆后执行 `git lfs pull` 获取完整数据与权重。
