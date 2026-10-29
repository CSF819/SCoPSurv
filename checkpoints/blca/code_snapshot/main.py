# -*- coding: utf-8 -*-
import sys
import csv
import time
import random
import os
import torch
import numpy as np
import warnings
from shutil import copyfile

from models.CFSurv import network_20260414_CFSurv as cfsurv_net
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from torch.utils.data import DataLoader
from utils.options import parse_args
from utils.loss_factory_new import Loss_factory
from utils.engine import Engine
from utils.dataset_survival_patch_only import Generic_MIL_Survival_Dataset
from utils.config_utils import (
    as_bool,
    path_feature_dim,
    pure_config_dict,
    validate_runtime_args,
)
from utils.logging_utils import configure_run_logging

warnings.simplefilter(action='ignore', category=FutureWarning)
# ---------------- 工具：固定随机种子 ----------------
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ====== 配置快照工具：不改训练，仅做I/O ======
import json
from copy import deepcopy

EFFICIENCY_HEADER = [
    "dataset",
    "fold",
    "model",
    "extractor",
    "modal",
    "total_params_m",
    "trainable_params_m",
    "profile_tokens",
    "profile_path_dim",
    "profile_omic_dim",
    "macs_g",
    "flops_g",
    "flops_status",
    "best_epoch",
    "best_c_index",
    "fold_train_time_sec",
    "checkpoint_size_mb",
    "best_train_epoch_time_sec",
    "best_eval_epoch_time_sec",
    "best_train_samples_per_sec",
    "best_eval_samples_per_sec",
    "best_train_avg_tokens_per_sample",
    "best_eval_avg_tokens_per_sample",
    "best_train_peak_cuda_allocated_mb",
    "best_eval_peak_cuda_allocated_mb",
    "best_eval_slot_selected_k_mean",
    "best_eval_slot_expected_k_mean",
    "best_eval_slot_k1_ratio",
    "best_eval_slot_k2_ratio",
    "best_eval_slot_k4_ratio",
    "best_eval_slot_k8_ratio",
    "best_eval_slot_capacity_selection_entropy",
    "best_eval_slot_capacity_collapsed",
    "last_train_epoch_time_sec",
    "last_eval_epoch_time_sec",
]

def _namespace_to_pure_dict(ns):
    """Backward-compatible wrapper around the shared serializer."""
    return pure_config_dict(ns)

def _collect_path_compressor_config(model):
    """
    Collect path-compressor settings from the instantiated model.
    仅读取模型配置，不影响训练。
    """
    config = {}
    try:
        # Gene-free pathology slot compressor.
        compressor = getattr(model, "path_compressor", None)
        if compressor is not None:
            adaptive_capacity = bool(
                getattr(compressor, "adaptive_capacity", False)
            )
            compressor_config = {
                "class": compressor.__class__.__name__,
                "dim": getattr(compressor, "dim", None),
                "num_slots": getattr(compressor, "num_slots", None),
                "capacity_mode": (
                    "adaptive" if adaptive_capacity else "fixed"
                ),
                "adaptive_capacity": adaptive_capacity,
                "temperature": getattr(compressor, "temperature", None),
                "top_k_method": getattr(compressor, "top_k_method", None),
                "train_init_noise": getattr(
                    getattr(compressor, "slot_attention", None),
                    "train_init_noise",
                    None,
                ),
            }
            if adaptive_capacity:
                compressor_config.update(
                    {
                        "candidate_ks": list(
                            getattr(
                                compressor,
                                "ADAPTIVE_CAPACITY_CANDIDATES",
                                (),
                            )
                        ),
                        "capacity_temperature": getattr(
                            compressor,
                            "capacity_temperature",
                            None,
                        ),
                    }
                )
            else:
                compressor_config.update(
                    {
                        "keep_k": getattr(compressor, "keep_k", None),
                        "topk_ratio": getattr(
                            compressor,
                            "topk_ratio",
                            None,
                        ),
                    }
                )
            config["path_compressor"] = compressor_config
        # 模型级别的稳健化旋钮
        config["model_level"] = {
            "spfusion_dim": getattr(model, "C_sp", None),
            "feat_dim": getattr(model, "feat_dim", None),
            "token_keep_ratio": getattr(model, "token_keep_ratio", None),
            "path_slot_num": getattr(model, "path_slot_num", None),
        }
    except Exception as e:
        config["error"] = f"collect_path_compressor_failed: {e}"
    return config


def _build_loader(dataset, args, shuffle):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": bool(shuffle),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "pin_memory": as_bool(getattr(args, "pin_memory", False), False),
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
        kwargs["persistent_workers"] = as_bool(
            getattr(args, "persistent_workers", False), False
        )
    return DataLoader(dataset, **kwargs)


def _write_data_audit(results_dir, dataset_name, fold, dataset, train_dataset, val_dataset, args):
    audit_dir = os.path.join(results_dir, f"{dataset_name}_fold_{fold}")
    os.makedirs(audit_dir, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "fold": int(fold),
        "split": getattr(dataset, "last_split_audit", {}),
    }
    if as_bool(getattr(args, "audit_wsi_features", True), True):
        load_tensors = as_bool(getattr(args, "token_audit", False), False)
        payload["wsi"] = {
            "train": train_dataset.audit_wsi_files(load_tensors=load_tensors),
            "val": val_dataset.audit_wsi_files(load_tensors=load_tensors),
        }
    path = os.path.join(audit_dir, "data_audit.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(pure_config_dict(payload), fp, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _format_efficiency_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_efficiency_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    tmp_path = csv_path + ".tmp"
    rows = []
    if exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            rows = list(csv.reader(fp))
    if not rows:
        rows.append(EFFICIENCY_HEADER)
    elif rows[0] != EFFICIENCY_HEADER:
        previous_header = rows[0]
        migrated_rows = []
        for previous_row in rows[1:]:
            previous_values = dict(zip(previous_header, previous_row))
            migrated_rows.append(
                [
                    previous_values.get(key, "")
                    for key in EFFICIENCY_HEADER
                ]
            )
        rows = [EFFICIENCY_HEADER] + migrated_rows
    rows.append([_format_efficiency_value(row.get(key, "")) for key in EFFICIENCY_HEADER])
    with open(tmp_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def _profile_model_flops(model, profile_tokens, path_dim, omic_dim):
    """Profile a reference forward pass without advancing torch RNG state."""
    from thop import profile

    device = next(model.parameters()).device
    was_training = model.training
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []

    class ModelWrapper(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.model = wrapped_model

        def forward(self, x_path, x_omic):
            out = self.model(x_path=x_path, x_omic=x_omic, phase="eval")
            return out["hazards"][-1]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            wrapper = ModelWrapper(model).eval()
            d_path = torch.zeros(
                1, int(profile_tokens), int(path_dim), device=device
            )
            d_omic = torch.zeros(1, int(omic_dim), device=device)
            macs, _ = profile(
                wrapper,
                inputs=(d_path, d_omic),
                verbose=False,
            )
    finally:
        model.train(was_training)

    return float(macs / 1e9), float(2 * macs / 1e9)

def _save_run_config(results_dir, dataset, fold, args, model, net_module_path=None):
    """
    Save args and path-compressor settings for this run.
    仅保存运行快照，不修改训练超参数或流程。
    """
    os.makedirs(results_dir, exist_ok=True)
    snap_dir = os.path.join(results_dir, f"{dataset}_fold_{fold}")
    os.makedirs(snap_dir, exist_ok=True)

    # 1) 原样文本（和终端打印一致的可读版）
    txt_path = os.path.join(snap_dir, "run_args.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(args) + "\n")

    # 2) JSON 版（方便后续程序化读取）
    json_path = os.path.join(snap_dir, "run_config.json")
    args_dict = _namespace_to_pure_dict(args)
    compressor_dict = _collect_path_compressor_config(model)
    payload = {
        "dataset": dataset,
        "fold": fold,
        "args": deepcopy(args_dict),
        "effective_path_slot_config": cfsurv_net.format_path_slot_config(args),
        "path_compressor_snapshot": compressor_dict,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 3) 复制本次使用的网络定义文件，确保代码留档。
    if net_module_path and os.path.isfile(net_module_path):
        try:
            dst = os.path.join(snap_dir, os.path.basename(net_module_path))
            copyfile(net_module_path, dst)
        except Exception as e:
            print(f"[warn] 复制网络文件失败：{e}")

    print(f"[Config] 已保存配置到：{snap_dir}")

# ---------------- 工具：解析命令行显式覆盖参数 ----------------
def _extract_cli_overrides(argv):
    """
    返回用户在命令行显式传入的参数名集合。
    例如: --lr 1e-4 --bank_length 16 -> {'lr','bank_length'}
    """
    flags = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            name = tok.lstrip("-")
            # argparse: 连字符会转下划线
            name = name.replace("-", "_")
            # 处理布尔开关：下一个可能是值也可能是下一个flag
            flags.add(name)
            # 若下一个 token 是参数值，则一并跳过。
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 2
                continue
        i += 1
    return flags


# ---------------- 工具：把 auto_config 写回 args（只填充未在命令行显式覆盖的项） ----------------
def _apply_auto_config_to_args(args, auto_cfg, protected_keys):
    """
    args: argparse.Namespace
    auto_cfg: {'base': {...}, 'customized': {k: {'type':T,'default':v}}}
    protected_keys: set, 命令行显式传入的键（不覆盖）
    """
    # base
    for k, v in auto_cfg.get('base', {}).items():
        if k not in protected_keys:
            setattr(args, k, v)

    # customized
    for k, spec in auto_cfg.get('customized', {}).items():
        v = spec.get('default', None)
        if k not in protected_keys:
            setattr(args, k, v)

    return args


def _effective_config_lines(args, net_mod=cfsurv_net):
    """Return mode-aware summaries of the effective post-merge configuration."""
    dataset = str(getattr(args, "dataset", "")).lower()
    data_root_dir = str(getattr(args, "data_root_dir", ""))
    dataset_data_dir = os.path.join(data_root_dir, dataset.upper())
    slot_config = net_mod.format_path_slot_config(args)

    return [
        (
            "[EffectiveConfig][Data] "
            f"dataset={dataset}, data_root_dir={data_root_dir}, "
            f"dataset_data_dir={dataset_data_dir}, "
            f"extractor={getattr(args, 'extractor', None)}, "
            f"modal={getattr(args, 'modal', None)}, "
            f"which_splits={getattr(args, 'which_splits', None)}, "
            f"folds={getattr(args, 'fold', None)}"
        ),
        (
            "[EffectiveConfig][Training] "
            f"seed={getattr(args, 'seed', None)}, "
            f"num_epoch={getattr(args, 'num_epoch', None)}, "
            f"batch_size={getattr(args, 'batch_size', None)}, "
            f"loss={getattr(args, 'loss', None)}, "
            f"optimizer={getattr(args, 'optimizer', None)}, "
            f"scheduler={getattr(args, 'scheduler', None)}, "
            f"lr={getattr(args, 'lr', None)}, "
            f"weight_decay={getattr(args, 'weight_decay', None)}, "
            f"use_amp={as_bool(getattr(args, 'use_amp', False), False)}, "
            f"amp_dtype={getattr(args, 'amp_dtype', None)}"
        ),
        f"[EffectiveConfig][PathSlot] {slot_config}",
        (
            "[EffectiveConfig][Decomposition] "
            f"enable_tri_factor={as_bool(getattr(args, 'enable_tri_factor', True), True)}, "
            f"tri_dim={getattr(args, 'tri_dim', None)}, "
            f"tri_use_zb={as_bool(getattr(args, 'tri_use_zb', True), True)}, "
            f"tri_use_zs={as_bool(getattr(args, 'tri_use_zs', True), True)}, "
            f"tri_use_zu={as_bool(getattr(args, 'tri_use_zu', True), True)}, "
            f"prog_proto_num={getattr(args, 'prog_proto_num', None)}, "
            f"risk_conditioning={getattr(args, 'risk_conditioning', None)}, "
            f"site_bg_stop_grad={as_bool(getattr(args, 'site_bg_stop_grad', True), True)}, "
            f"grl_lambda={getattr(args, 'grl_lambda', None)}, "
            f"lambda_site_bg={getattr(args, 'lambda_site_bg', None)}, "
            f"lambda_site_adv={getattr(args, 'lambda_site_adv', None)}, "
            f"lambda_orth={getattr(args, 'lambda_orth', None)}, "
            f"lambda_rec={getattr(args, 'lambda_rec', None)}, "
            f"lambda_proto={getattr(args, 'lambda_proto', None)}, "
            f"lambda_proto_usage={getattr(args, 'lambda_proto_usage', None)}"
        ),
    ]


def main(args):
    validate_runtime_args(args)

    # 解析 fold 列表并构造统一后缀。
    folds = list(map(int, args.fold.split(',')))
    fold_suffix = "fold" + "+".join(str(f) for f in folds)

    datalist = args.sets.split(',')
    all_best_score = {}

    # 记录命令行显式覆盖项，避免被 auto-config 改写。
    cli_overrides = _extract_cli_overrides(sys.argv[2:])

    def _write_fold_progress_csv(csv_path, header, best_epoch_row, best_score_row, folds, fold_scores):
        completed = max(len(best_epoch_row) - 1, 0)
        pending = max(len(folds) - completed, 0)
        mean_score = float(np.mean(fold_scores)) if len(fold_scores) > 0 else 0.0
        std_score = float(np.std(fold_scores)) if len(fold_scores) > 0 else 0.0

        epoch_row = list(best_epoch_row) + [""] * pending
        score_row = list(best_score_row) + [""] * pending + [mean_score, std_score]

        tmp_path = csv_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerow(epoch_row)
            writer.writerow(score_row)
        os.replace(tmp_path, csv_path)

    for dataname in datalist:
        # 每个数据集使用独立运行目录。
        args.dataset = dataname.lower()

        # 目录格式：数据集_模型_时间_fold_实验名。
        run_time = time.strftime("[%Y-%m-%d]-[%H-%M-%S]")
        exp_suffix = f"_{args.experiment_name}" if getattr(args, "experiment_name", "") else ""
        results_dir = os.path.join("./results", f"{dataname}_{args.model}_{run_time}_{fold_suffix}{exp_suffix}")
        os.makedirs(results_dir, exist_ok=True)
        logger = configure_run_logging(os.path.join(results_dir, "run.log"))
        logger.info(
            "Starting dataset=%s model=%s folds=%s experiment=%s",
            dataname,
            args.model,
            folds,
            getattr(args, "experiment_name", ""),
        )
        efficiency_csv_path = os.path.join(results_dir, "efficiency_summary.csv")

        # Result CSV for this dataset/fold run
        csv_path = os.path.join(results_dir, f"{args.sets}_{fold_suffix}{exp_suffix}.csv")
        header = ["name"] + [f"fold {f}" for f in folds] + ["mean", "std"]
        print("############", csv_path)
        with open(csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)

        # === 加载模型模块，并对当前数据集应用 auto-config ===
        # 只填充默认超参数，命令行显式参数保持不变。
        net_mod = cfsurv_net


        if hasattr(net_mod, 'get_auto_config'):
            auto_cfg = net_mod.get_auto_config(dataset_name=args.dataset)
            args = _apply_auto_config_to_args(args, auto_cfg, protected_keys=cli_overrides)
            validate_runtime_args(args)
            for config_line in _effective_config_lines(args, net_mod):
                print(config_line)
        else:
            print("[Warn] model module does not expose get_auto_config; skip auto config.")

        # 记录每个 fold 的最佳 epoch 和分数。
        best_epoch_row = ["best epoch"]
        best_score_row = [dataname]
        fold_scores = []

        # ====== K-fold CV ======
        for fold in folds:
            set_seed(args.seed)
            dataset_data_dir = os.path.join(args.data_root_dir, dataname.upper())
            dataset = Generic_MIL_Survival_Dataset(
                # csv_path=rf"E:\WSI\csv20251002\tcga_{dataname}_all_clean_filtered.csv",
                csv_path=f"./csv/tcga_{dataname}_all_clean.csv",
                modal=args.modal,
                apply_sig=True,
                data_dir=dataset_data_dir,
                shuffle=False,
                seed=args.seed,
                patient_strat=False,
                n_bins=4,
                label_col="survival_months",
                args=args, #ablation专用
            )
            split_dir = f"./splits/{args.which_splits}/tcga_{dataname}"
            # split_dir = f"F:/PythonProjects/MCAT-master/splits/{args.which_splits}/tcga_{dataname}"
            train_dataset, val_dataset = dataset.return_splits(
                from_id=False,
                csv_path=f"{split_dir}/splits_{fold}.csv",
                set_name=args.dataset,
                extractor=args.extractor
            )
            args.num_sites = int(getattr(dataset, "num_sites", 1))
            print(f"[TriFactor] num_sites={args.num_sites}")
            audit_path = _write_data_audit(
                results_dir,
                dataname,
                fold,
                dataset,
                train_dataset,
                val_dataset,
                args,
            )
            logger.info("[DataAudit] saved: %s", audit_path)
            train_loader = _build_loader(train_dataset, args, shuffle=True)
            val_loader = _build_loader(val_dataset, args, shuffle=False)
            logger.info(
                "Dataset %s split_%s: train=%s val=%s",
                dataname,
                fold,
                len(train_dataset),
                len(val_dataset),
            )

            # ====== 构建模型、优化器和调度器 ======
            try:
                # 模型类由 args.model 指定。
                model_class = getattr(net_mod, args.model)
                model = model_class(args)

                # Save this run's config and path-compressor snapshot.
                try:
                    # 还原网络文件的真实路径，便于复制留档
                    net_file_path = None
                    if hasattr(net_mod, "__file__"):
                        net_file_path = net_mod.__file__
                    _save_run_config(results_dir, dataname, fold, args, model, net_module_path=net_file_path)
                except Exception as e:
                    print(f"[Config] 保存失败但不影响训练：{e}")

                # 统计模型参数量。
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params_m = total_params / 1e6
                trainable_params_m = trainable_params / 1e6
                print(f"[Model Efficiency] Total Params: {total_params_m:.2f} M")
                print(f"[Model Efficiency] Trainable Params: {trainable_params_m:.2f} M")

                # FLOPs 统计依赖 thop，可通过配置关闭。
                device_curr = next(model.parameters()).device
                profile_tokens = 1000
                profile_path_dim = int(
                    getattr(model, "C_sp", path_feature_dim(args.extractor))
                )
                profile_omic_dim = int(getattr(model, "genomics_idx", [0])[-1])
                macs_g = None
                flops_g = None
                flops_status = "ok"

                try:
                    if not as_bool(getattr(args, "profile_flops", True), True):
                        flops_status = "skipped_by_config"
                        print("[FLOPs] skipped by --profile_flops false")
                    else:
                        macs_g, flops_g = _profile_model_flops(
                            model,
                            profile_tokens,
                            profile_path_dim,
                            profile_omic_dim,
                        )
                        print(
                            f"[FLOPs] {flops_g:.2f} G FLOPs / {macs_g:.2f} G MACs "
                            f"(at {profile_tokens} patches)"
                        )

                except Exception as e:
                    flops_status = f"failed: {e}"
                    print(f"[FLOPs] Calculation failed or skipped: {e}")

                finally:
                    if 'model' in locals(): del model
                    torch.cuda.empty_cache()

                    # 重新实例化干净模型用于正式训练。
                    print("Re-instantiating model for training...")
                    model = model_class(args)
                    model.to(device_curr)

                print("----------------------------------------------")

            except Exception as e:
                raise RuntimeError(f"实例化模型 '{args.model}' 失败: {e}") from e

            '''
            base_lr = float(getattr(args, "lr", 2e-4))
            weight_decay = float(getattr(args, "weight_decay", 1e-4))
            optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

            total_epochs = int(getattr(args, "num_epoch", getattr(args, "max_epoch", 100)))
            warmup_epochs = int(getattr(args, "warmup_epochs", 3))
            T_max = max(int(total_epochs) - int(warmup_epochs), 1)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6),
                ],
                milestones=[warmup_epochs],
            )
            '''
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)

            engine = Engine(args, results_dir, fold)
            criterion = Loss_factory(args)

            # ====== 开始训练 ======
            fold_start_time = time.time()
            score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler, dataname)
            fold_end_time = time.time()
            elapsed_time = fold_end_time - fold_start_time
            print(f"Fold {fold} Total Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

            checkpoint_path = getattr(engine, "best_checkpoint_path", None)
            checkpoint_size_mb = 0.0
            if checkpoint_path and os.path.isfile(checkpoint_path):
                checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
            best_train_stats = getattr(engine, "best_epoch_stats", {}).get("train", {})
            best_eval_stats = getattr(engine, "best_epoch_stats", {}).get("eval", {})
            last_train_stats = getattr(engine, "last_epoch_stats", {}).get("train", {})
            last_eval_stats = getattr(engine, "last_epoch_stats", {}).get("eval", {})
            _write_efficiency_row(
                efficiency_csv_path,
                {
                    "dataset": dataname,
                    "fold": fold,
                    "model": args.model,
                    "extractor": args.extractor,
                    "modal": args.modal,
                    "total_params_m": total_params_m,
                    "trainable_params_m": trainable_params_m,
                    "profile_tokens": profile_tokens,
                    "profile_path_dim": profile_path_dim,
                    "profile_omic_dim": profile_omic_dim,
                    "macs_g": macs_g,
                    "flops_g": flops_g,
                    "flops_status": flops_status,
                    "best_epoch": epoch,
                    "best_c_index": score,
                    "fold_train_time_sec": elapsed_time,
                    "checkpoint_size_mb": checkpoint_size_mb,
                    "best_train_epoch_time_sec": best_train_stats.get("epoch_time_sec"),
                    "best_eval_epoch_time_sec": best_eval_stats.get("epoch_time_sec"),
                    "best_train_samples_per_sec": best_train_stats.get("samples_per_sec"),
                    "best_eval_samples_per_sec": best_eval_stats.get("samples_per_sec"),
                    "best_train_avg_tokens_per_sample": best_train_stats.get("avg_tokens_per_sample"),
                    "best_eval_avg_tokens_per_sample": best_eval_stats.get("avg_tokens_per_sample"),
                    "best_train_peak_cuda_allocated_mb": best_train_stats.get("peak_cuda_allocated_mb"),
                    "best_eval_peak_cuda_allocated_mb": best_eval_stats.get("peak_cuda_allocated_mb"),
                    "best_eval_slot_selected_k_mean": best_eval_stats.get("slot_selected_k_mean"),
                    "best_eval_slot_expected_k_mean": best_eval_stats.get("slot_expected_k_mean"),
                    "best_eval_slot_k1_ratio": best_eval_stats.get("slot_k1_ratio"),
                    "best_eval_slot_k2_ratio": best_eval_stats.get("slot_k2_ratio"),
                    "best_eval_slot_k4_ratio": best_eval_stats.get("slot_k4_ratio"),
                    "best_eval_slot_k8_ratio": best_eval_stats.get("slot_k8_ratio"),
                    "best_eval_slot_capacity_selection_entropy": best_eval_stats.get(
                        "slot_capacity_selection_entropy"
                    ),
                    "best_eval_slot_capacity_collapsed": best_eval_stats.get(
                        "slot_capacity_collapsed"
                    ),
                    "last_train_epoch_time_sec": last_train_stats.get("epoch_time_sec"),
                    "last_eval_epoch_time_sec": last_eval_stats.get("epoch_time_sec"),
                },
            )

            best_epoch_row.append(epoch)
            best_score_row.append(score)
            fold_scores.append(score)
            _write_fold_progress_csv(csv_path, header, best_epoch_row, best_score_row, folds, fold_scores)

        # ====== 汇总统计并写回 CSV ======
        mean_score = float(np.mean(fold_scores)) if len(fold_scores) > 0 else 0.0
        std_score = float(np.std(fold_scores)) if len(fold_scores) > 0 else 0.0
        all_best_score[dataname] = fold_scores

        print("############", csv_path)
        _write_fold_progress_csv(csv_path, header, best_epoch_row, best_score_row, folds, fold_scores)

    print(all_best_score)
    return all_best_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()

    args = parse_args(sys.argv[1])
    if args.model != "CFSurv":
        raise ValueError("This branch only supports CFSurv from models.CFSurv.network_20260414_CFSurv")

    main(args)
    print("Finished!")
