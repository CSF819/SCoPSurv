import os
import json
import random
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.config_utils import as_bool, pure_config_dict
from utils.logging_utils import get_run_logger

try:
    from utils.tri_factor_probe import run_site_probes, summarize_prototype_usage
except Exception:
    run_site_probes = None
    summarize_prototype_usage = None


torch.set_num_threads(4)


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.best_score = 0.0
        self.best_epoch = 0
        self.filename_best = None
        self.epoch = 0
        self.last_epoch_stats = {}
        self.best_epoch_stats = {}
        self.best_checkpoint_path = None
        self._tri_buffers = {}
        self.last_checkpoint_path = None
        self.resume_epoch = -1
        self.use_amp = False
        self.scaler = None
        self.optimizer_step_count = 0
        self.non_blocking = as_bool(getattr(args, "non_blocking", False), False)
        self.logger = get_run_logger()

    def _log(self, message, level="info"):
        if self.logger.handlers:
            getattr(self.logger, level)(message)
        else:
            print(message)

    @staticmethod
    def _gradient_norm(optimizer):
        norms = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    norms.append(parameter.grad.detach().norm(2))
        if not norms:
            return 0.0
        return float(torch.stack(norms).norm(2).detach().cpu().item())

    def _take_optimizer_step(self, optimizer, grad_clip_norm):
        if self.use_amp:
            self.scaler.unscale_(optimizer)

        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if grad_clip_norm > 0 and parameters:
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                parameters,
                grad_clip_norm,
            )
            if torch.is_tensor(grad_norm_value):
                grad_norm_value = grad_norm_value.detach().cpu().item()
            grad_norm = float(grad_norm_value)
            was_clipped = grad_norm > grad_clip_norm
        else:
            grad_norm = self._gradient_norm(optimizer)
            was_clipped = False

        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        self.optimizer_step_count += 1
        return grad_norm, was_clipped

    def _pretrim_wsi_tokens(self, data_WSI, phase):
        if not torch.is_tensor(data_WSI) or data_WSI.dim() != 3:
            return data_WSI
        max_tokens = int(getattr(self.args, "max_tokens", 0) or 0)
        min_tokens = int(getattr(self.args, "min_tokens", 0) or 0)
        adaptive_sampling = as_bool(
            getattr(self.args, "adaptive_sampling", False), False
        )
        if not adaptive_sampling:
            return data_WSI

        B, N, C = data_WSI.shape
        if N == 0 and min_tokens > 0:
            return torch.randn(B, min_tokens, C, dtype=data_WSI.dtype) * 1e-4

        if max_tokens > 0 and N > max_tokens:
            if phase == "train":
                idx = torch.randperm(N)[:max_tokens]
            else:
                idx = torch.linspace(0, N - 1, steps=max_tokens).long()
            data_WSI = data_WSI[:, idx, :]
            N = data_WSI.shape[1]

        if min_tokens > 0 and 0 < N < min_tokens:
            repeat_times = int(np.ceil(min_tokens / float(N)))
            data_WSI = data_WSI.repeat(1, repeat_times, 1)[:, :min_tokens, :]
            data_WSI = data_WSI + torch.randn_like(data_WSI) * 1e-4

        return data_WSI

    @staticmethod
    def _log_path_compressor(writer, phase, viz, global_step):
        path_compressor = viz.get("path_compressor", None)
        if path_compressor is None:
            return

        slot_confidence = path_compressor.get("slot_confidence", None)
        if slot_confidence is not None:
            try:
                value = (
                    float(slot_confidence.detach().cpu().item())
                    if isinstance(slot_confidence, torch.Tensor)
                    else float(slot_confidence)
                )
                writer.add_scalar(f"{phase}/path_compressor/slot_confidence", value, global_step)
            except Exception:
                pass

        slot_usage = path_compressor.get("slot_usage", None)
        if slot_usage is not None:
            try:
                arr = slot_usage.detach().cpu().numpy() if isinstance(slot_usage, torch.Tensor) else np.asarray(slot_usage)
                var = 0.0 if len(arr) == 0 or np.allclose(arr, arr[0]) else float(np.var(arr, ddof=1))
                writer.add_scalar(f"{phase}/path_compressor/slot_usage_var", var, global_step)
            except Exception:
                pass

        for key in ("selected_k", "expected_capacity"):
            value = path_compressor.get(key, None)
            if value is None:
                continue
            try:
                arr = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                writer.add_scalar(
                    f"{phase}/path_compressor/{key}",
                    float(np.asarray(arr).mean()),
                    global_step,
                )
            except Exception:
                pass

        capacity_probs = path_compressor.get("capacity_probs", None)
        candidate_ks = path_compressor.get("candidate_ks", None)
        if capacity_probs is not None and candidate_ks is not None:
            try:
                probs = (
                    capacity_probs.detach().cpu().numpy()
                    if isinstance(capacity_probs, torch.Tensor)
                    else np.asarray(capacity_probs)
                )
                ks = (
                    candidate_ks.detach().cpu().numpy()
                    if isinstance(candidate_ks, torch.Tensor)
                    else np.asarray(candidate_ks)
                )
                mean_probs = np.asarray(probs).mean(axis=0)
                for k, probability in zip(np.asarray(ks), mean_probs):
                    writer.add_scalar(
                        f"{phase}/path_compressor/capacity_prob_k{int(k)}",
                        float(probability),
                        global_step,
                    )
            except Exception:
                pass

    @staticmethod
    def _summarize_slot_capacity(
        selected_k_values,
        capacity_prob_values,
        candidate_ks=(1, 2, 4, 8),
    ):
        if not selected_k_values or not capacity_prob_values:
            return {}

        selected = np.concatenate(
            [np.asarray(value).reshape(-1) for value in selected_k_values],
            axis=0,
        ).astype(float)
        probabilities = np.concatenate(
            [
                np.asarray(value).reshape(-1, len(candidate_ks))
                for value in capacity_prob_values
            ],
            axis=0,
        ).astype(float)
        if selected.size == 0 or probabilities.shape[0] != selected.size:
            return {}
        if not np.isfinite(selected).all() or not np.isfinite(
            probabilities
        ).all():
            return {}

        ks = np.asarray(candidate_ks, dtype=float)
        selected_ratios = np.asarray(
            [np.mean(np.isclose(selected, k)) for k in ks],
            dtype=float,
        )
        expected_k = probabilities @ ks
        positive_ratios = selected_ratios[selected_ratios > 0]
        selection_entropy = 0.0
        if positive_ratios.size > 0:
            selection_entropy = float(
                -np.sum(positive_ratios * np.log(positive_ratios))
                / np.log(float(len(ks)))
            )

        summary = {
            "slot_capacity_samples": int(selected.size),
            "slot_selected_k_mean": float(selected.mean()),
            "slot_expected_k_mean": float(expected_k.mean()),
            "slot_capacity_selection_entropy": selection_entropy,
            "slot_capacity_collapsed": bool(
                selected_ratios.max(initial=0.0) >= 0.95
            ),
        }
        for k, ratio in zip(ks.astype(int), selected_ratios):
            summary[f"slot_k{k}_ratio"] = float(ratio)
        return summary

    @staticmethod
    def _tensor_to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        if value.ndim == 0:
            value = value.reshape(1)
        return value

    @staticmethod
    def _add_vector_columns(row, prefix, array, row_idx):
        if array is None:
            return
        if array.ndim == 1:
            vector = array
        else:
            vector = array[row_idx]
        vector = np.asarray(vector).reshape(-1)
        for col_idx, value in enumerate(vector):
            row[f"{prefix}_{col_idx:04d}"] = float(value)

    def _case_ids_from_indices(self, dataset, indices):
        case_ids = []
        slide_data = getattr(dataset, "slide_data", None)
        if slide_data is not None and "case_id" in getattr(slide_data, "columns", []):
            for idx in indices:
                try:
                    case_ids.append(str(slide_data["case_id"].iloc[int(idx)]))
                except Exception:
                    case_ids.append(f"idx_{int(idx)}")
        else:
            case_ids = [f"idx_{int(idx)}" for idx in indices]
        return case_ids
    def _run_validation_epoch(self, val_loader, model, criterion, device,):
        eval_seed = 100000 + int(self.args.seed) * 1000 + int(self.fold)

        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index or torch.cuda.current_device()]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(eval_seed)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)

            return self.run_epoch(
                val_loader,
                model,
                criterion,
                phase="eval",
            )

    def _collect_tri_rows(self, rows, dataset, out, idx, site_idx, label, event_time, c, risks):
        if not as_bool(getattr(self.args, "export_tri_embeddings", True), True):
            return
        if not isinstance(out, dict):
            return

        idx_np = self._tensor_to_numpy(idx).reshape(-1).astype(int)
        label_np = self._tensor_to_numpy(label).reshape(-1)
        event_np = self._tensor_to_numpy(event_time).reshape(-1)
        c_np = self._tensor_to_numpy(c).reshape(-1)
        risk_np = np.asarray(risks).reshape(-1)
        if site_idx is None:
            site_np = np.full_like(idx_np, -1)
        else:
            site_np = self._tensor_to_numpy(site_idx).reshape(-1).astype(int)
        case_ids = self._case_ids_from_indices(dataset, idx_np)

        path_viz = (out.get("viz") or {}).get("path_compressor") or {}
        selected_k_np = self._tensor_to_numpy(
            path_viz.get("selected_k")
        )
        expected_k_np = self._tensor_to_numpy(
            path_viz.get("expected_capacity")
        )
        capacity_probs_np = self._tensor_to_numpy(
            path_viz.get("capacity_probs")
        )
        candidate_ks_np = self._tensor_to_numpy(
            path_viz.get("candidate_ks")
        )
        if selected_k_np is not None:
            selected_k_np = selected_k_np.reshape(-1)
        if expected_k_np is not None:
            expected_k_np = expected_k_np.reshape(-1)
        if capacity_probs_np is not None:
            capacity_probs_np = capacity_probs_np.reshape(
                len(idx_np),
                -1,
            )
        if candidate_ks_np is not None:
            candidate_ks_np = candidate_ks_np.reshape(-1)

        vector_arrays = {
            "encoded_path": self._tensor_to_numpy(out.get("encoded_path")),
            "encoded_gene": self._tensor_to_numpy(out.get("encoded_gene")),
            "h": self._tensor_to_numpy(out.get("h")),
            "z_b": self._tensor_to_numpy(out.get("z_b")),
            "z_s": self._tensor_to_numpy(out.get("z_s")),
            "z_u": self._tensor_to_numpy(out.get("z_u")),
            "z_u_cond": self._tensor_to_numpy(out.get("z_u_cond")),
            "proto_assign": self._tensor_to_numpy(out.get("proto_assign")),
            "bg_calibration_logits": self._tensor_to_numpy(
                out.get("bg_calibration_logits")
            ),
        }

        for row_idx, sample_idx in enumerate(idx_np):
            row = {
                "case_id": case_ids[row_idx],
                "idx": int(sample_idx),
                "site_idx": int(site_np[row_idx]),
                "label": float(label_np[row_idx]),
                "event_time": float(event_np[row_idx]),
                "censorship": float(c_np[row_idx]),
                "risk": float(risk_np[row_idx]),
                "dataset": getattr(self.args, "dataset", ""),
                "fold": int(self.fold),
                "epoch": int(self.epoch + 1),
            }
            if selected_k_np is not None:
                row["slot_selected_k"] = float(selected_k_np[row_idx])
            if expected_k_np is not None:
                row["slot_expected_k"] = float(expected_k_np[row_idx])
            if (
                capacity_probs_np is not None
                and candidate_ks_np is not None
            ):
                for capacity_index, k in enumerate(candidate_ks_np):
                    row[f"slot_prob_k{int(k)}"] = float(
                        capacity_probs_np[row_idx, capacity_index]
                    )
            for prefix, array in vector_arrays.items():
                self._add_vector_columns(row, prefix, array, row_idx)
            rows.append(row)

    def _write_tri_frame(self, frame, export_dir, stem):
        compression = str(getattr(self.args, "tri_export_compression", "none")).lower()
        chunksize = int(getattr(self.args, "tri_export_chunksize", 1000))
        if compression == "gzip":
            path = os.path.join(export_dir, f"{stem}.csv.gz")
            frame.to_csv(
                path,
                index=False,
                encoding="utf-8",
                compression="gzip",
                chunksize=chunksize,
            )
        else:
            path = os.path.join(export_dir, f"{stem}.csv")
            frame.to_csv(path, index=False, encoding="utf-8", chunksize=chunksize)
        return path

    def _export_tri_artifacts(self, subset):
        if not as_bool(getattr(self.args, "export_tri_embeddings", True), True):
            return

        export_dir = os.path.join(self.results_dir, f"{subset}_fold_{self.fold}", "tri_exports")
        os.makedirs(export_dir, exist_ok=True)

        train_df = pd.DataFrame(self._tri_buffers.get("train", []))
        eval_df = pd.DataFrame(self._tri_buffers.get("eval", []))

        if not train_df.empty:
            train_path = self._write_tri_frame(
                train_df, export_dir, "tri_embeddings_train_best"
            )
            self._log(f"[TriExport] saved train embeddings: {train_path}")

        if not eval_df.empty:
            eval_path = self._write_tri_frame(
                eval_df, export_dir, "tri_embeddings_eval_best"
            )
            self._log(f"[TriExport] saved eval embeddings: {eval_path}")

        if summarize_prototype_usage is not None:
            try:
                usage_df = pd.concat([train_df, eval_df], ignore_index=True)
                usage_path = os.path.join(export_dir, "prototype_usage_best.csv")
                summarize_prototype_usage(usage_df, usage_path)
                self._log(f"[TriExport] saved prototype usage: {usage_path}")
            except Exception as e:
                self._log(f"[TriExport] prototype usage export failed: {e}", "warning")

        if as_bool(getattr(self.args, "run_tri_probe", True), True) and run_site_probes is not None:
            try:
                probe_path = os.path.join(export_dir, "site_probe_best.csv")
                run_site_probes(train_df, eval_df, probe_path)
                self._log(f"[TriExport] saved site probe: {probe_path}")
            except Exception as e:
                self._log(f"[TriExport] site probe failed: {e}", "warning")

    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and "cuda" in state:
            cuda_states = []
            for rng_state in state["cuda"]:
                tensor = torch.as_tensor(
                    rng_state,
                    dtype=torch.uint8,
                    device="cpu",
                ).contiguous()
                cuda_states.append(tensor)
            torch.cuda.set_rng_state_all(cuda_states)

    def _checkpoint_state(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        subset,
        epoch,
    ):
        return {
            "checkpoint_version": 3,
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "criterion_runtime": (
                criterion.get_runtime_state()
                if hasattr(criterion, "get_runtime_state")
                else None
            ),
            "rng_state": self._capture_rng_state(),
            "best_score": float(self.best_score),
            "best_epoch": int(self.best_epoch),
            "optimizer_step_count": int(self.optimizer_step_count),
            "subset": subset,
            "dataset": str(getattr(self.args, "dataset", subset)),
            "fold": int(self.fold),
            "args": pure_config_dict(self.args),
        }

    def _load_checkpoint(
        self,
        path,
        model,
        optimizer,
        scheduler,
        criterion,
        subset,
        device,
    ):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)

        checkpoint_dataset = str(
            checkpoint.get("dataset", checkpoint.get("subset", subset))
        ).lower()
        if checkpoint_dataset != str(subset).lower():
            raise ValueError(
                f"Checkpoint dataset mismatch: checkpoint={checkpoint_dataset}, "
                f"current={subset}."
            )
        checkpoint_fold = checkpoint.get("fold", self.fold)
        if int(checkpoint_fold) != int(self.fold):
            raise ValueError(
                f"Checkpoint fold mismatch: checkpoint={checkpoint_fold}, current={self.fold}."
            )
        saved_args = checkpoint.get("args") or {}
        current_args = pure_config_dict(self.args)
        compatibility_keys = (
            "model",
            "extractor",
            "modal",
            "n_classes",
            "path_slot_num",
            "path_slot_topk_ratio",
            "path_slot_temperature",
            "path_slot_top_k_method",
            "path_slot_train_init_noise",
            "path_slot_adaptive_capacity",
            "path_slot_capacity_temperature",
            "lambda_path_slot_capacity",
            "tri_dim",
            "prog_proto_num",
            "risk_conditioning",
            "num_sites",
            "lambda_tri",
            "nll_alpha",
            "gradient_accumulation_steps",
            "grad_clip_norm",
            "optimizer_warmup_epochs",
        )
        legacy_training_defaults = {
            "lambda_tri": 1.0,
            "nll_alpha": 0.0,
            "gradient_accumulation_steps": 1,
            "grad_clip_norm": 0.0,
            "optimizer_warmup_epochs": 0,
        }
        saved_adaptive_capacity = as_bool(
            saved_args.get("path_slot_adaptive_capacity", False),
            False,
        )
        current_adaptive_capacity = as_bool(
            current_args.get("path_slot_adaptive_capacity", False),
            False,
        )
        capacity_only_keys = {
            "path_slot_capacity_temperature",
            "lambda_path_slot_capacity",
        }
        bool_keys = {
            "path_slot_train_init_noise",
            "path_slot_adaptive_capacity",
        }
        float_keys = {
            "path_slot_topk_ratio",
            "path_slot_temperature",
            "path_slot_capacity_temperature",
            "lambda_path_slot_capacity",
            "lambda_tri",
            "nll_alpha",
            "grad_clip_norm",
        }
        mismatches = []
        for key in compatibility_keys:
            if key == "path_slot_adaptive_capacity":
                saved_value = saved_adaptive_capacity
                current_value = current_adaptive_capacity
                if saved_value != current_value:
                    mismatches.append(
                        f"{key}: checkpoint={saved_value!r}, "
                        f"current={current_value!r}"
                    )
                continue
            if (
                key in capacity_only_keys
                and not (
                    saved_adaptive_capacity
                    or current_adaptive_capacity
                )
            ):
                continue
            if key in legacy_training_defaults:
                saved_raw = saved_args.get(key, legacy_training_defaults[key])
                current_raw = current_args.get(key, legacy_training_defaults[key])
            else:
                if key not in saved_args or key not in current_args:
                    continue
                saved_raw = saved_args[key]
                current_raw = current_args[key]
            if key in bool_keys:
                saved_value = as_bool(saved_raw, False)
                current_value = as_bool(current_raw, False)
            elif key in float_keys:
                saved_value = float(saved_raw)
                current_value = float(current_raw)
            else:
                saved_value = str(saved_raw)
                current_value = str(current_raw)
            if saved_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint={saved_raw!r}, "
                    f"current={current_raw!r}"
                )
        if mismatches:
            raise ValueError(
                "Checkpoint configuration mismatch:\n- " + "\n- ".join(mismatches)
            )

        checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
        if checkpoint_version >= 2:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
            allowed_missing_prefixes = ("bg_calibration_head.",)
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if disallowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Legacy checkpoint is not architecture-compatible. "
                    f"Missing={disallowed_missing}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
            if incompatible.missing_keys:
                self._log(
                    "Legacy checkpoint loaded with optional parameters initialized "
                    f"from current defaults: {incompatible.missing_keys}",
                    "warning",
                )
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if hasattr(criterion, "load_runtime_state"):
            criterion.load_runtime_state(checkpoint.get("criterion_runtime"))

        self.best_score = float(checkpoint.get("best_score", 0.0))
        self.best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        self.optimizer_step_count = int(checkpoint.get("optimizer_step_count", 0))
        self.resume_epoch = int(checkpoint.get("epoch", -1))
        self.best_checkpoint_path = path
        self._restore_rng_state(checkpoint.get("rng_state"))
        return self.resume_epoch + 1

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler, subset):
        self.writer = None
        try:
            return self._learning_impl(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                subset,
            )
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    def _learning_impl(self, model, train_loader, val_loader, criterion, optimizer, scheduler, subset):
        writer_dir = os.path.join(self.results_dir, f"{subset}_fold_{self.fold}")
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = None
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        model = model.to(device)

        self.use_amp = (
            use_cuda and as_bool(getattr(self.args, "use_amp", False), False)
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        start_epoch = 0
        resume_path = str(getattr(self.args, "resume", "") or "").strip()
        if resume_path:
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            self._log(f"=> loading checkpoint '{resume_path}'")
            start_epoch = self._load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                criterion,
                subset,
                device,
            )
            self._log(
                f"=> loaded checkpoint at epoch {self.resume_epoch} "
                f"(best score: {self.best_score:.6f})"
            )

        self.writer = SummaryWriter(writer_dir, flush_secs=15)
        if getattr(self.args, "evaluate", False):
            self.epoch = max(self.resume_epoch, 0)
            _, c_index = self._run_validation_epoch(val_loader, model, criterion, device)
            return c_index, self.epoch

        for epoch in range(start_epoch, self.args.num_epoch):
            self.epoch = epoch
            train_loss, train_c_index = self.run_epoch(
                train_loader, model, criterion, phase="train", optimizer=optimizer
            )
            train_stats = dict(self.last_epoch_stats.get("train", {}))
            val_loss, c_index = self._run_validation_epoch(val_loader, model, criterion, device)
            eval_stats = dict(self.last_epoch_stats.get("eval", {}))

            if scheduler is not None:
                if getattr(self.args, "scheduler", "") == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if c_index >= self.best_score:
                self.best_score = c_index
                self.best_epoch = self.epoch
                checkpoint_state = self._checkpoint_state(
                    model,
                    optimizer,
                    scheduler,
                    criterion,
                    subset,
                    epoch,
                )
                self.best_checkpoint_path = self.save_checkpoint(checkpoint_state)
                self.best_epoch_stats = {
                    "train": train_stats,
                    "eval": eval_stats,
                }

                try:
                    if getattr(self, "_km_buffer", None) is not None:
                        km_dir = os.path.join(self.results_dir, f"{subset}_fold_{self.fold}", "km_exports")
                        os.makedirs(km_dir, exist_ok=True)
                        km_csv = os.path.join(km_dir, f"km_{subset}_fold{self.fold}_best.csv")
                        df = pd.DataFrame(self._km_buffer)
                        df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        df.dropna(subset=["time", "event", "risk"], inplace=True)
                        df.to_csv(km_csv, index=False, encoding="utf-8")
                        self._log(f"[KM] saved best KM csv: {km_csv}")
                except Exception as e:
                    self._log(f"[KM] failed to save KM csv: {e}", "warning")

                try:
                    self._export_tri_artifacts(subset)
                except Exception as e:
                    self._log(f"[TriExport] failed: {e}", "warning")

            if as_bool(getattr(self.args, "save_last_checkpoint", True), True):
                last_state = self._checkpoint_state(
                    model,
                    optimizer,
                    scheduler,
                    criterion,
                    subset,
                    epoch,
                )
                self.last_checkpoint_path = self.save_last_checkpoint(last_state)

            self._log(
                f"best c-index={self.best_score:.4f} at epoch {self.best_epoch}"
            )
        return self.best_score, self.best_epoch

    def run_epoch(self, data_loader, model, criterion, phase="train", optimizer=None):
        if phase == "train":
            model.train()
        else:
            model.eval()
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(self.epoch)
        if hasattr(criterion, "set_phase"):
            criterion.set_phase(phase)
        self._log(f"[{phase}] DataLoader reports {len(data_loader.dataset)} samples")

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        epoch_start_time = time.time()
        sum_loss = 0.0
        risk_list, cens_list, time_list = [], [], []
        idx_list = []
        batch_time_sum = 0.0
        token_sum = 0.0
        token_count = 0
        max_tokens_per_sample = 0
        tri_rows = []
        capacity_selected_values = []
        capacity_prob_values = []
        capacity_candidate_ks = None

        num_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
        num_samples = len(getattr(data_loader, "dataset", [])) if hasattr(data_loader, "dataset") else None
        if num_samples == 0 or num_batches == 0:
            raise ValueError(
                f"Empty {phase} loader for dataset={getattr(self.args, 'dataset', '')}, "
                f"fold={self.fold}."
            )
        processed_samples = 0
        log_every = int(getattr(self.args, "log_every_n_steps", 1))
        amp_dtype = (
            torch.bfloat16
            if str(getattr(self.args, "amp_dtype", "fp16")).lower() == "bf16"
            else torch.float16
        )
        accumulation_steps = (
            int(getattr(self.args, "gradient_accumulation_steps", 1))
            if phase == "train"
            else 1
        )
        if phase == "train" and accumulation_steps > 1 and num_batches is None:
            raise ValueError(
                "gradient_accumulation_steps > 1 requires a DataLoader with __len__."
            )
        grad_clip_norm = float(getattr(self.args, "grad_clip_norm", 0.0))
        optimizer_steps_this_epoch = 0
        grad_norm_values = []
        clipped_step_count = 0

        pbar = tqdm(
            data_loader,
            total=num_batches,
            desc=f"{phase} Epoch {self.epoch + 1}",
            dynamic_ncols=True,
            leave=True,
            position=0 if phase == "train" else 1,
        )

        for batch_idx, batch in enumerate(pbar):
            if phase == "train":
                if optimizer is None:
                    raise ValueError("Training phase requires an optimizer.")
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize()
            batch_start_time = time.time()
            if len(batch) == 8:
                data_WSI, data_omic, label, event_time, c, idx, patient_uid, site_idx = batch
            elif len(batch) == 7:
                data_WSI, data_omic, label, event_time, c, idx, patient_uid = batch
                site_idx = None
            else:
                data_WSI, data_omic, label, event_time, c, idx = batch
                patient_uid = idx
                site_idx = None

            bs = label.size(0) if hasattr(label, "size") else (len(label) if hasattr(label, "__len__") else 0)
            processed_samples += bs

            data_WSI = self._pretrim_wsi_tokens(data_WSI.float(), phase)
            if torch.is_tensor(data_WSI) and data_WSI.dim() == 3:
                tokens_per_sample = int(data_WSI.shape[1])
                token_sum += float(tokens_per_sample * max(bs, 1))
                token_count += max(bs, 1)
                max_tokens_per_sample = max(max_tokens_per_sample, tokens_per_sample)
            device = torch.device("cuda" if use_cuda else "cpu")
            data_WSI = data_WSI.to(device, non_blocking=self.non_blocking)
            data_omic = data_omic.float().to(device, non_blocking=self.non_blocking)
            label = label.float().to(device, non_blocking=self.non_blocking)
            event_time = event_time.float().to(device, non_blocking=self.non_blocking)
            c = c.float().to(device, non_blocking=self.non_blocking)
            if torch.is_tensor(patient_uid):
                patient_uid = patient_uid.to(device, non_blocking=self.non_blocking)
            if site_idx is not None:
                site_idx = site_idx.long().to(device, non_blocking=self.non_blocking)

            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if self.use_amp
                    else nullcontext()
                )
                with amp_context:
                    out = model(
                        x_path=data_WSI,
                        x_omic=data_omic,
                        phase=phase,
                        label=label,
                        c=c,
                        patient_uid=patient_uid,
                        site=site_idx,
                    )
                    target = {
                        "label": label,
                        "event_time": event_time,
                        "c": c,
                        "phase": phase,
                    }
                    if site_idx is not None:
                        target["site"] = site_idx
                    loss, loss_dict = criterion(out, target)

            if isinstance(out, dict):
                path_viz = (
                    (out.get("viz") or {}).get("path_compressor") or {}
                )
                selected_k_value = self._tensor_to_numpy(
                    path_viz.get("selected_k")
                )
                capacity_prob_value = self._tensor_to_numpy(
                    path_viz.get("capacity_probs")
                )
                candidate_k_value = self._tensor_to_numpy(
                    path_viz.get("candidate_ks")
                )
                if (
                    selected_k_value is not None
                    and capacity_prob_value is not None
                    and candidate_k_value is not None
                ):
                    current_candidate_ks = tuple(
                        int(k)
                        for k in candidate_k_value.reshape(-1)
                    )
                    if capacity_candidate_ks is None:
                        capacity_candidate_ks = current_candidate_ks
                    elif capacity_candidate_ks != current_candidate_ks:
                        raise ValueError(
                            "Slot capacity candidates changed within an epoch."
                        )
                    capacity_selected_values.append(
                        selected_k_value.reshape(-1)
                    )
                    capacity_prob_values.append(
                        capacity_prob_value.reshape(
                            -1,
                            len(current_candidate_ks),
                        )
                    )

            if (
                hasattr(self, "writer")
                and self.writer
                and isinstance(out, dict)
                and batch_idx % log_every == 0
            ):
                viz = out.get("viz", {})
                try:
                    total_batches = num_batches if num_batches is not None else len(data_loader)
                    global_step = int(self.epoch) * int(total_batches) + int(batch_idx)
                except Exception:
                    global_step = self.epoch
                self._log_path_compressor(self.writer, phase, viz, global_step)
                for key, value in loss_dict.items():
                    try:
                        scalar = (
                            float(value.detach().cpu().item())
                            if isinstance(value, torch.Tensor) and value.numel() == 1
                            else float(value)
                        )
                        self.writer.add_scalar(f"{phase}/loss_component/{key}", scalar, global_step)
                    except Exception:
                        pass

            if phase == "train":
                group_start = (batch_idx // accumulation_steps) * accumulation_steps
                group_size = (
                    min(accumulation_steps, num_batches - group_start)
                    if num_batches is not None
                    else accumulation_steps
                )
                backward_loss = loss / float(group_size)
                if self.use_amp:
                    self.scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
                should_step = (
                    (batch_idx + 1) % accumulation_steps == 0
                    or (
                        num_batches is not None
                        and batch_idx + 1 == num_batches
                    )
                )
                if should_step:
                    grad_norm, was_clipped = self._take_optimizer_step(
                        optimizer,
                        grad_clip_norm,
                    )
                    optimizer_steps_this_epoch += 1
                    grad_norm_values.append(grad_norm)
                    clipped_step_count += int(was_clipped)
                    if hasattr(self, "writer") and self.writer:
                        step = int(self.optimizer_step_count)
                        self.writer.add_scalar(
                            "train/optimizer/grad_norm",
                            grad_norm,
                            step,
                        )
                        self.writer.add_scalar(
                            "train/optimizer/grad_clipped",
                            int(was_clipped),
                            step,
                        )
                        self.writer.add_scalar(
                            "train/optimizer/lr",
                            optimizer.param_groups[-1]["lr"],
                            step,
                        )
                        self.writer.add_scalar(
                            "train/optimizer/step",
                            step,
                            step,
                        )
                lr_val = optimizer.param_groups[-1]["lr"]
            else:
                lr_val = 0.0

            sum_loss += loss.item()
            avg_loss = sum_loss / (batch_idx + 1)

            risks = -torch.sum(out["S"][-1], dim=1).detach().cpu().numpy()
            risk_list.append(risks)
            cens_list.append(c.detach().cpu().numpy().reshape(-1))
            time_list.append(event_time.detach().cpu().numpy().reshape(-1))
            self._collect_tri_rows(
                tri_rows,
                getattr(data_loader, "dataset", None),
                out,
                idx,
                site_idx,
                label,
                event_time,
                c,
                risks,
            )

            if torch.is_tensor(idx):
                idx_list.append(idx.detach().cpu().numpy().reshape(-1))
            else:
                idx_list.append(np.asarray(idx).reshape(-1))

            batch_prog = f"[{batch_idx}/{num_batches}]" if num_batches is not None else f"[{batch_idx}]"
            sample_prog = (
                f"({processed_samples}/{num_samples} samples)"
                if num_samples is not None
                else f"({processed_samples} samples)"
            )
            pbar.set_postfix_str(
                f"{batch_prog} {sample_prog} | LR: {lr_val:.1e} | loss: {loss.item():.4f} | avg: {avg_loss:.4f}"
            )
            if use_cuda:
                torch.cuda.synchronize()
            batch_time_sum += time.time() - batch_start_time

        if use_cuda:
            torch.cuda.synchronize()
        epoch_time_sec = time.time() - epoch_start_time
        mean_loss = sum_loss / max(1, (num_batches if num_batches is not None else (batch_idx + 1)))
        all_risk_scores = np.concatenate(risk_list, axis=0)
        all_censorships = np.concatenate(cens_list, axis=0)
        all_event_times = np.concatenate(time_list, axis=0)
        events = (1 - all_censorships).astype(bool)
        if int(events.sum()) == 0:
            raise ValueError(
                f"No observed events in {phase} data for dataset="
                f"{getattr(self.args, 'dataset', '')}, fold={self.fold}; C-index is undefined."
            )
        try:
            c_index = concordance_index_censored(
                events,
                all_event_times,
                all_risk_scores,
                tied_tol=1e-08,
            )[0]
        except Exception as exc:
            raise ValueError(
                f"C-index failed for {phase} data, dataset="
                f"{getattr(self.args, 'dataset', '')}, fold={self.fold}: {exc}"
            ) from exc

        tqdm.write(f"[{phase}] Epoch {self.epoch + 1} | loss: {mean_loss:.4f}, c_index: {c_index:.4f}")

        completed_batches = num_batches if num_batches is not None else (batch_idx + 1)
        completed_batches = max(1, int(completed_batches))
        stats = {
            "phase": phase,
            "epoch": int(self.epoch),
            "epoch_time_sec": float(epoch_time_sec),
            "samples_per_sec": float(processed_samples / epoch_time_sec) if epoch_time_sec > 0 else 0.0,
            "batches_per_sec": float(completed_batches / epoch_time_sec) if epoch_time_sec > 0 else 0.0,
            "avg_batch_time_sec": float(batch_time_sum / completed_batches),
            "avg_tokens_per_sample": float(token_sum / token_count) if token_count > 0 else 0.0,
            "max_tokens_per_sample": int(max_tokens_per_sample),
            "peak_cuda_allocated_mb": 0.0,
            "peak_cuda_reserved_mb": 0.0,
        }
        if phase == "train":
            stats.update(
                {
                    "optimizer_steps": int(optimizer_steps_this_epoch),
                    "grad_norm_mean": (
                        float(np.mean(grad_norm_values))
                        if grad_norm_values
                        else 0.0
                    ),
                    "grad_norm_max": (
                        float(np.max(grad_norm_values))
                        if grad_norm_values
                        else 0.0
                    ),
                    "grad_clipped_fraction": (
                        float(clipped_step_count / optimizer_steps_this_epoch)
                        if optimizer_steps_this_epoch > 0
                        else 0.0
                    ),
                }
            )
        if use_cuda:
            stats["peak_cuda_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            stats["peak_cuda_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        capacity_stats = self._summarize_slot_capacity(
            capacity_selected_values,
            capacity_prob_values,
            candidate_ks=(
                capacity_candidate_ks
                if capacity_candidate_ks is not None
                else (1, 2, 4, 8)
            ),
        )
        stats.update(capacity_stats)
        if capacity_stats.get("slot_capacity_collapsed", False):
            ratios = {
                k: capacity_stats.get(f"slot_k{k}_ratio", 0.0)
                for k in (1, 2, 4, 8)
            }
            dominant_k = max(ratios, key=ratios.get)
            self._log(
                "[SlotCapacity] selection collapsed: "
                f"K={dominant_k} ratio={ratios[dominant_k]:.3f} "
                f"during {phase} epoch {self.epoch + 1}.",
                "warning",
            )
        self.last_epoch_stats[phase] = stats
        self._tri_buffers[phase] = tri_rows
        dataset = getattr(data_loader, "dataset", None)
        runtime_wsi = getattr(dataset, "wsi_load_stats", None)
        if runtime_wsi is not None:
            runtime_path = os.path.join(
                self.results_dir,
                f"{getattr(self.args, 'dataset', '')}_fold_{self.fold}",
                "wsi_runtime_stats.json",
            )
            runtime_payload = {}
            if os.path.isfile(runtime_path):
                try:
                    with open(runtime_path, "r", encoding="utf-8") as fp:
                        runtime_payload = json.load(fp)
                except Exception:
                    runtime_payload = {}
            runtime_payload[phase] = pure_config_dict(runtime_wsi)
            tmp_path = runtime_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(runtime_payload, fp, ensure_ascii=False, indent=2)
            os.replace(tmp_path, runtime_path)

        if hasattr(self, "writer") and self.writer:
            self.writer.add_scalar(f"{phase}/loss", mean_loss, self.epoch)
            self.writer.add_scalar(f"{phase}/c_index", c_index, self.epoch)
            self.writer.add_scalar(f"{phase}/epoch_time_sec", stats["epoch_time_sec"], self.epoch)
            self.writer.add_scalar(f"{phase}/samples_per_sec", stats["samples_per_sec"], self.epoch)
            self.writer.add_scalar(f"{phase}/batches_per_sec", stats["batches_per_sec"], self.epoch)
            self.writer.add_scalar(f"{phase}/avg_batch_time_sec", stats["avg_batch_time_sec"], self.epoch)
            self.writer.add_scalar(f"{phase}/avg_tokens_per_sample", stats["avg_tokens_per_sample"], self.epoch)
            self.writer.add_scalar(f"{phase}/max_tokens_per_sample", stats["max_tokens_per_sample"], self.epoch)
            self.writer.add_scalar(f"{phase}/peak_cuda_allocated_mb", stats["peak_cuda_allocated_mb"], self.epoch)
            self.writer.add_scalar(f"{phase}/peak_cuda_reserved_mb", stats["peak_cuda_reserved_mb"], self.epoch)
            if phase == "train":
                self.writer.add_scalar(
                    "train/optimizer_steps",
                    stats["optimizer_steps"],
                    self.epoch,
                )
                self.writer.add_scalar(
                    "train/grad_norm_mean",
                    stats["grad_norm_mean"],
                    self.epoch,
                )
                self.writer.add_scalar(
                    "train/grad_norm_max",
                    stats["grad_norm_max"],
                    self.epoch,
                )
                self.writer.add_scalar(
                    "train/grad_clipped_fraction",
                    stats["grad_clipped_fraction"],
                    self.epoch,
                )
            for key, value in capacity_stats.items():
                if key == "slot_capacity_samples":
                    continue
                self.writer.add_scalar(
                    f"{phase}/path_compressor/{key}",
                    float(value),
                    self.epoch,
                )

        if phase == "eval":
            try:
                all_idx = np.concatenate(idx_list, axis=0)
                ds = getattr(data_loader, "dataset", None)
                if hasattr(ds, "slide_data") and ("case_id" in getattr(ds.slide_data, "columns", [])):
                    case_ids = ds.slide_data["case_id"].iloc[all_idx].astype(str).tolist()
                elif hasattr(ds, "case_ids"):
                    case_ids = np.asarray(ds.case_ids)[all_idx].astype(str).tolist()
                else:
                    case_ids = [f"idx_{int(i)}" for i in all_idx]

                self._km_buffer = {
                    "case_id": case_ids,
                    "time": all_event_times,
                    "event": (1 - all_censorships).astype(int),
                    "risk": all_risk_scores,
                    "dataset": getattr(self.args, "dataset", ""),
                    "fold": int(self.fold),
                    "epoch": int(self.epoch + 1),
                }
            except Exception as e:
                self._km_buffer = None
                self._log(f"[KM] cache failed: {e}", "warning")

        return mean_loss, c_index

    @staticmethod
    def _atomic_torch_save(state, path):
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

    def save_checkpoint(self, state):
        ckpt_dir = os.path.join(self.results_dir, f"{state['subset']}_fold_{self.fold}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.filename_best is not None:
            try:
                os.remove(self.filename_best)
            except Exception:
                pass

        self.filename_best = os.path.join(
            ckpt_dir,
            "model_best_{score:.4f}_epoch{epoch}_fold{fold}.pth.tar".format(
                score=state["best_score"],
                epoch=state["epoch"],
                fold=self.fold,
            ),
        )
        self._log(f"save best model {self.filename_best}")
        self._atomic_torch_save(state, self.filename_best)
        return self.filename_best

    def save_last_checkpoint(self, state):
        ckpt_dir = os.path.join(self.results_dir, f"{state['subset']}_fold_{self.fold}")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"checkpoint_last_fold{self.fold}.pth.tar")
        self._atomic_torch_save(state, path)
        return path
