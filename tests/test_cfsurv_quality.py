from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

# These warnings originate from optional upstream dependencies.  Keep the test
# output focused while leaving all other warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"distutils Version classes are deprecated\..*",
    category=DeprecationWarning,
    module=r"thop\.profile",
)

import torch
from torch.utils.data import DataLoader, Dataset

from main import (
    _collect_path_compressor_config,
    _effective_config_lines,
    _profile_model_flops,
    _write_efficiency_row,
)
from models.CFSurv import network_20260414_CFSurv as net
from models.CFSurv.path_compressor import (
    CFSurvPathSlotCompressor,
    CFSurvSlotAttention,
)
from utils.config_utils import strip_slide_suffix
from utils.dataset_survival_patch_only import Generic_MIL_Survival_Dataset
from utils.engine import Engine
from utils.loss_factory_new import Loss_factory, nll_loss


@contextlib.contextmanager
def quiet_expected_output():
    """Hide expected training logs, but replay them if the wrapped call fails."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield
    except Exception:
        sys.stdout.write(stdout.getvalue())
        sys.stderr.write(stderr.getvalue())
        raise


def make_args(**overrides):
    values = {key: spec["default"] for key, spec in net.custom_config["customized"].items()}
    values.update(net.custom_config["base"])
    values.update(
        {
            "dataset": "ucec",
            "extractor": "resnet50",
            "n_classes": 4,
            "seed": 0,
            "num_sites": 5,
            "loss": "nllsurv,tri",
            "resume": "",
            "evaluate": False,
            "save_last_checkpoint": True,
            "non_blocking": False,
            "use_amp": False,
            "amp_dtype": "fp16",
            "num_epoch": 1,
            "scheduler": "cosine",
            "adaptive_sampling": False,
            "max_tokens": 4096,
            "min_tokens": 6,
            "log_every_n_steps": 1,
            "tri_export_compression": "none",
            "tri_export_chunksize": 100,
            "export_tri_embeddings": True,
            "run_tri_probe": False,
        }
    )
    values.update(overrides)
    return Namespace(**values)


class ModelQualityTests(unittest.TestCase):
    def test_adaptive_sampling_is_disabled_by_default(self):
        self.assertIs(net.CFSurv_DEFAULT_CONFIG["adaptive_sampling"], False)

    def test_slot_config_log_matches_effective_capacity_mode(self):
        adaptive_summary = net.format_path_slot_config(make_args())
        self.assertIn("path_slot_mode=adaptive", adaptive_summary)
        self.assertIn("candidate_ks=[1,2,4,8]", adaptive_summary)
        self.assertNotIn("slot_topk_ratio", adaptive_summary)

        fixed_summary = net.format_path_slot_config(
            make_args(
                path_slot_adaptive_capacity=False,
                path_slot_num=8,
                path_slot_topk_ratio=0.25,
            )
        )
        self.assertIn("path_slot_mode=fixed", fixed_summary)
        self.assertIn("slot_topk_ratio=0.25", fixed_summary)
        self.assertIn("effective_keep_k=2", fixed_summary)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            net.get_auto_config(dataset_name="blca")
        self.assertNotIn("Slot top-k ratio", output.getvalue())

    def test_effective_config_output_and_snapshot_are_mode_aware(self):
        args = make_args(
            dataset="blca",
            data_root_dir=r"E:\Resnet50_Features_cyp",
            fold="0,1,2,3,4",
            site_bg_stop_grad=False,
            lambda_site_bg=0.2,
            lambda_site_adv=0.05,
            lr=1e-4,
        )
        lines = _effective_config_lines(args, net)
        self.assertEqual(len(lines), 4)
        self.assertIn(
            r"data_root_dir=E:\Resnet50_Features_cyp",
            lines[0],
        )
        self.assertIn(
            r"dataset_data_dir=E:\Resnet50_Features_cyp\BLCA",
            lines[0],
        )
        self.assertIn("lr=0.0001", lines[1])
        self.assertIn("path_slot_mode=adaptive", lines[2])
        self.assertNotIn("slot_topk_ratio", lines[2])
        self.assertIn("site_bg_stop_grad=False", lines[3])
        self.assertIn("lambda_site_bg=0.2", lines[3])
        self.assertIn("lambda_site_adv=0.05", lines[3])

        adaptive_snapshot = _collect_path_compressor_config(
            net.CFSurv(args)
        )["path_compressor"]
        self.assertEqual(adaptive_snapshot["capacity_mode"], "adaptive")
        self.assertEqual(adaptive_snapshot["candidate_ks"], [1, 2, 4, 8])
        self.assertNotIn("keep_k", adaptive_snapshot)
        self.assertNotIn("topk_ratio", adaptive_snapshot)

        fixed_snapshot = _collect_path_compressor_config(
            net.CFSurv(
                make_args(path_slot_adaptive_capacity=False)
            )
        )["path_compressor"]
        self.assertEqual(fixed_snapshot["capacity_mode"], "fixed")
        self.assertEqual(fixed_snapshot["keep_k"], 2)
        self.assertEqual(fixed_snapshot["topk_ratio"], 0.25)

    def test_flops_profile_preserves_torch_rng(self):
        class TinyProfileModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.path = torch.nn.Linear(4, 2)
                self.omic = torch.nn.Linear(3, 2)

            def forward(self, x_path, x_omic, phase="eval"):
                logits = self.path(x_path.mean(dim=1)) + self.omic(x_omic)
                return {"hazards": [torch.sigmoid(logits)]}

        model = TinyProfileModel().train()
        before = torch.get_rng_state().clone()
        with quiet_expected_output():
            macs_g, flops_g = _profile_model_flops(model, 5, 4, 3)
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(model.training)
        self.assertGreater(macs_g, 0.0)
        self.assertEqual(flops_g, 2.0 * macs_g)

    def test_eval_is_reproducible_with_fixed_slot_seed(self):
        model = net.CFSurv(make_args()).eval()
        path = torch.randn(1, 24, 1024)
        omic = torch.randn(1, 150)

        with torch.no_grad():
            torch.manual_seed(23)
            first = model(path, omic, "eval")["hazards"][-1]
            torch.manual_seed(23)
            second = model(path, omic, "eval")["hazards"][-1]

        self.assertTrue(torch.equal(first, second))

    def test_eval_does_not_mutate_prototype_queue(self):
        args = make_args()
        model = net.CFSurv(args).train()
        output = model(torch.randn(1, 24, 1024), torch.randn(1, 150), "train")
        target = {
            "label": torch.tensor([1.0]),
            "event_time": torch.tensor([12.0]),
            "c": torch.tensor([0.0]),
            "site": torch.tensor([1]),
        }
        criterion = Loss_factory(args)
        criterion.set_phase("train")
        loss, _ = criterion(output, target)
        loss.backward()
        before = criterion.loss_fns["tri"].proto_usage_queue.clone()
        criterion.set_phase("eval")
        criterion(output, target)
        self.assertTrue(torch.equal(before, criterion.loss_fns["tri"].proto_usage_queue))

    def test_checkpoint_roundtrip_and_legacy_load(self):
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp, quiet_expected_output():
            model = net.CFSurv(args)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            criterion = Loss_factory(args)
            engine = Engine(args, tmp, 0)
            engine.scaler = torch.amp.GradScaler("cuda", enabled=False)
            engine.best_score = 0.61
            engine.best_epoch = 1
            state = engine._checkpoint_state(
                model, optimizer, scheduler, criterion, "ucec", 1
            )
            path = Path(tmp) / "checkpoint.pth.tar"
            engine._atomic_torch_save(state, str(path))

            restored = net.CFSurv(args)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=3
            )
            restored_criterion = Loss_factory(args)
            restored_engine = Engine(args, tmp, 0)
            restored_engine.scaler = torch.amp.GradScaler("cuda", enabled=False)
            start_epoch = restored_engine._load_checkpoint(
                str(path),
                restored,
                restored_optimizer,
                restored_scheduler,
                restored_criterion,
                "ucec",
                torch.device("cpu"),
            )
            self.assertEqual(start_epoch, 2)
            self.assertEqual(restored_engine.best_score, 0.61)

            fixed_args = make_args(path_slot_adaptive_capacity=False)
            fixed_model = net.CFSurv(fixed_args)
            legacy_path = Path(tmp) / "legacy.pth.tar"
            legacy_state = {
                key: value
                for key, value in fixed_model.state_dict().items()
                if not key.startswith("bg_calibration_head.")
            }
            torch.save(
                {
                    "epoch": 0,
                    "state_dict": legacy_state,
                    "best_score": 0.5,
                    "subset": "ucec",
                },
                legacy_path,
            )
            legacy_engine = Engine(fixed_args, tmp, 0)
            legacy_engine.scaler = torch.amp.GradScaler("cuda", enabled=False)
            legacy_start = legacy_engine._load_checkpoint(
                str(legacy_path),
                net.CFSurv(fixed_args),
                None,
                None,
                Loss_factory(fixed_args),
                "ucec",
                torch.device("cpu"),
            )
            self.assertEqual(legacy_start, 1)

    def test_optional_sampling_and_gzip_export(self):
        args = make_args(
            adaptive_sampling=True,
            max_tokens=5,
            min_tokens=3,
            tri_export_compression="gzip",
            tri_export_chunksize=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(args, tmp, 0)
            tokens = torch.arange(10 * 4, dtype=torch.float32).view(1, 10, 4)
            first = engine._pretrim_wsi_tokens(tokens, "eval")
            second = engine._pretrim_wsi_tokens(tokens, "eval")
            self.assertEqual(tuple(first.shape), (1, 5, 4))
            self.assertTrue(torch.equal(first, second))
            path = engine._write_tri_frame(
                pd.DataFrame({"value": [1, 2, 3]}),
                tmp,
                "tri_test",
            )
            self.assertTrue(path.endswith(".csv.gz"))
            self.assertTrue(Path(path).exists())

    def test_efficiency_summary_migrates_existing_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "efficiency_summary.csv"
            path.write_text(
                "dataset,fold,best_c_index\nucec,0,0.600000\n",
                encoding="utf-8",
            )
            _write_efficiency_row(
                str(path),
                {
                    "dataset": "luad",
                    "fold": 1,
                    "best_c_index": 0.7,
                    "best_eval_slot_selected_k_mean": 2.0,
                },
            )
            frame = pd.read_csv(path)

        self.assertEqual(frame["dataset"].tolist(), ["ucec", "luad"])
        self.assertTrue(
            pd.isna(
                frame.loc[
                    0,
                    "best_eval_slot_selected_k_mean",
                ]
            )
        )
        self.assertEqual(
            frame.loc[1, "best_eval_slot_selected_k_mean"],
            2.0,
        )

    def test_one_epoch_generates_core_artifacts(self):
        class TinyDataset(Dataset):
            def __init__(self):
                generator = torch.Generator().manual_seed(3)
                self.path = torch.randn(4, 18, 1024, generator=generator)
                self.omic = torch.randn(4, 150, generator=generator)
                self.slide_data = pd.DataFrame(
                    {"case_id": [f"TINY{i}" for i in range(4)]}
                )

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return (
                    self.path[index],
                    self.omic[index],
                    torch.tensor(index % 4),
                    torch.tensor(float(4 - index)),
                    torch.tensor(0.0),
                    torch.tensor(index),
                    torch.tensor(index),
                    torch.tensor(index % 2),
                )

        args = make_args(num_sites=2)
        with tempfile.TemporaryDirectory() as tmp, quiet_expected_output():
            dataset = TinyDataset()
            train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            model = net.CFSurv(args)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            criterion = Loss_factory(args)
            engine = Engine(args, tmp, 0)
            score, epoch = engine.learning(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                "ucec",
            )
            self.assertTrue(np.isfinite(score))
            self.assertEqual(epoch, 0)
            fold_dir = Path(tmp) / "ucec_fold_0"
            self.assertTrue(list(fold_dir.glob("model_best_*.pth.tar")))
            self.assertTrue((fold_dir / "checkpoint_last_fold0.pth.tar").exists())
            self.assertTrue(
                (fold_dir / "km_exports" / "km_ucec_fold0_best.csv").exists()
            )
            self.assertTrue(
                (
                    fold_dir
                    / "tri_exports"
                    / "tri_embeddings_eval_best.csv"
                ).exists()
            )
            efficiency = Path(tmp) / "efficiency_summary.csv"
            _write_efficiency_row(
                str(efficiency),
                {"dataset": "ucec", "fold": 0, "best_c_index": score},
            )
            self.assertTrue(efficiency.exists())

            resume_path = fold_dir / "checkpoint_last_fold0.pth.tar"
            eval_args = make_args(
                num_sites=2,
                evaluate=True,
                resume=str(resume_path),
            )
            eval_model = net.CFSurv(eval_args)
            eval_optimizer = torch.optim.AdamW(eval_model.parameters(), lr=1e-4)
            eval_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                eval_optimizer, T_max=1
            )
            eval_score, eval_epoch = Engine(eval_args, tmp, 0).learning(
                eval_model,
                train_loader,
                val_loader,
                Loss_factory(eval_args),
                eval_optimizer,
                eval_scheduler,
                "ucec",
            )
            self.assertAlmostEqual(eval_score, score, places=7)
            self.assertEqual(eval_epoch, 0)


class DatasetQualityTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path):
        rows = []
        times = [1, 2, 3, 4, 5, 6, 7, 8, 1000, 2000]
        for index, survival_time in enumerate(times):
            row = {
                "case_id": f"C{index}",
                "slide_id": f"S{index}.svs",
                "site": f"T{index % 2}",
                "oncotree_code": "TEST",
                "survival_months": survival_time,
                "censorship": 0,
                "train": int(index < 8),
            }
            row.update({f"g{feature}": float(index + feature) for feature in range(150)})
            rows.append(row)
        csv_path = root / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        split_path = root / "split.csv"
        pd.DataFrame(
            {
                "train": pd.Series([f"C{i}" for i in range(8)]),
                "val": pd.Series(["C8", "C9"]),
            }
        ).to_csv(split_path, index=False)
        return csv_path, split_path

    def test_training_only_bins_and_modal_shapes(self):
        with tempfile.TemporaryDirectory() as tmp, quiet_expected_output():
            root = Path(tmp)
            csv_path, split_path = self._write_fixture(root)
            common = {
                "csv_path": str(csv_path),
                "apply_sig": False,
                "data_dir": str(root),
                "shuffle": False,
                "seed": 0,
                "patient_strat": False,
                "n_bins": 4,
                "label_col": "survival_months",
                "args": Namespace(extractor="resnet50"),
            }
            path_dataset = Generic_MIL_Survival_Dataset(modal="path", **common)
            train, val = path_dataset.return_splits(
                from_id=False,
                csv_path=str(split_path),
                set_name="ucec",
                extractor="resnet50",
            )
            self.assertFalse(
                {"C8", "C9"} & set(path_dataset.last_split_audit["bin_fit_case_ids"])
            )
            self.assertEqual(train[0][1].numel(), 150)
            self.assertTrue(np.isneginf(path_dataset.bins[0]))
            self.assertTrue(np.isposinf(path_dataset.bins[-1]))

            omic_dataset = Generic_MIL_Survival_Dataset(modal="omic", **common)
            omic_train, _ = omic_dataset.return_splits(
                from_id=False,
                csv_path=str(split_path),
                set_name="ucec",
                extractor="resnet50",
            )
            path_features, omic_features = omic_train[0][:2]
            self.assertEqual(tuple(path_features.shape), (6, 1024))
            self.assertEqual(omic_features.numel(), 150)
            self.assertEqual(len(train), 8)
            self.assertEqual(len(val), 2)

    def test_slide_suffix_is_exact(self):
        self.assertEqual(strip_slide_suffix("case.svs"), "case")
        self.assertEqual(strip_slide_suffix("glass"), "glass")


class SlotAttentionInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = CFSurvSlotAttention(
            num_slots=8,
            dim=32,
            heads=4,
            dim_head=8,
            iters=3,
            train_init_noise=True,
        )
        self.inputs = torch.randn(2, 24, 32)
        self.module.eval()

    def test_eval_slots_are_not_symmetric(self):
        torch.manual_seed(17)
        slots, attn = self.module(self.inputs, return_attn=True)

        slot_difference = (
            slots[:, 1:] - slots[:, :1]
        ).abs().max().item()

        attention_difference = (
            attn[:, :, 1:] - attn[:, :, :1]
        ).abs().max().item()

        self.assertGreater(slot_difference, 1e-5)
        self.assertGreater(attention_difference, 1e-7)

    def test_eval_initialization_is_seed_reproducible(self):
        torch.manual_seed(17)
        slots_first, attn_first = self.module(
            self.inputs,
            return_attn=True,
        )

        torch.manual_seed(17)
        slots_second, attn_second = self.module(
            self.inputs,
            return_attn=True,
        )

        self.assertTrue(torch.equal(slots_first, slots_second))
        self.assertTrue(torch.equal(attn_first, attn_second))

    def test_different_seeds_produce_different_slots(self):
        torch.manual_seed(17)
        slots_first = self.module(self.inputs)

        torch.manual_seed(18)
        slots_second = self.module(self.inputs)

        self.assertFalse(torch.allclose(slots_first, slots_second))


class AdaptiveSlotCapacityTests(unittest.TestCase):
    @staticmethod
    def _make_compressor(
        top_k_method="parallel_topk_st",
        temperature=0.5,
        adaptive_capacity=True,
    ):
        return CFSurvPathSlotCompressor(
            dim=16,
            num_slots=8,
            num_classes=4,
            iters=1,
            heads=2,
            dim_head=8,
            temperature=temperature,
            topk_ratio=0.25,
            top_k_method=top_k_method,
            train_init_noise=True,
            adaptive_capacity=adaptive_capacity,
            capacity_temperature=1.0,
        )

    def test_adaptive_capacity_is_enabled_by_default(self):
        args = make_args()
        model = net.CFSurv(args)

        self.assertIs(
            net.CFSurv_DEFAULT_CONFIG["path_slot_adaptive_capacity"],
            True,
        )
        self.assertTrue(model.path_compressor.adaptive_capacity)
        self.assertTrue(hasattr(model.path_compressor, "slot_decoder"))
        self.assertTrue(hasattr(model.path_compressor, "capacity_head"))

    def test_adaptive_capacity_requires_eight_slots(self):
        with self.assertRaisesRegex(ValueError, "path_slot_num=8"):
            CFSurvPathSlotCompressor(
                dim=16,
                num_slots=4,
                num_classes=4,
                adaptive_capacity=True,
            )

    def test_direct_compressor_rejects_invalid_numeric_configuration(self):
        invalid_overrides = (
            {"temperature": 0.0},
            {"temperature": float("nan")},
            {"capacity_temperature": float("inf")},
            {"num_classes": 0},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    CFSurvPathSlotCompressor(
                        dim=16,
                        num_slots=8,
                        adaptive_capacity=True,
                        **overrides,
                    )

    def test_stable_gate_normalizes_extreme_low_temperature_masks(self):
        compressor = self._make_compressor(
            temperature=0.01,
        )
        keep_logits = torch.tensor(
            [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        fixed_mask = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        candidate_masks = torch.tensor(
            [[
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]]
        )

        fixed_gate = compressor._stable_masked_slot_gate(
            keep_logits,
            fixed_mask,
        )
        candidate_gates = compressor._stable_masked_slot_gate(
            keep_logits,
            candidate_masks,
        )

        self.assertTrue(torch.isfinite(fixed_gate).all())
        self.assertTrue(torch.isfinite(candidate_gates).all())
        self.assertTrue(
            torch.allclose(fixed_gate.sum(dim=-1), torch.ones(1))
        )
        self.assertTrue(
            torch.allclose(
                candidate_gates.sum(dim=-1),
                torch.ones(1, 4),
            )
        )

    def test_fixed_mode_forward_uses_stable_default_temperature_gate(self):
        tokens = torch.randn(2, 12, 16)
        for method in ("gumbel_topk_st", "parallel_topk_st"):
            with self.subTest(method=method):
                compressor = self._make_compressor(
                    top_k_method=method,
                    temperature=0.01,
                    adaptive_capacity=False,
                ).train()
                torch.manual_seed(37)
                path_vec, viz, slot_aux = compressor(tokens)

                self.assertIsNone(slot_aux)
                self.assertTrue(torch.isfinite(path_vec).all())
                self.assertTrue(torch.isfinite(viz["slot_gate"]).all())
                self.assertTrue(
                    torch.allclose(
                        viz["slot_gate"].sum(dim=-1),
                        torch.ones(2),
                    )
                )

    def test_stable_gate_matches_slot_spe_formula_in_regular_range(self):
        compressor = self._make_compressor(temperature=0.5)
        keep_logits = torch.tensor(
            [[0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]]
        )
        keep_mask = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        )

        slot_spe_gate = torch.softmax(
            keep_logits / compressor.temperature,
            dim=-1,
        )
        slot_spe_gate = slot_spe_gate * keep_mask
        slot_spe_gate = slot_spe_gate / (
            slot_spe_gate.sum(dim=-1, keepdim=True) + 1e-8
        )
        stable_gate = compressor._stable_masked_slot_gate(
            keep_logits,
            keep_mask,
        )

        self.assertTrue(
            torch.allclose(stable_gate, slot_spe_gate, atol=1e-7)
        )

    def test_default_temperature_gate_gradients_are_finite(self):
        positions = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        for method in ("gumbel_topk_st", "parallel_topk_st"):
            with self.subTest(method=method):
                compressor = self._make_compressor(
                    top_k_method=method,
                    temperature=0.01,
                ).train()
                keep_logits = torch.tensor(
                    [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    requires_grad=True,
                )
                torch.manual_seed(5)
                masks = compressor._adaptive_keep_masks(keep_logits)
                gates = compressor._stable_masked_slot_gate(
                    keep_logits,
                    masks,
                )
                loss = (gates * positions).sum()
                loss.backward()

                self.assertTrue(torch.isfinite(gates).all())
                self.assertTrue(
                    torch.allclose(
                        gates.sum(dim=-1),
                        torch.ones(1, 4),
                    )
                )
                self.assertIsNotNone(keep_logits.grad)
                self.assertTrue(torch.isfinite(keep_logits.grad).all())
                self.assertGreater(keep_logits.grad.abs().sum().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_stable_gate_is_finite_under_cuda_fp16_autocast(self):
        compressor = self._make_compressor(
            temperature=0.01,
        ).cuda().train()
        keep_logits = torch.tensor(
            [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float16,
        )
        keep_mask = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            device="cuda",
            dtype=torch.float16,
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            gate = compressor._stable_masked_slot_gate(
                keep_logits,
                keep_mask,
            )

        self.assertEqual(gate.dtype, torch.float16)
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(
            torch.allclose(
                gate.sum(dim=-1).float(),
                torch.ones(1, device="cuda"),
            )
        )

        compressor.zero_grad(set_to_none=True)
        tokens = torch.randn(2, 12, 16, device="cuda")
        torch.manual_seed(43)
        torch.cuda.manual_seed_all(43)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            path_vec, viz, slot_aux = compressor(tokens)
            loss = (
                path_vec.float().square().mean()
                + slot_aux["survival_logits"].square().mean()
                + slot_aux["expected_capacity_ratio"].mean()
            )
        loss.backward()

        self.assertTrue(torch.isfinite(viz["slot_gate"]).all())
        self.assertTrue(
            torch.allclose(
                viz["slot_gate"].float().sum(dim=-1),
                torch.ones(2, device="cuda"),
                atol=1e-3,
            )
        )
        for parameter in (
            compressor.slot_decoder.weight,
            compressor.slot_score[-1].weight,
            compressor.capacity_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_candidate_masks_are_nested_for_both_slot_spe_methods(self):
        keep_logits = torch.tensor(
            [[0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6]]
        )

        for method in ("gumbel_topk_st", "parallel_topk_st"):
            with self.subTest(method=method):
                compressor = self._make_compressor(method).train()
                torch.manual_seed(29)
                masks = compressor._adaptive_keep_masks(keep_logits).detach()

                expected_counts = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
                self.assertTrue(
                    torch.allclose(masks.sum(dim=-1), expected_counts)
                )
                for index in range(masks.size(1) - 1):
                    self.assertTrue(
                        torch.all(masks[:, index] <= masks[:, index + 1])
                    )

    def test_forced_capacity_selects_matching_path_and_survival_outputs(self):
        compressor = self._make_compressor().eval()
        slots = torch.randn(2, 8, 16)
        keep_logits = torch.tensor(
            [
                [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5],
                [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
            ]
        )

        with torch.no_grad():
            compressor.capacity_head.weight.zero_()
            slot_logits = compressor.slot_decoder(slots)
            base_gate = torch.softmax(
                keep_logits / compressor.temperature,
                dim=-1,
            )

            for capacity_index, k in enumerate(
                compressor.ADAPTIVE_CAPACITY_CANDIDATES
            ):
                compressor.capacity_head.bias.fill_(-20.0)
                compressor.capacity_head.bias[capacity_index] = 20.0
                path_vec, gate, keep_mask, slot_aux, _ = (
                    compressor._forward_adaptive_capacity(
                        slots,
                        keep_logits,
                    )
                )

                hard_mask = torch.zeros_like(keep_logits)
                top_indices = keep_logits.topk(k=k, dim=-1).indices
                hard_mask.scatter_(1, top_indices, 1.0)
                expected_gate = base_gate * hard_mask
                expected_gate = expected_gate / (
                    expected_gate.sum(dim=-1, keepdim=True) + 1e-8
                )
                expected_path = torch.einsum(
                    "bs,bsd->bd",
                    expected_gate,
                    slots,
                )
                expected_logits = torch.einsum(
                    "bs,bsc->bc",
                    expected_gate,
                    slot_logits,
                )

                self.assertTrue(
                    torch.equal(
                        slot_aux["selected_k"],
                        torch.full((2,), float(k)),
                    )
                )
                self.assertTrue(torch.allclose(keep_mask, hard_mask))
                self.assertTrue(torch.allclose(gate, expected_gate))
                self.assertTrue(torch.allclose(path_vec, expected_path))
                self.assertTrue(
                    torch.allclose(
                        slot_aux["survival_logits"],
                        expected_logits,
                    )
                )

    def test_adaptive_capacity_gradients_reach_predictive_modules(self):
        compressor = self._make_compressor().train()
        tokens = torch.randn(2, 12, 16)

        torch.manual_seed(41)
        path_vec, _, slot_aux = compressor(tokens)
        loss = (
            path_vec.square().mean()
            + slot_aux["survival_logits"].square().mean()
            + slot_aux["expected_capacity_ratio"].mean()
        )
        loss.backward()

        for parameter in (
            compressor.slot_decoder.weight,
            compressor.slot_score[-1].weight,
            compressor.capacity_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_eval_topk_masks_do_not_sample_gumbel_noise(self):
        compressor = self._make_compressor().eval()
        keep_logits = torch.randn(3, 8)

        torch.manual_seed(3)
        first = compressor._adaptive_keep_masks(keep_logits)
        torch.manual_seed(97)
        second = compressor._adaptive_keep_masks(keep_logits)

        self.assertTrue(torch.equal(first, second))

    def test_capacity_controller_is_invariant_to_keep_score_shift(self):
        compressor = self._make_compressor().eval()
        slots = torch.randn(2, 8, 16)
        keep_logits = torch.randn(2, 8)

        with torch.no_grad():
            first = compressor._forward_adaptive_capacity(
                slots,
                keep_logits,
            )
            shifted = compressor._forward_adaptive_capacity(
                slots,
                keep_logits + 7.0,
            )

        for first_value, shifted_value in zip(first[:3], shifted[:3]):
            self.assertTrue(
                torch.allclose(first_value, shifted_value, atol=1e-6)
            )
        for key in (
            "survival_logits",
            "capacity_probs",
            "expected_capacity_ratio",
            "selected_k",
        ):
            self.assertTrue(
                torch.allclose(
                    first[3][key],
                    shifted[3][key],
                    atol=1e-6,
                )
            )

    def test_adaptive_eval_is_reproducible_with_fixed_slot_seed(self):
        model = net.CFSurv(
            make_args(path_slot_adaptive_capacity=True)
        ).eval()
        path = torch.randn(1, 24, 1024)
        omic = torch.randn(1, 150)

        with torch.no_grad():
            torch.manual_seed(71)
            first = model(path, omic, "eval")
            torch.manual_seed(71)
            second = model(path, omic, "eval")

        first_viz = first["viz"]["path_compressor"]
        second_viz = second["viz"]["path_compressor"]
        for key in ("selected_k", "slot_keep_mask", "slot_gate"):
            self.assertTrue(torch.equal(first_viz[key], second_viz[key]))
        self.assertTrue(
            torch.equal(first["hazards"][-1], second["hazards"][-1])
        )

    def test_adaptive_auxiliary_loss_matches_alpha_half_nll(self):
        args = make_args(
            loss="nllsurv",
            path_slot_adaptive_capacity=True,
            lambda_path_slot_capacity=0.01,
        )
        criterion = Loss_factory(args)
        main_hazards = torch.tensor(
            [[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]]
        )
        main_survival = torch.cumprod(1 - main_hazards, dim=1)
        slot_logits = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.5], [0.5, 0.0, -0.5, -1.0]],
            requires_grad=True,
        )
        expected_ratio = torch.tensor([0.25, 0.75], requires_grad=True)
        preds = {
            "hazards": [main_hazards],
            "S": [main_survival],
            "slot_aux": {
                "survival_logits": slot_logits,
                "expected_capacity_ratio": expected_ratio,
            },
        }
        target = {
            "label": torch.tensor([1.0, 2.0]),
            "c": torch.tensor([0.0, 1.0]),
            "phase": "train",
        }

        total, components = criterion(preds, target)
        slot_hazards = torch.sigmoid(slot_logits)
        expected_aux_nll = nll_loss(
            slot_hazards,
            torch.cumprod(1 - slot_hazards, dim=1),
            target["label"],
            target["c"],
            alpha=0.5,
        )
        expected_capacity_penalty = 0.01 * expected_ratio.mean()
        expected_main = nll_loss(
            main_hazards,
            main_survival,
            target["label"],
            target["c"],
        )

        self.assertTrue(
            torch.allclose(components["slot_aux_nll"], expected_aux_nll)
        )
        self.assertTrue(
            torch.allclose(
                components["slot_capacity_penalty"],
                expected_capacity_penalty,
            )
        )
        self.assertTrue(
            torch.allclose(
                total,
                expected_main + expected_aux_nll + expected_capacity_penalty,
            )
        )

        eval_target = dict(target, phase="eval")
        eval_total, eval_components = criterion(preds, eval_target)
        self.assertNotIn("slot_aux_nll", eval_components)
        self.assertTrue(torch.allclose(eval_total, expected_main))

    def test_loss_phase_uses_criterion_state_and_rejects_conflicts(self):
        args = make_args(
            loss="nllsurv",
            path_slot_adaptive_capacity=True,
        )
        criterion = Loss_factory(args)
        hazards = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        preds = {
            "hazards": [hazards],
            "S": [torch.cumprod(1 - hazards, dim=1)],
            "slot_aux": {
                "survival_logits": torch.zeros(1, 4),
                "expected_capacity_ratio": torch.tensor([0.5]),
            },
        }
        target = {
            "label": torch.tensor([1.0]),
            "c": torch.tensor([0.0]),
        }

        criterion.set_phase("eval")
        _, components = criterion(preds, target)
        self.assertNotIn("slot_aux_nll", components)

        with self.assertRaisesRegex(ValueError, "phase mismatch"):
            criterion(preds, dict(target, phase="train"))

        with self.assertRaisesRegex(ValueError, "train.*eval"):
            criterion.set_phase("validation")

    def test_capacity_summary_and_patient_export(self):
        selected = [
            np.asarray([1.0, 1.0, 2.0]),
            np.asarray([4.0, 8.0]),
        ]
        probabilities = [
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
            np.asarray(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
        summary = Engine._summarize_slot_capacity(
            selected,
            probabilities,
        )

        self.assertEqual(summary["slot_capacity_samples"], 5)
        self.assertAlmostEqual(summary["slot_selected_k_mean"], 3.2)
        self.assertAlmostEqual(summary["slot_expected_k_mean"], 3.2)
        self.assertAlmostEqual(summary["slot_k1_ratio"], 0.4)
        self.assertAlmostEqual(summary["slot_k2_ratio"], 0.2)
        self.assertAlmostEqual(summary["slot_k4_ratio"], 0.2)
        self.assertAlmostEqual(summary["slot_k8_ratio"], 0.2)
        self.assertFalse(summary["slot_capacity_collapsed"])
        self.assertGreater(
            summary["slot_capacity_selection_entropy"],
            0.0,
        )

        collapsed = Engine._summarize_slot_capacity(
            [np.full(20, 8.0)],
            [np.tile([0.0, 0.0, 0.0, 1.0], (20, 1))],
        )
        self.assertTrue(collapsed["slot_capacity_collapsed"])
        self.assertEqual(collapsed["slot_capacity_selection_entropy"], 0.0)

        args = make_args(export_tri_embeddings=True)
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(args, tmp, 0)
            rows = []
            out = {
                "viz": {
                    "path_compressor": {
                        "selected_k": torch.tensor([1.0, 4.0]),
                        "expected_capacity": torch.tensor([1.5, 4.5]),
                        "capacity_probs": torch.tensor(
                            [
                                [0.8, 0.1, 0.1, 0.0],
                                [0.0, 0.1, 0.8, 0.1],
                            ]
                        ),
                        "candidate_ks": torch.tensor([1.0, 2.0, 4.0, 8.0]),
                    }
                }
            }
            engine._collect_tri_rows(
                rows,
                None,
                out,
                torch.tensor([0, 1]),
                None,
                torch.tensor([1.0, 2.0]),
                torch.tensor([10.0, 20.0]),
                torch.tensor([0.0, 1.0]),
                np.asarray([-1.0, -2.0]),
            )

        self.assertEqual(rows[0]["slot_selected_k"], 1.0)
        self.assertEqual(rows[1]["slot_expected_k"], 4.5)
        self.assertAlmostEqual(rows[0]["slot_prob_k1"], 0.8)
        self.assertAlmostEqual(rows[1]["slot_prob_k4"], 0.8)

    def test_adaptive_checkpoint_roundtrip_and_mode_mismatch(self):
        args = make_args(path_slot_adaptive_capacity=True)
        with tempfile.TemporaryDirectory() as tmp, quiet_expected_output():
            model = net.CFSurv(args)
            criterion = Loss_factory(args)
            engine = Engine(args, tmp, 0)
            engine.scaler = torch.amp.GradScaler("cuda", enabled=False)
            state = engine._checkpoint_state(
                model,
                None,
                None,
                criterion,
                "ucec",
                0,
            )
            path = Path(tmp) / "adaptive_checkpoint.pth.tar"
            engine._atomic_torch_save(state, str(path))

            restored_engine = Engine(args, tmp, 0)
            restored_engine.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=False,
            )
            start_epoch = restored_engine._load_checkpoint(
                str(path),
                net.CFSurv(args),
                None,
                None,
                Loss_factory(args),
                "ucec",
                torch.device("cpu"),
            )
            self.assertEqual(start_epoch, 1)

            fixed_args = make_args(path_slot_adaptive_capacity=False)
            fixed_engine = Engine(fixed_args, tmp, 0)
            fixed_engine.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=False,
            )
            with self.assertRaisesRegex(
                ValueError,
                "path_slot_adaptive_capacity",
            ):
                fixed_engine._load_checkpoint(
                    str(path),
                    net.CFSurv(fixed_args),
                    None,
                    None,
                    Loss_factory(fixed_args),
                    "ucec",
                    torch.device("cpu"),
                )

    def test_checkpoint_rejects_slot_capacity_hyperparameter_changes(self):
        args = make_args(path_slot_adaptive_capacity=True)
        mismatches = {
            "path_slot_topk_ratio": 0.5,
            "path_slot_temperature": 0.02,
            "path_slot_top_k_method": "gumbel_topk_st",
            "path_slot_train_init_noise": False,
            "path_slot_capacity_temperature": 0.25,
            "lambda_path_slot_capacity": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmp, quiet_expected_output():
            engine = Engine(args, tmp, 0)
            engine.scaler = torch.amp.GradScaler("cuda", enabled=False)
            base_state = engine._checkpoint_state(
                net.CFSurv(args),
                None,
                None,
                Loss_factory(args),
                "ucec",
                0,
            )

            for key, changed_value in mismatches.items():
                with self.subTest(key=key):
                    state = dict(base_state)
                    state["args"] = dict(base_state["args"])
                    state["args"][key] = changed_value
                    path = Path(tmp) / f"mismatch_{key}.pth.tar"
                    engine._atomic_torch_save(state, str(path))

                    current_engine = Engine(args, tmp, 0)
                    current_engine.scaler = torch.amp.GradScaler(
                        "cuda",
                        enabled=False,
                    )
                    with self.assertRaisesRegex(ValueError, key):
                        current_engine._load_checkpoint(
                            str(path),
                            net.CFSurv(args),
                            None,
                            None,
                            Loss_factory(args),
                            "ucec",
                            torch.device("cpu"),
                        )
