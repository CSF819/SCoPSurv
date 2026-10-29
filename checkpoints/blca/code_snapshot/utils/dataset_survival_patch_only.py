from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from utils.config_utils import (
    EXPECTED_OMIC_DIMS,
    expected_omic_dim,
    path_feature_dim,
    strip_slide_suffix,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

METADATA_CANDIDATES = [
    "patient_uid",
    "disc_label",
    "label",
    "case_id",
    "slide_id",
    "site",
    "site_idx",
    "oncotree_code",
    "survival_months",
    "censorship",
    "is_female",
    "age",
    "train",
    "Unnamed: 0",
    "Unnamed: 0.1",
]

FORBIDDEN_GENOMIC_COLUMNS = set(METADATA_CANDIDATES)


class FeatureDimensionError(ValueError):
    """Raised when a WSI tensor contradicts the configured extractor."""

def _metadata_index(slide_data):
    return pd.Index([c for c in METADATA_CANDIDATES if c in slide_data.columns])


def _assert_no_leakage_columns(genomic_features):
    leaked = FORBIDDEN_GENOMIC_COLUMNS.intersection(set(genomic_features.columns))
    if leaked:
        raise ValueError(f"Metadata leakage into genomic features: {sorted(leaked)}")


def _load_patch_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ============================================================
# 通用 WSI 生存数据集（不含具体特征读取）
# ============================================================
class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv', modal='omic', apply_sig=False,
                 shuffle=False, seed=7, n_bins=4, ignore=None,
                 patient_strat=False, label_col=None, filter_dict=None, eps=1e-6, **kwargs):
        self.custom_test_ids = None
        self.seed = seed
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.bins = None
        self.last_split_audit = {}

        slide_data = pd.read_csv(csv_path, low_memory=False)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data.values)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # 只保留 IDC
        if "IDC" in slide_data['oncotree_code'].astype(str).values:
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 先构造病人表；生存时间分箱必须在读取 split 后仅用训练病人拟合。
        patients_df = slide_data.drop_duplicates(['case_id']).copy()

        # 建立 patient -> slide_id 映射
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})
        self.patient_dict = patient_dict

        # 回到病人粒度，每个病人一条记录
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        # 标签字典的大小固定；具体分箱标签在 return_splits 中生成。
        label_dict = {}
        key_count = 0
        for i in range(self.n_bins):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1
        self.label_dict = label_dict
        self.num_classes = len(self.label_dict)
        slide_data["disc_label"] = -1
        slide_data["label"] = -1

        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {
            'case_id': patients_df['case_id'].values,
            'label': patients_df['label'].values
        }
        self.patient_uid_map = {
            str(case_id): uid for uid, case_id in enumerate(self.patient_data['case_id'])
        }
        slide_data['patient_uid'] = slide_data['case_id'].astype(str).map(self.patient_uid_map).astype(int)

        if "site" not in slide_data.columns:
            slide_data["site"] = slide_data["case_id"].astype(str).str.split("-").str[1]
        slide_data["site"] = slide_data["site"].astype(str).fillna("UNKNOWN")
        slide_data.loc[slide_data["site"].isin(["", "nan", "None", "NONE"]), "site"] = "UNKNOWN"

        sites = sorted(slide_data["site"].unique().tolist())
        self.site_to_idx = {site: idx for idx, site in enumerate(sites)}
        self.idx_to_site = {idx: site for site, idx in self.site_to_idx.items()}
        self.num_sites = len(sites)
        slide_data["site_idx"] = slide_data["site"].map(self.site_to_idx).astype(int)

        self.slide_data = slide_data

        # 显式元信息集合，避免新增字段意外进入 genomic features。
        self.metadata = _metadata_index(slide_data)
        self.modal = modal
        self.cls_ids_prep()

        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv(
                './splits/signatures.csv'
            )
        else:
            self.signatures = None

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def _fit_bins_from_training_split(self, all_splits: pd.DataFrame):
        train_ids = set(all_splits.get("train", pd.Series(dtype=object)).dropna().astype(str))
        val_ids = set(all_splits.get("val", pd.Series(dtype=object)).dropna().astype(str))
        test_ids = set(all_splits.get("test", pd.Series(dtype=object)).dropna().astype(str))
        overlap = {
            "train_val": sorted(train_ids & val_ids),
            "train_test": sorted(train_ids & test_ids),
            "val_test": sorted(val_ids & test_ids),
        }
        if any(overlap.values()):
            raise ValueError(
                "Patient leakage across split columns: "
                + ", ".join(f"{name}={len(ids)}" for name, ids in overlap.items() if ids)
            )

        train_mask = self.slide_data["slide_id"].astype(str).isin(train_ids)
        train_df = self.slide_data.loc[train_mask]
        if train_df.empty:
            raise ValueError("Training split is empty; cannot fit survival time bins.")
        uncensored = train_df.loc[
            pd.to_numeric(train_df["censorship"], errors="coerce") < 1,
            self.label_col,
        ]
        uncensored = pd.to_numeric(uncensored, errors="coerce").dropna()
        if len(uncensored) < self.n_bins or uncensored.nunique() < self.n_bins:
            raise ValueError(
                f"Training split needs at least {self.n_bins} uncensored samples with "
                f"{self.n_bins} distinct survival times; got n={len(uncensored)}, "
                f"unique={uncensored.nunique()}."
            )
        try:
            _, q_bins = pd.qcut(
                uncensored,
                q=self.n_bins,
                retbins=True,
                labels=False,
                duplicates="raise",
            )
        except ValueError as exc:
            raise ValueError(f"Failed to fit {self.n_bins} training-only survival bins: {exc}") from exc

        q_bins = np.asarray(q_bins, dtype=float)
        q_bins[0] = -np.inf
        q_bins[-1] = np.inf
        disc_labels = pd.cut(
            pd.to_numeric(self.slide_data[self.label_col], errors="coerce"),
            bins=q_bins,
            labels=False,
            right=False,
            include_lowest=True,
        )
        if disc_labels.isna().any():
            bad = int(disc_labels.isna().sum())
            raise ValueError(f"Failed to assign survival bins for {bad} patients.")

        disc = disc_labels.astype(int).to_numpy()
        censorship = pd.to_numeric(
            self.slide_data["censorship"], errors="raise"
        ).astype(int).to_numpy()
        combined = np.asarray(
            [self.label_dict[(int(bin_idx), int(censor))] for bin_idx, censor in zip(disc, censorship)],
            dtype=int,
        )
        self.slide_data.loc[:, "disc_label"] = disc
        self.slide_data.loc[:, "label"] = combined
        self.bins = q_bins
        self.patient_data = {
            "case_id": self.slide_data["case_id"].to_numpy(),
            "label": combined,
        }
        self.cls_ids_prep()
        self.last_split_audit = {
            "train_cases": len(train_ids),
            "val_cases": len(val_ids),
            "test_cases": len(test_ids),
            "overlap_counts": {name: len(ids) for name, ids in overlap.items()},
            "bin_fit_case_ids": sorted(train_df["case_id"].astype(str).tolist()),
            "uncensored_train_cases": int(len(uncensored)),
            "bins": [None if not np.isfinite(value) else float(value) for value in q_bins],
        }

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def get_split_from_df(self, all_splits: pd.DataFrame, split_key: str = 'train',
                          scaler=None, set_name='BLCA', extractor='resnet50'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True).tolist()

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split)
            df_slice = self.slide_data[mask].reset_index(drop=True)

            split = Generic_Split(
                slide_data=df_slice,
                metadata=self.metadata,
                modal=self.modal,
                signatures=self.signatures,
                data_dir=self.data_dir,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes,
                set_name=set_name,
                extractor=extractor,
                site_to_idx=self.site_to_idx,
                idx_to_site=self.idx_to_site,
                num_sites=self.num_sites,
                args=getattr(self, 'args', None)
            )
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None,
                      set_name='BLCA', extractor='resnet50'):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            self._fit_bins_from_training_split(all_splits)
            train_split = self.get_split_from_df(
                all_splits=all_splits,
                split_key='train',
                set_name=set_name,
                extractor=extractor
            )
            val_split = self.get_split_from_df(
                all_splits=all_splits,
                split_key='val',
                set_name=set_name,
                extractor=extractor
            )

            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)

        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]


# ============================================================
# MIL Dataset：只在这里读 patch 特征（patch-only + top-k）
# ============================================================
class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, modal: str = 'omic', args=None, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.modal = modal
        self.use_h5 = False

        # args 主要用来拿 max_tokens、extractor 等
        self.args = args
        self.extractor = getattr(args, 'extractor', 'feather') if args is not None else 'feather'
        self.wsi_load_stats = {
            "requested": 0,
            "loaded": 0,
            "missing": 0,
            "corrupt": 0,
            "dimension_errors": 0,
        }

    def __len__(self):
        return len(self.slide_data)

    def _data_dir_for_index(self, idx):
        if isinstance(self.data_dir, dict):
            source = self.slide_data["oncotree_code"].iloc[idx]
            return self.data_dir[source]
        return self.data_dir

    def _patch_path(self, data_dir, slide_id):
        return os.path.join(data_dir, "patch", f"{strip_slide_suffix(slide_id)}.pt")

    def audit_wsi_files(self, load_tensors=False):
        """Audit WSI availability and optionally token counts without changing samples."""
        expected_dim = path_feature_dim(self.extractor)
        paths = {}
        for idx in range(len(self.slide_data)):
            data_dir = self._data_dir_for_index(idx)
            case_id = self.slide_data["case_id"].iloc[idx]
            for slide_id in self.patient_dict[case_id]:
                path = self._patch_path(data_dir, slide_id)
                paths[path] = str(slide_id)

        report = {
            "patients": int(len(self.slide_data)),
            "unique_slides": int(len(paths)),
            "expected_feature_dim": int(expected_dim),
            "available": 0,
            "missing": 0,
            "corrupt": 0,
            "dimension_errors": 0,
            "load_tensors": bool(load_tensors),
        }
        token_counts = []
        for path, slide_id in paths.items():
            if not os.path.isfile(path):
                report["missing"] += 1
                continue
            report["available"] += 1
            if not load_tensors:
                continue
            try:
                payload = _load_patch_tensor(path)
                if not torch.is_tensor(payload):
                    raise TypeError(f"Expected Tensor, got {type(payload).__name__}")
                tensor = payload.squeeze(0)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                if tensor.dim() != 2:
                    raise ValueError(f"Expected 2D patch tensor, got {tuple(tensor.shape)}")
                if int(tensor.shape[-1]) != expected_dim:
                    report["dimension_errors"] += 1
                    continue
                token_counts.append(int(tensor.shape[0]))
            except Exception:
                report["corrupt"] += 1

        denominator = max(1, report["unique_slides"])
        report["missing_rate"] = float(report["missing"] / denominator)
        if token_counts:
            arr = np.asarray(token_counts, dtype=np.float64)
            report["token_count"] = {
                "mean": float(arr.mean()),
                "p95": float(np.percentile(arr, 95)),
                "max": int(arr.max()),
            }
            max_tokens = float(arr.max())
            input_bytes = max_tokens * expected_dim * 4.0
            projected_bytes = max_tokens * 256 * 4.0
            attention_bytes = max_tokens * 8 * 8 * 4.0
            report["estimated_peak_input_activation_mb"] = float(
                (input_bytes + projected_bytes + attention_bytes) / (1024 ** 2)
            )
        return report

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'].iloc[idx]
        patient_uid = int(self.slide_data['patient_uid'].iloc[idx])
        label = self.slide_data['disc_label'].iloc[idx]
        event_time = self.slide_data[self.label_col].iloc[idx]
        c = self.slide_data['censorship'].iloc[idx]
        slide_ids = self.patient_dict[case_id]

        data_dir = self._data_dir_for_index(idx)
        default_feat_dim = path_feature_dim(self.extractor)

        # ==================== Pathology：仅 patch 特征 + top-k ====================
        if 'path' in self.modal or 'multi' in self.modal:
            collected_tokens = []
            feat_dim = None

            for slide_id in slide_ids:
                patch_path = self._patch_path(data_dir, slide_id)
                self.wsi_load_stats["requested"] += 1

                try:
                    if not os.path.isfile(patch_path):
                        self.wsi_load_stats["missing"] += 1
                        continue
                    patch_bag = _load_patch_tensor(patch_path)
                    if not torch.is_tensor(patch_bag):
                        raise TypeError(
                            f"Expected Tensor in {patch_path}, got {type(patch_bag).__name__}"
                        )

                    patch_tokens = patch_bag.squeeze(0).cpu()
                    if patch_tokens.dim() == 1:
                        patch_tokens = patch_tokens.unsqueeze(0)  # [1, D]
                    if patch_tokens.dim() != 2:
                        raise ValueError(
                            f"Expected 2D patch tensor for {slide_id}, got {tuple(patch_tokens.shape)}"
                        )
                    if int(patch_tokens.size(-1)) != default_feat_dim:
                        self.wsi_load_stats["dimension_errors"] += 1
                        raise FeatureDimensionError(
                            f"Feature dimension mismatch for {slide_id}: expected "
                            f"{default_feat_dim} from extractor={self.extractor}, "
                            f"got {patch_tokens.size(-1)}"
                        )

                    if feat_dim is None:
                        feat_dim = patch_tokens.size(-1)

                    collected_tokens.append(patch_tokens)
                    self.wsi_load_stats["loaded"] += 1

                except FeatureDimensionError:
                    raise
                except Exception as e:
                    self.wsi_load_stats["corrupt"] += 1
                    print(f"[Warning] Load failed for {slide_id}: {e}")
                    continue

            if collected_tokens:
                path_features = torch.cat(collected_tokens, dim=0)   # [T, D]
            else:
                if feat_dim is None:
                    feat_dim = default_feat_dim
                path_features = torch.zeros((6, feat_dim), dtype=torch.float32)

            # 数值消毒
            path_features = torch.nan_to_num(path_features, nan=0.0)
            path_features = torch.clamp(path_features, -1e6, 1e6)

            # 至少保证 6 个 token（与原先接口保持一致）
            if path_features.shape[0] < 6:
                pad = 6 - path_features.shape[0]
                path_features = torch.cat(
                    [path_features, path_features[:1].repeat(pad, 1)],
                    dim=0
                )
        else:
            # 不使用病理模态时，用零占位
            path_features = torch.zeros((6, default_feat_dim), dtype=torch.float32)

        # Omic-only and multimodal modes use the same signature ordering expected by the model.
        if self.modal in {"omic", "multi"}:
            if getattr(self, "omic_names", None):
                genomic_features = torch.cat([
                    torch.tensor(
                        self.genomic_features[omic_name].iloc[idx].values,
                        dtype=torch.float32
                    )
                    for omic_name in self.omic_names
                ], dim=0)
            else:
                genomic_features = torch.tensor(
                    self.genomic_features.iloc[idx].values, dtype=torch.float32
                )
        else:
            genomic_features = torch.zeros(
                (expected_omic_dim(getattr(self, "dataset", "")),),
                dtype=torch.float32,
            )

        expected_dim = expected_omic_dim(getattr(self, "dataset", ""))
        if int(genomic_features.numel()) != expected_dim:
            raise ValueError(
                f"Omic dimension mismatch for {self.dataset}: expected {expected_dim}, "
                f"got {genomic_features.numel()} in modal={self.modal}"
            )

        site_idx = int(self.slide_data['site_idx'].iloc[idx]) if 'site_idx' in self.slide_data.columns else 0

        return path_features, genomic_features, label, event_time, c, idx, patient_uid, site_idx


# ============================================================
# Split：继承上面的 Dataset，主要负责 scaler 和 omic 子集划分
# ============================================================
class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, modal, signatures=None,
                 data_dir=None, label_col=None, patient_dict=None,
                 num_classes=2, set_name='BLCA', extractor='resnet50',
                 site_to_idx=None, idx_to_site=None, num_sites=None, args=None):
        """
        这里继续保持“轻量 Split”，不调用父类 __init__，
        只补齐 __getitem__ 需要用到的字段，并新增 self.args。
        """
        # 不使用 h5
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.modal = modal
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        self.dataset = set_name
        self.extractor = extractor
        self.args = args
        self.wsi_load_stats = {
            "requested": 0,
            "loaded": 0,
            "missing": 0,
            "corrupt": 0,
            "dimension_errors": 0,
        }
        self.site_to_idx = site_to_idx or {}
        self.idx_to_site = idx_to_site or {}
        self.num_sites = int(num_sites) if num_sites is not None else len(self.site_to_idx)

        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        # 去掉 metadata 后，剩下的就是基因特征
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        _assert_no_leakage_columns(self.genomic_features)
        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate(
                    [omic + modal for modal in ['_mut', '_cnv', '_rnaseq']]
                )
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
            expected = EXPECTED_OMIC_DIMS.get(str(set_name).lower())
            actual = int(sum(self.omic_sizes))
            if expected is not None and actual != expected:
                raise ValueError(
                    f"Signature omic dimension mismatch for {set_name}: "
                    f"expected {expected}, got {actual}."
                )

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple = None):
        transformed = pd.DataFrame(
            scalers[0].transform(self.genomic_features)
        )
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
