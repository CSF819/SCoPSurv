import numpy as np
import pandas as pd
import re

REPRESENTATION_PREFIXES = [
    "encoded_path",
    "encoded_gene",
    "h",
    "z_b",
    "z_s",
    "z_u",
    "z_u_cond",
    "proto_assign",
]


def _vector_columns(df, prefix):
    if df is None:
        return []

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    indexed_columns = []

    for column in df.columns:
        match = pattern.fullmatch(str(column))
        if match is not None:
            indexed_columns.append((int(match.group(1)), column))

    indexed_columns.sort(key=lambda item: item[0])
    return [column for _, column in indexed_columns]

def _valid_probe_frame(df, columns):
    if df is None or df.empty or not columns or "site_idx" not in df.columns:
        return None, None
    local = df[df["site_idx"].astype(int) >= 0].copy()
    if local.empty:
        return None, None
    x = local[columns].to_numpy(dtype=np.float32)
    y = local["site_idx"].to_numpy(dtype=np.int64)
    finite = np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    if x.shape[0] == 0:
        return None, None
    return x, y


def summarize_prototype_usage(df, output_csv):
    proto_cols = _vector_columns(df, "proto_assign") if df is not None else []
    rows = []
    if proto_cols:
        usage = df[proto_cols].to_numpy(dtype=np.float64)
        usage = usage[np.isfinite(usage).all(axis=1)]
        if usage.size > 0:
            mean_usage = usage.mean(axis=0)
            entropy = float(-(mean_usage * np.log(np.clip(mean_usage, 1e-8, None))).sum())
            for idx, value in enumerate(mean_usage):
                rows.append(
                    {
                        "prototype": idx,
                        "mean_usage": float(value),
                        "usage_entropy": entropy,
                        "usage_min": float(mean_usage.min()),
                        "usage_max": float(mean_usage.max()),
                        "n": int(usage.shape[0]),
                    }
                )
    if not rows:
        rows.append({"prototype": -1, "status": "skipped_no_proto_assign"})
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")


def run_site_probes(train_df, eval_df, output_csv, random_state=0):
    rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        pd.DataFrame([{"representation": "all", "status": f"skipped_import_error: {e}"}]).to_csv(
            output_csv, index=False, encoding="utf-8"
        )
        return

    for prefix in REPRESENTATION_PREFIXES:
        train_cols = _vector_columns(train_df, prefix)
        eval_cols = _vector_columns(eval_df, prefix)
        columns = [col for col in train_cols if col in set(eval_cols)]
        x_train, y_train = _valid_probe_frame(train_df, columns)
        x_eval, y_eval = _valid_probe_frame(eval_df, columns)

        if x_train is None or x_eval is None:
            rows.append({"representation": prefix, "status": "skipped_missing_embedding_or_site"})
            continue
        if np.unique(y_train).size < 2 or np.unique(y_eval).size < 2:
            rows.append(
                {
                    "representation": prefix,
                    "status": "skipped_single_site_class",
                    "train_n": int(x_train.shape[0]),
                    "eval_n": int(x_eval.shape[0]),
                    "train_site_classes": int(np.unique(y_train).size),
                    "eval_site_classes": int(np.unique(y_eval).size),
                }
            )
            continue

        try:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_eval = scaler.transform(x_eval)
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(x_train, y_train)
            pred = clf.predict(x_eval)
            rows.append(
                {
                    "representation": prefix,
                    "status": "ok",
                    "accuracy": float(accuracy_score(y_eval, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred)),
                    "macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
                    "train_n": int(x_train.shape[0]),
                    "eval_n": int(x_eval.shape[0]),
                    "train_site_classes": int(np.unique(y_train).size),
                    "eval_site_classes": int(np.unique(y_eval).size),
                    "dim": int(x_train.shape[1]),
                }
            )
        except Exception as e:
            rows.append({"representation": prefix, "status": f"failed: {e}"})

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
