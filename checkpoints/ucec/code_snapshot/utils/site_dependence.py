"""Fixed-kernel HSIC diagnostics for frozen patient representations.

Gretton et al. (NIPS 2007), Eq. (4): trace(K H L H) / n**2.
For categorical site labels, L is the linear kernel of one-hot labels.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


def centered_kernel(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square")
    if not np.isfinite(kernel).all() or not np.allclose(kernel, kernel.T, atol=1e-12):
        raise ValueError("Kernel must be finite and symmetric")
    return kernel - kernel.mean(0)[None, :] - kernel.mean(1)[:, None] + kernel.mean()


def frozen_rbf_kernel(train, evaluation):
    """Fit scale and median bandwidth on training patients only; no PCA."""
    train = np.asarray(train, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("Training and evaluation feature dimensions must agree")
    if len(train) < 2 or len(evaluation) < 4:
        raise ValueError("At least two training and four evaluation patients are required")
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Features must be finite")
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    squared_bandwidth = float(np.median(pdist(train_scaled, metric="sqeuclidean")))
    if squared_bandwidth <= 0 or not np.isfinite(squared_bandwidth):
        raise ValueError("Median training distance is zero or nonfinite")
    evaluation_scaled = scaler.transform(evaluation)
    distances = squareform(pdist(evaluation_scaled, metric="sqeuclidean"))
    kernel = np.exp(-distances / (2 * squared_bandwidth))
    return kernel, scaler, squared_bandwidth


def hsic_from_labels(centered, labels):
    labels = np.asarray(labels)
    if len(labels) != len(centered):
        raise ValueError("One site label per kernel row is required")
    same_site = labels[:, None] == labels[None, :]
    return float(np.sum(centered * same_site) / len(labels) ** 2)


def permutation_null_mean(centered, labels):
    """Exact expectation over count-preserving label permutations."""
    _, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    same_off_diagonal = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    return float(np.trace(centered) * (1 - same_off_diagonal) / n ** 2)


def permutation_pvalue(observed, null):
    null = np.asarray(null, dtype=float)
    # Include ties despite floating-point summation differences.
    tolerance = 1e-12 * max(abs(observed), np.max(np.abs(null)), 1e-15)
    return float((1 + np.count_nonzero(null >= observed - tolerance)) / (len(null) + 1))


def hsic_permutation_test(kernel, labels, permutations=9999, seed=20260911, batch_size=128):
    centered = centered_kernel(kernel)
    labels = np.asarray(labels)
    if len(labels) != len(centered) or len(labels) < 4:
        raise ValueError("At least four matched patient labels are required")
    classes, labels = np.unique(labels, return_inverse=True)
    if len(classes) < 2 or len(classes) == len(labels):
        raise ValueError("Require at least two sites and one repeated site")
    if permutations < 1 or batch_size < 1:
        raise ValueError("Permutation count and batch size must be positive")
    observed = hsic_from_labels(centered, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for start in range(0, permutations, batch_size):
        stop = min(start + batch_size, permutations)
        shuffled = np.stack([rng.permutation(labels) for _ in range(stop - start)])
        matches = shuffled[:, :, None] == shuffled[:, None, :]
        null[start:stop] = np.einsum("bij,ij->b", matches, centered, optimize=False) / len(labels) ** 2
    expectation = permutation_null_mean(centered, labels)
    result = {
        "hsic": observed,
        "null_mean_exact": expectation,
        "null_mean_mc": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_low": float(np.quantile(null, .025)),
        "null_high": float(np.quantile(null, .975)),
        "excess_hsic": observed - expectation,
        "observed_over_null": observed / expectation if expectation > 0 else None,
        "p_permutation_exploratory": permutation_pvalue(observed, null),
        "permutations": permutations,
        "seed": seed,
    }
    return result, null


def holm_adjust(pvalues):
    """Holm correction, valid without independence between tests."""
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("P values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result
