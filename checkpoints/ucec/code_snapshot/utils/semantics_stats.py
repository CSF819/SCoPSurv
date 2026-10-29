"""Numerical building blocks for the frozen Site--Cohort--Patient experiment.

All estimators are trained separately from evaluation outcomes. Censoring KM
uses sksurv's reverse-KM convention, including its treatment of tied times.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

REPRESENTATIONS = ("h", "z_b", "z_s", "z_u")


@dataclass(frozen=True)
class SemanticsConfig:
    version: int = 1
    seed: int = 20260913
    permutations: int = 9999
    null_draws: int = 9999
    bootstrap: int = 2000
    random_repeats: int = 20
    calibration_reps: int = 200
    calibration_draws: int = 999
    trees: int = 200
    min_site_count: int = 5
    rff_features: int = 64
    # The KRR objective is mean squared error + lambda * RKHS norm squared.
    ridge_alphas: tuple = (1e-4, 1e-3, 1e-2, 1e-1)
    bandwidth_multipliers: tuple = (.5, 1., 2.)
    leaf_sizes: tuple = (5, 10, 20)


def seed_for(seed, *indices):
    return int(np.random.SeedSequence([seed, *indices]).generate_state(1)[0])


def p_upper(observed, null):
    null = np.asarray(null, float)
    if not np.isfinite(observed) or not len(null) or not np.isfinite(null).all():
        return np.nan
    tol = 1e-12 * max(abs(observed), float(np.max(np.abs(null))), 1e-15)
    return float((1 + np.count_nonzero(null >= observed - tol)) / (len(null) + 1))


def bootstrap_counts(labels, draws, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, len(labels)), dtype=np.int32)
    for level in np.unique(labels):
        ids = np.flatnonzero(labels == level)
        counts[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    return counts


def permanova_many(x, labels, weights=None):
    """Euclidean PERMANOVA, also for bootstrap patient multiplicities.

    x is (patients, representations, dimensions); output is (draws, reps, 3).
    The last axis is R2, adjusted R2, pseudo-F. Duplicated bootstrap patients
    have frequency weights, not extra independent observations.
    """
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x[:, None, :]
    labels = np.asarray(labels)
    w = np.ones((1, len(x))) if weights is None else np.atleast_2d(weights).astype(float)
    n = w.sum(1)
    flat = x.reshape(len(x), -1)
    sums = (w @ flat).reshape(len(w), *x.shape[1:])
    raw = w @ np.sum(x * x, axis=2)
    grand = np.sum(sums * sums, axis=2) / n[:, None]
    total = raw - grand
    between, k = -grand.copy(), np.zeros(len(w), int)
    for level in np.unique(labels):
        mask = labels == level
        ng = w[:, mask].sum(1)
        sx = (w[:, mask] @ flat[mask]).reshape(len(w), *x.shape[1:])
        between += np.divide(np.sum(sx * sx, 2), ng[:, None],
                             out=np.zeros_like(between), where=ng[:, None] > 0)
        k += ng > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = between / total
        adjusted = 1 - (1 - r2) * (n[:, None] - 1) / (n - k)[:, None]
        f = between / (k[:, None] - 1) / ((total - between) / (n - k)[:, None])
    out = np.stack([r2, adjusted, f], axis=2)
    out[(n <= k) | (k < 2)] = np.nan
    out[np.broadcast_to(total[:, :, None] <= 1e-14, out.shape)] = np.nan
    return out


def site_permutations(x, labels, draws, seed):
    """Count-preserving common permutations, sparse group aggregation."""
    x = np.asarray(x, float)
    classes, y = np.unique(labels, return_inverse=True)
    n, reps, d = x.shape
    k = len(classes)
    if k < 2 or n <= k:
        raise ValueError("PERMANOVA requires >=2 sites and residual degrees of freedom")
    counts = np.bincount(y)
    centered = x - x.mean(0)
    total = np.sum(centered * centered, axis=(0, 2))
    rng = np.random.default_rng(seed)
    result = np.empty((draws, reps))
    for start in range(0, draws, 128):
        b = min(128, draws - start)
        yp = np.stack([rng.permutation(y) for _ in range(b)])
        groups = csr_matrix((np.ones(b * n), ((yp + np.arange(b)[:, None] * k).ravel(),
                                             np.tile(np.arange(n), b))), shape=(b * k, n))
        sx = (groups @ centered.reshape(n, -1)).reshape(b, k, reps, d)
        between = (np.sum(sx * sx, axis=3) / counts[None, :, None]).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[start:start+b] = (between / (k - 1)) / ((total - between) / (n - k))
    return result


def administrative_censor(train_time, time, event):
    cap = float(np.nextafter(np.max(train_time), -np.inf))
    time, event = np.asarray(time, float), np.asarray(event, bool)
    return np.minimum(time, cap), event & (time <= cap), cap


def weighted_km(time, event, weights=None):
    time, event = np.asarray(time, float), np.asarray(event, bool)
    w = np.ones(len(time)) if weights is None else np.asarray(weights, float)
    if not len(time) or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Empty donor pool or invalid/zero prototype weights")
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("Invalid donor follow-up time")
    unique, inv = np.unique(time, return_inverse=True)
    totals = np.bincount(inv, weights=w, minlength=len(unique))
    deaths = np.bincount(inv, weights=w * event, minlength=len(unique))
    at_risk = np.cumsum(totals[::-1])[::-1]
    factor = 1 - np.divide(deaths, at_risk, out=np.zeros_like(deaths), where=at_risk > 0)
    survival = np.cumprod(np.clip(factor, 0, 1))
    return {"time": unique, "survival": survival, "risk_weight": at_risk,
            "event_weight": deaths, "support": float(time[w > 0].max())}


def km_predict(curve, times, extrapolate=False):
    query = np.asarray(times, float)
    indices = np.searchsorted(curve["time"], query, side="right") - 1
    values = np.ones(query.shape)
    positive = indices >= 0
    values[positive] = curve["survival"][indices[positive]]
    if not extrapolate:
        values[(query > curve["support"]) & (values > 1e-14)] = np.nan
    return values


def prototype_weights(sites, assignments):
    a = np.asarray(assignments, float)
    if a.ndim != 2 or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid soft assignments")
    w = np.zeros_like(a)
    for site in np.unique(sites):
        mask = np.asarray(sites) == site
        den = a[mask].sum(0)
        if (den <= 0).any():
            raise ValueError(f"Prototype has zero total weight at donor site {site}")
        w[mask] = (mask.sum() / len(a)) * a[mask] / den
    return w


def prototype_predictions(train_time, train_event, train_site, train_a,
                          eval_site, eval_a, cuts):
    """No evaluation outcome argument: donor estimates cannot depend on it."""
    n, k = len(eval_site), train_a.shape[1]
    predicted = {name: np.full((n, len(cuts)), np.nan) for name in ("proto", "reference", "km")}
    records, failures = {}, []
    for site in np.unique(eval_site):
        donor = np.asarray(train_site) != site
        target = np.asarray(eval_site) == site
        try:
            a = train_a[donor]
            w = prototype_weights(np.asarray(train_site)[donor], a)
            curves = [weighted_km(train_time[donor], train_event[donor], w[:, j]) for j in range(k)]
            matrix = np.stack([km_predict(curve, cuts) for curve in curves])
            predicted["proto"][target] = eval_a[target] @ matrix
            predicted["reference"][target] = a.mean(0) @ matrix
            pooled = weighted_km(train_time[donor], train_event[donor])
            predicted["km"][target] = km_predict(pooled, cuts)
            records[str(site)] = {"donor_indices": np.flatnonzero(donor), "weights": w,
                                  "mean_assignment": a.mean(0), "curves": curves, "pooled": pooled,
                                  "ess": w.sum(0)**2 / np.sum(w*w, 0)}
        except ValueError as exc:
            failures.append({"site": str(site), "reason": str(exc)})
    return predicted, records, failures


class MarginalCensoring:
    def fit(self, time, event):
        self.model = CensoringDistributionEstimator().fit(Surv.from_arrays(event, time))
        return self

    def predict(self, time):
        time = np.asarray(time, float)
        output = np.full(time.shape, np.nan)
        valid = np.isfinite(time) & (time >= 0) & (time <= self.model.unique_time_[-1])
        # Reverse KM with an all-event training sample has G(t)=1 everywhere.
        if np.all(self.model.prob_ == 1):
            valid = np.isfinite(time) & (time >= 0)
        if np.any(valid):
            output[valid] = self.model.predict_proba(time[valid])
        return output


def brier_rows(time, event, survival, endpoint, g_at_time, g_at_endpoint):
    """Patient IPCW contributions, including zero for early censored patients.

    Missing censoring weights are required only when the patient contributes.
    The denominator is the size of the common patient population, not the sum
    of IPCW weights and not the number of observed endpoint outcomes.
    """
    time, event, survival = np.asarray(time), np.asarray(event, bool), np.asarray(survival)
    gt = np.broadcast_to(g_at_time, time.shape)
    gu = np.broadcast_to(g_at_endpoint, time.shape)
    cases, controls = event & (time <= endpoint), time > endpoint
    valid = np.isfinite(survival) & (survival >= -1e-12) & (survival <= 1+1e-12)
    valid &= (~cases | (np.isfinite(gt) & (gt > 0))) & (~controls | (np.isfinite(gu) & (gu > 0)))
    scores = np.full(len(time), np.nan)
    scores[valid] = 0
    mask = valid & cases
    scores[mask] = survival[mask]**2 / gt[mask]
    mask = valid & controls
    scores[mask] = (1-survival[mask])**2 / gu[mask]
    return scores, valid


def conditional_kernel(site1, a1, site2, a2, bandwidth2, clinical1=None, clinical2=None):
    categorical = np.equal.outer(site1, site2).astype(float)
    ka = np.exp(-cdist(a1, a2, "sqeuclidean") / (2 * bandwidth2))
    components = [categorical, ka, categorical * ka]
    if clinical1 is not None:
        kc = np.equal.outer(clinical1, clinical2).astype(float)
        # Nested categorical main effects and all pairwise/triple interactions.
        components += [kc, categorical * kc, ka * kc, categorical * ka * kc]
    return sum(components) / len(components)


def positive_median_distance(x):
    values = pdist(x, "sqeuclidean")
    median = float(np.median(values)) if len(values) else 0.
    # A constant assignment makes the RBF identically one for any bandwidth.
    return median if median > 1e-14 else 1.


class FeatureMap:
    def __init__(self, dimension=64, seed=0):
        self.dimension, self.seed = dimension, seed

    def fit(self, x):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.bandwidth2 = positive_median_distance(z)
        rng = np.random.default_rng(self.seed)
        self.omega = rng.normal(size=(z.shape[1], self.dimension)) / np.sqrt(self.bandwidth2)
        self.phase = rng.uniform(0, 2*np.pi, self.dimension)
        self.output_scaler = StandardScaler().fit(self._raw(z))
        return self

    def _raw(self, z):
        return np.concatenate([z, np.sqrt(2/self.dimension) * np.cos(z @ self.omega + self.phase)], axis=1)

    def transform(self, x):
        return self.output_scaler.transform(self._raw(self.scaler.transform(x)))


def multiplier_gcm(products, draws, seed):
    """Representation-level max statistic; centered Gaussian multiplier null."""
    v = np.asarray(products, float)
    if v.ndim != 2 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("GCM requires >=3 finite patient product vectors")
    n = len(v)
    mean = v.mean(0)
    centered = v - mean
    sd = np.sqrt(np.mean(centered*centered, 0))
    tolerance = max(1e-12, float(np.max(sd))*1e-10)
    keep = sd > tolerance
    if not keep.any():
        raise ValueError("All residual-product coordinates have zero numerical variance")
    normalized = centered[:, keep] / sd[keep] / np.sqrt(n)
    statistic = float(np.max(np.abs(np.sqrt(n) * mean[keep] / sd[keep])))
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for start in range(0, draws, 128):
        end = min(start + 128, draws)
        multipliers = rng.normal(size=(end-start, n))
        null[start:end] = np.max(np.abs(multipliers @ normalized), axis=1)
    result = {"statistic": statistic, "p": p_upper(statistic, null),
              "null_q025": float(np.quantile(null, .025)), "null_q975": float(np.quantile(null, .975)),
              "kept_dimensions": int(keep.sum()), "dropped_dimensions": np.flatnonzero(~keep).tolist(),
              "variance_tolerance": tolerance, "n": n, "draws": draws}
    return result, null


def partial_explained(x, site, subtype):
    """Additional multivariate sum-of-squares explained by subtype given site."""
    site, subtype = np.asarray(site), np.asarray(subtype)
    base = np.column_stack([site == s for s in np.unique(site)]).astype(float)
    extra = np.column_stack([subtype == s for s in np.unique(subtype)]).astype(float)
    residual = x - base @ np.linalg.lstsq(base, x, rcond=None)[0]
    extra = extra - base @ np.linalg.lstsq(base, extra, rcond=None)[0]
    rank = np.linalg.matrix_rank(extra, tol=1e-10)
    if rank == 0 or len(x) <= np.linalg.matrix_rank(base) + rank:
        return np.nan, rank
    rss = float(np.sum(residual**2))
    if rss <= 1e-14:
        return np.nan, rank
    final = residual - extra @ np.linalg.lstsq(extra, residual, rcond=1e-10)[0]
    return float((rss - np.sum(final**2)) / rss), rank
