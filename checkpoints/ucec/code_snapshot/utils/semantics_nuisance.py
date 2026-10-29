"""Training-only nuisance fitting for survival residual GCM and IPCW checks."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from utils.semantics_stats import (
    FeatureMap, MarginalCensoring, administrative_censor, brier_rows,
    conditional_kernel, positive_median_distance, multiplier_gcm,
)


class ConditionMap:
    def fit(self, site, a, clinical=None):
        self.sites = np.unique(site)
        self.clinical = None if clinical is None else np.unique(clinical)
        self.scaler = StandardScaler().fit(a)
        return self

    def transform(self, site, a, clinical=None):
        parts = [np.equal.outer(site, self.sites).astype(float), self.scaler.transform(a)]
        if self.clinical is not None:
            parts.append(np.equal.outer(clinical, self.clinical).astype(float))
        return np.concatenate(parts, axis=1)


@dataclass
class ForestNuisance:
    mapping: ConditionMap
    model: RandomSurvivalForest
    leaf: int

    def values(self, site, a, query, clinical=None, cumulative=False):
        x = self.mapping.transform(site, a, clinical)
        values = (self.model.predict_cumulative_hazard_function(x, return_array=True) if cumulative else
                  self.model.predict_survival_function(x, return_array=True))
        query = np.asarray(query, float)
        if query.ndim == 1:
            # Each patient has its own observed-time query.
            query = query[:, None]
        if query.shape[0] != len(x):
            query = np.broadcast_to(query, (len(x), query.shape[-1]))
        index = np.searchsorted(self.model.unique_times_, query, side="right") - 1
        result = values[np.arange(len(x))[:, None], np.maximum(index, 0)]
        result[index < 0] = 0. if cumulative else 1.
        result[query > self.model.unique_times_[-1]] = np.nan
        return result


def fit_forest(site, a, time, event, leaf, config, seed, clinical=None):
    if not np.asarray(event, bool).any():
        raise ValueError("No events for survival nuisance fitting")
    mapping = ConditionMap().fit(site, a, clinical)
    forest = RandomSurvivalForest(n_estimators=config.trees, max_depth=5, max_features=1.,
                                  min_samples_leaf=leaf, random_state=seed, n_jobs=1,
                                  low_memory=False)
    forest.fit(mapping.transform(site, a, clinical), Surv.from_arrays(event, time))
    return ForestNuisance(mapping, forest, leaf)


def select_survival(site, a, time, event, cuts, fit, validation, config, seed, clinical=None):
    site, a, time, event = map(np.asarray, (site, a, time, event))
    cf = None if clinical is None else np.asarray(clinical)[fit]
    cv = None if clinical is None else np.asarray(clinical)[validation]
    vt, ve, cap = administrative_censor(time[fit], time[validation], event[validation])
    censor = MarginalCensoring().fit(time[fit], event[fit])
    gt = censor.predict(vt)
    diagnostics, best, chosen = [], np.inf, None
    for leaf in sorted(config.leaf_sizes, reverse=True):
        forest = fit_forest(site[fit], a[fit], time[fit], event[fit], leaf, config, seed, cf)
        pred = forest.values(site[validation], a[validation], np.asarray(cuts)[None, :], cv)
        scores = []
        for j, cut in enumerate(cuts):
            if not (vt.min() <= cut < vt.max() and cut < cap):
                scores.append(np.nan)
                continue
            rows, valid = brier_rows(vt, ve, pred[:, j], cut, gt, censor.predict(np.array([cut]))[0])
            # Use the same whole internal validation population for all candidates.
            scores.append(float(rows.mean()) if valid.all() else np.nan)
        value = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.inf
        diagnostics.append({"min_samples_leaf": leaf, "brier": scores, "mean_brier": value,
                            "internal_train_n": len(fit), "internal_validation_n": len(validation)})
        if value < best - 1e-12:
            best, chosen = value, leaf
    if chosen is None:
        raise ValueError("No estimable internal IPCW Brier endpoint for survival nuisance selection")
    final = fit_forest(site, a, time, event, chosen, config, seed, clinical)
    return final, {"selected_leaf": chosen, "internal_scores": diagnostics,
                   "criterion": "mean of estimable IPCW Brier at three original training cuts",
                   "ties": "larger leaf", "training_n": len(time)}


@dataclass
class KernelMean:
    condition_scaler: StandardScaler
    train_site: np.ndarray
    train_a: np.ndarray
    train_clinical: np.ndarray | None
    bandwidth2: float
    kernel_mean: np.ndarray
    kernel_grand_mean: float
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict(self, site, a, clinical=None):
        k = conditional_kernel(site, self.condition_scaler.transform(a), self.train_site,
                               self.train_a, self.bandwidth2, clinical, self.train_clinical)
        centered = k - k.mean(1)[:, None] - self.kernel_mean[None, :] + self.kernel_grand_mean
        return self.target_mean + centered @ self.coefficients


def _kernel_setup(site, a, target, multiplier, clinical=None):
    scaler = StandardScaler().fit(a)
    scaled = scaler.transform(a)
    bandwidth2 = positive_median_distance(scaled) * multiplier**2
    k = conditional_kernel(site, scaled, site, scaled, bandwidth2, clinical, clinical)
    km, kg = k.mean(0), float(k.mean())
    kc = k - km[None, :] - km[:, None] + kg
    eigenvalues, u = eigh(kc, check_finite=False, driver="evr")
    eigenvalues = np.maximum(eigenvalues, 0.)
    tm = target.mean(0)
    projected_target = u.T @ (target-tm)
    return scaler, scaled, bandwidth2, km, kg, eigenvalues, u, tm, projected_target


def fit_feature_mean(x, site, a, fit, validation, config, seed, clinical=None):
    """Tune on inner split; re-fit mapping and KRR using all original train."""
    x, site, a = map(np.asarray, (x, site, a))
    cf = None if clinical is None else np.asarray(clinical)[fit]
    cv = None if clinical is None else np.asarray(clinical)[validation]
    fmap = FeatureMap(config.rff_features, seed).fit(x[fit])
    target, val_target = fmap.transform(x[fit]), fmap.transform(x[validation])
    diagnostics, best, selected = [], np.inf, None
    for multiplier in sorted(config.bandwidth_multipliers, reverse=True):
        scaler, scaled, bw, km, kg, ev, u, tm, ut = _kernel_setup(site[fit], a[fit], target, multiplier, cf)
        kv = conditional_kernel(site[validation], scaler.transform(a[validation]), site[fit],
                                scaled, bw, cv, cf)
        kv = kv - kv.mean(1)[:, None] - km[None, :] + kg
        kvu = kv @ u
        for alpha in sorted(config.ridge_alphas, reverse=True):
            prediction = tm + (kvu / (ev + len(fit)*alpha)[None, :]) @ ut
            mse = float(np.mean((prediction-val_target)**2))
            diagnostics.append({"bandwidth_multiplier": multiplier, "alpha": alpha, "standardized_mse": mse})
            if (mse < best - 1e-12 or
                (abs(mse-best) <= 1e-12 and (alpha, multiplier) > (selected[1], selected[0]))):
                best, selected = mse, (multiplier, alpha)
    multiplier, alpha = selected
    fmap = FeatureMap(config.rff_features, seed).fit(x)
    target = fmap.transform(x)
    scaler, scaled, bw, km, kg, ev, u, tm, ut = _kernel_setup(site, a, target, multiplier, clinical)
    model = KernelMean(scaler, site, scaled, clinical, bw, km, kg, tm,
                       u @ (ut / (ev + len(x)*alpha)[:, None]), alpha)
    return fmap, model, {"selected_multiplier": multiplier, "selected_alpha": alpha,
                         "internal_standardized_mse": best, "internal_scores": diagnostics,
                         "kernel_regularization": "K_centered + n_train * alpha * I",
                         "kernel_intercept": "centered kernel and target; unpenalized target mean",
                         "feature_dimensions": target.shape[1], "rff_seed": seed,
                         "rff_bandwidth_squared": fmap.bandwidth2,
                         "training_n": len(x), "constant_features": int(np.sum(fmap.output_scaler.var_ <= 1e-14))}


def patient_association(train, evaluation, fit, validation, cuts, config, seed,
                        draws=None, clinical=False, save_models=False):
    """Inputs are dictionaries; no evaluation outcome enters any fit/selection."""
    tc = train.get("clinical") if clinical else None
    ec = evaluation.get("clinical") if clinical else None
    survival, sd = select_survival(train["site"], train["a"], train["time"], train["event"],
                                    cuts, fit, validation, config, seed, tc)
    et, ee, cap = administrative_censor(train["time"], evaluation["time"], evaluation["event"])
    hazard = survival.values(evaluation["site"], evaluation["a"], et, ec, cumulative=True)[:, 0]
    m = ee.astype(float)-hazard
    if not np.isfinite(m).all():
        raise ValueError("Nonfinite survival nuisance residual")
    result = {"survival_diagnostic": sd, "cap": cap,
              "event_after_admin": ee, "time_after_admin": et,
              "martingale": m, "hazard": hazard, "representations": {}, "models": {}}
    if save_models:
        result["models"]["survival"] = survival
    for representation, x in train["representations"].items():
        fmap, mean, diagnostic = fit_feature_mean(x, train["site"], train["a"], fit, validation,
                                                  config, seed, tc)
        phi = fmap.transform(evaluation["representations"][representation])
        mu = mean.predict(evaluation["site"], evaluation["a"], ec)
        residual = phi-mu
        products = m[:, None]*residual
        stat, null = multiplier_gcm(products, config.null_draws if draws is None else draws, seed)
        stat.update(martingale_mean=float(m.mean()), martingale_variance=float(m.var()),
                    representation_residual_variance=float(np.mean(residual.var(0))))
        result["representations"][representation] = {
            "stat": stat, "null": null, "diagnostic": diagnostic,
            "phi": phi, "conditional_mean": mu, "residual": residual, "products": products}
        if save_models:
            result["models"][representation] = {"feature_map": fmap, "conditional_mean": mean}
    return result
