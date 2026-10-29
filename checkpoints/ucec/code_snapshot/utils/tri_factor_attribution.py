"""Patch-level Input x Gradient attribution for the three CFSurv branches.

These targets have different meanings and units: site log probability,
prototype log probability, and risk through z_u with conditioning held fixed.
They are not an additive decomposition of the patient's survival risk.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


FACTOR_NAMES = ("z_b", "z_s", "z_u")


@contextmanager
def fixed_selection_gradients(model):
    """Keep inference values, excluding training-only straight-through gradients.

    Hard Slot membership and capacity are discrete inference decisions. The
    attribution is local, conditional on these decisions. Attention and the
    softmax gate among retained Slots remain differentiable.
    """
    compressor = model.path_compressor
    name = "_stable_masked_slot_gate"
    if not hasattr(compressor, name):
        from utils import legacy_probe_path_compressor as legacy

        if not isinstance(compressor, legacy.CFSurvPathSlotCompressor):
            raise ValueError("Unknown path compressor; cannot fix Slot selection gradients.")
        function_name = compressor.top_k_method
        original_topk = getattr(legacy, function_name)

        def fixed_topk(*args, **kwargs):
            keep_mask, selected = original_topk(*args, **kwargs)
            return keep_mask.detach(), selected

        setattr(legacy, function_name, fixed_topk)
        try:
            yield
        finally:
            setattr(legacy, function_name, original_topk)
        return
    original = getattr(compressor, name)
    had_instance_method = name in compressor.__dict__

    def gate(keep_logits, keep_mask):
        return original(keep_logits, keep_mask.detach())

    setattr(compressor, name, gate)
    try:
        yield
    finally:
        if had_instance_method:
            setattr(compressor, name, original)
        else:
            delattr(compressor, name)


def branch_targets(model, output, site_index=None, prototype_index=None):
    """Build scalar, role-specific targets without training stop-gradient paths."""
    if not output.get("tri_enabled") or any(k not in output for k in FACTOR_NAMES):
        raise ValueError("Heatmaps require all three factors: z_b, z_s, z_u.")
    if output["z_b"].shape[0] != 1:
        raise ValueError("Attribute one patient bag at a time.")

    # The site loss can use E_b(h.detach()). Re-evaluate its head on the
    # connected z_b; the value is identical in eval mode, and dT_b/dX exists.
    site_logits = model.site_bg_head(output["z_b"])
    if site_logits.shape[-1] < 2:
        raise ValueError("Site evidence requires a checkpoint with >= 2 sites.")
    if output["proto_assign"].shape[-1] < 2:
        raise ValueError("Prototype evidence requires >= 2 prototypes.")
    predicted_site = int(site_logits.argmax(dim=-1).item())
    dominant_prototype = int(output["proto_assign"].argmax(dim=-1).item())
    site_index = predicted_site if site_index is None else int(site_index)
    prototype_index = dominant_prototype if prototype_index is None else int(prototype_index)
    if not 0 <= site_index < site_logits.shape[-1]:
        raise ValueError("Site target index is outside the checkpoint's site vocabulary.")
    if not 0 <= prototype_index < output["proto_assign"].shape[-1]:
        raise ValueError("Prototype target index is outside the prototype bank.")

    logits, _ = model._risk_from_factors(output["z_u"], output["proto_assign"].detach())
    if output.get("bg_calibration_enabled"):
        logits = logits + model.bg_calibration_weight * output["bg_calibration_logits"].detach()
    hazards = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    risk_u = -torch.cumprod(1 - hazards, dim=1).sum()
    full_risk = -output["S"][0].sum()
    torch.testing.assert_close(risk_u.detach(), full_risk.detach())

    targets = (
        F.log_softmax(site_logits, dim=-1)[0, site_index],
        F.log_softmax(output["proto_logits"], dim=-1)[0, prototype_index],
        risk_u,
    )
    details = {
        "site_target_index": site_index,
        "predicted_site_index": predicted_site,
        "prototype_target_index": prototype_index,
        "dominant_prototype_index": dominant_prototype,
        "site_target_probability": float(site_logits.softmax(-1)[0, site_index].detach()),
        "prototype_probabilities": output["proto_assign"][0].detach().cpu().tolist(),
        "risk": float(full_risk.detach()),
        "target_values": {key: float(value.detach()) for key, value in zip(FACTOR_NAMES, targets)},
    }
    return targets, details


def attribute_patches(model, x_path, x_omic, *, site_index=None, prototype_index=None, seed=0):
    """Return signed [3, N] patch scores, using sum_d x[id] * dT/dx[id].

    Genomics remains fixed. All targets use one forward pass and one sampled
    Slot initialization. No parameter gradients are accumulated or updated.
    """
    if x_path.ndim != 3 or x_path.shape[0] != 1 or x_path.shape[1] == 0:
        raise ValueError("x_path must be a nonempty [1, N, D] tensor.")
    if x_omic.ndim != 2 or x_omic.shape[0] != 1:
        raise ValueError("x_omic must be [1, G].")
    if not torch.isfinite(x_path).all() or not torch.isfinite(x_omic).all():
        raise ValueError("Attribution inputs must contain only finite values.")
    modes = [(module, module.training) for module in model.modules()]
    devices = list(range(torch.cuda.device_count())) if x_path.is_cuda else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.enable_grad(), fixed_selection_gradients(model):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            patches = x_path.detach().clone().requires_grad_(True)
            output = model(patches, x_omic.detach(), phase="test")
            targets, details = branch_targets(model, output, site_index, prototype_index)
            scores = []
            for index, target in enumerate(targets):
                gradient, = torch.autograd.grad(target, patches, retain_graph=index < len(targets) - 1)
                scores.append((patches.detach() * gradient).sum(dim=-1)[0].detach().cpu())
            signed = torch.stack(scores)
            if not torch.isfinite(signed).all():
                raise ValueError("Attribution produced nonfinite scores.")
            path_viz = output["viz"]["path_compressor"]
            details["slot_gate"] = path_viz["slot_gate"].tolist()
            details["slot_keep_mask"] = path_viz["slot_keep_mask"].tolist()
            if "selected_k" in path_viz:
                details["selected_k"] = path_viz["selected_k"].tolist()
            details.update({
                "method": "input_x_gradient",
                "seed": int(seed),
                "targets": {
                    "z_b": "log p(site_target | z_b)",
                    "z_s": "log p(prototype_target | z_s)",
                    "z_u": "-sum(S), differentiating through z_u only; prototype assignment and background calibration fixed",
                },
                "selection_gradient": (
                    "hard Slot membership and capacity fixed; retained Slot gates differentiable"
                    if hasattr(model.path_compressor, "_stable_masked_slot_gate") else
                    "legacy hard Slot membership fixed; retained Slot gates differentiable"
                ),
                "interpretation": "Local feature-space attribution; not causal effects or additive risk shares.",
            })
            return signed.numpy(), details
    finally:
        for module, training in modes:
            module.training = training
