from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

_EPS = 1e-6


def _as_mask(mask: torch.Tensor, threshold: float = 0.30) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape (B,H,W) or (B,1,H,W)")
    return (mask.float() > float(threshold)).float()


def _cross_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)


def _boundary4(mask: torch.Tensor) -> torch.Tensor:
    neighbours = F.conv2d(mask, _cross_kernel(mask.device, mask.dtype), padding=1)
    interior = mask * (neighbours >= 3.999).float()
    return (mask - interior).clamp(0.0, 1.0)


def _coordinate_grid(batch: int, height: int, width: int, device, dtype):
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    try:
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(yy, xx)
    gx = gx.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)
    gy = gy.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)
    return gx, gy


class PoissonHarmonicField(nn.Module):
    """Parameter-free support-induced anatomical field.

    Output channels are [occupancy, poisson, h_E, h_N, h_W, h_S].
    Only the five geometry channels are transferred to the query; occupancy is
    never transported or diffused.
    """

    def __init__(
        self,
        poisson_iterations: int = 56,
        harmonic_iterations: int = 56,
        directional_kappa: float = 4.0,
        mask_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.poisson_iterations = int(poisson_iterations)
        self.harmonic_iterations = int(harmonic_iterations)
        self.directional_kappa = float(directional_kappa)
        self.mask_threshold = float(mask_threshold)

    @torch.no_grad()
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask = _as_mask(mask, self.mask_threshold)
        batch, _, height, width = mask.shape
        boundary = _boundary4(mask)
        interior = (mask - boundary).clamp(0.0, 1.0)
        kernel = _cross_kernel(mask.device, mask.dtype)

        # -Delta rho = 1 inside the support organ, rho = 0 on its boundary.
        rho = torch.zeros_like(mask)
        for _ in range(self.poisson_iterations):
            neigh = F.conv2d(rho, kernel, padding=1)
            rho = interior * 0.25 * (neigh + 1.0)
        rho_max = rho.flatten(1).amax(dim=1, keepdim=True).clamp_min(_EPS)
        rho = rho / rho_max[:, :, None, None]
        rho = rho * mask

        # Boundary-anchored directional harmonic coordinates.
        gx, gy = _coordinate_grid(batch, height, width, mask.device, mask.dtype)
        mass = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(_EPS)
        cx = (gx * mask).sum(dim=(-2, -1), keepdim=True) / mass
        cy = (gy * mask).sum(dim=(-2, -1), keepdim=True) / mass
        theta = torch.atan2(gy - cy, gx - cx)

        phis = mask.new_tensor([0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi])
        logits = self.directional_kappa * torch.cos(
            theta.unsqueeze(1) - phis.view(1, 4, 1, 1, 1)
        ).squeeze(2)
        boundary_prob = F.softmax(logits, dim=1) * boundary

        harmonic = boundary_prob.clone()
        harmonic_kernel = kernel.expand(4, 1, 3, 3).contiguous()
        interior4 = interior.expand(-1, 4, -1, -1)
        for _ in range(self.harmonic_iterations):
            neigh = F.conv2d(harmonic, harmonic_kernel, padding=1, groups=4)
            harmonic = boundary_prob + interior4 * 0.25 * neigh
        harmonic = harmonic.clamp(0.0, 1.0) * mask

        return torch.cat((mask, rho, harmonic), dim=1)


def build_field_at_resolution(
    builder: PoissonHarmonicField,
    mask: torch.Tensor,
    build_size: Tuple[int, int],
    out_size: Tuple[int, int],
) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask_build = F.interpolate(
        mask.float(), size=build_size, mode="bilinear", align_corners=False
    )
    field = builder(mask_build)
    if tuple(build_size) != tuple(out_size):
        field = F.interpolate(
            field, size=out_size, mode="bilinear", align_corners=False
        )
    return field.clamp(0.0, 1.0)


def _spatial_standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    var = (x - mean).square().mean(dim=(-2, -1), keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5)


class ResidualCorrelationAdapter(nn.Module):
    """Near-identity adapter that preserves the pretrained correlation space."""

    def __init__(self, channels: int = 256, hidden: int = 128) -> None:
        super().__init__()
        groups = 8 if hidden % 8 == 0 else 1
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        self.scale_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 0.35 * torch.sigmoid(self.scale_raw)
        y = x + scale * self.adapter(_spatial_standardize(x))
        y = _spatial_standardize(y)
        return F.normalize(y, dim=1, eps=1e-6)


class EpisodeCalibratedAnatomicalTrust(nn.Module):
    """Parameter-free, target-label-free calibration of anatomical reliability.

    Absolute reciprocal/coherence values can shift across modalities and
    anatomies. A fixed quality threshold therefore over-trusts geometry in some
    episodes and suppresses useful semantics in others. This module calibrates
    the field quality *inside each target episode* using high-confidence
    semantic foreground/background anchors, then permits anatomical correction
    primarily where the semantic matcher is uncertain.

    No target mask, domain ID, fine-tuning, or learnable domain classifier is
    used at inference.
    """

    def __init__(
        self,
        fg_anchor: float = 0.70,
        bg_anchor: float = 0.30,
        anchor_softness: float = 0.08,
        separation_margin: float = 0.06,
        uncertainty_floor: float = 0.10,
    ) -> None:
        super().__init__()
        self.fg_anchor = float(fg_anchor)
        self.bg_anchor = float(bg_anchor)
        self.anchor_softness = float(anchor_softness)
        self.separation_margin = float(separation_margin)
        self.uncertainty_floor = float(uncertainty_floor)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dims = (-2, -1)
        return (value * weight).sum(dim=dims, keepdim=True) / (
            weight.sum(dim=dims, keepdim=True).clamp_min(_EPS)
        )

    def forward(
        self,
        semantic_prob: torch.Tensor,
        reciprocal_confidence: torch.Tensor,
        coherence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # The trust estimator is intentionally non-backpropagating: geometry is
        # not allowed to reshape the semantic correlation space during training.
        p = semantic_prob.detach().clamp(1e-4, 1.0 - 1e-4)
        quality = (reciprocal_confidence.detach() * coherence.detach()).clamp(0.0, 1.0)

        softness = max(self.anchor_softness, 1e-4)
        fg_weight = torch.sigmoid((p - self.fg_anchor) / softness)
        bg_weight = torch.sigmoid((self.bg_anchor - p) / softness)

        q_fg = self._weighted_mean(quality, fg_weight)
        q_bg = self._weighted_mean(quality, bg_weight)

        # Geometry is trusted only if its quality is actually higher on the
        # episode's semantic foreground anchors than on background anchors.
        separation = (q_fg - q_bg).clamp_min(0.0)
        margin = max(self.separation_margin, 1e-4)
        separation_confidence = separation / (separation + margin)

        midpoint = 0.5 * (q_fg + q_bg)
        normalized_quality = (quality - midpoint) / (separation + margin)
        field_evidence = torch.tanh(normalized_quality)

        # Maximum at p=0.5 and minimum at confident semantic decisions.
        semantic_uncertainty = 4.0 * p * (1.0 - p)
        ambiguity_gate = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * semantic_uncertainty

        trust = (separation_confidence * ambiguity_gate).clamp(0.0, 1.0)
        adjustment = trust * field_evidence

        diagnostics = {
            "quality": quality,
            "q_fg": q_fg,
            "q_bg": q_bg,
            "quality_separation": separation,
            "separation_confidence": separation_confidence,
            "semantic_uncertainty": semantic_uncertainty,
            "trust": trust,
            "field_evidence": field_evidence,
        }
        return adjustment, diagnostics


class ReliabilityAdaptiveFieldMatcher(nn.Module):
    """SIAF-RA core: strong semantic matching + passive field + adaptive trust.

    Compared with SIAF-V2, there is deliberately no geometry re-ranking and no
    query-field completion. The support-induced field is a second source of
    evidence whose reliability is calibrated per episode before it can modify
    the strong semantic decision.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        adapter_hidden: int = 128,
        topk: int = 12,
        semantic_temperature: float = 10.0,
        affinity_temperature: float = 0.09,
        reciprocal_weight: float = 0.35,
        coherence_temperature: float = 0.06,
        max_field_scale: float = 2.0,
        initial_field_scale: float = 0.60,
        fg_anchor: float = 0.70,
        bg_anchor: float = 0.30,
        anchor_softness: float = 0.08,
        separation_margin: float = 0.06,
        uncertainty_floor: float = 0.10,
    ) -> None:
        super().__init__()
        self.support_adapter = ResidualCorrelationAdapter(feat_dim, adapter_hidden)
        self.query_adapter = ResidualCorrelationAdapter(feat_dim, adapter_hidden)
        self.topk = int(topk)
        self.affinity_temperature = float(affinity_temperature)
        self.reciprocal_weight = float(reciprocal_weight)
        self.coherence_temperature = float(coherence_temperature)
        self.max_field_scale = float(max_field_scale)

        sem_ratio = (float(semantic_temperature) - 4.0) / 16.0
        sem_ratio = max(min(sem_ratio, 1.0 - 1e-4), 1e-4)
        self.semantic_scale_raw = nn.Parameter(
            torch.tensor(math.log(sem_ratio / (1.0 - sem_ratio)))
        )

        field_ratio = float(initial_field_scale) / max(self.max_field_scale, 1e-4)
        field_ratio = max(min(field_ratio, 1.0 - 1e-4), 1e-4)
        self.field_scale_raw = nn.Parameter(
            torch.tensor(math.log(field_ratio / (1.0 - field_ratio)))
        )

        self.trust_calibrator = EpisodeCalibratedAnatomicalTrust(
            fg_anchor=fg_anchor,
            bg_anchor=bg_anchor,
            anchor_softness=anchor_softness,
            separation_margin=separation_margin,
            uncertainty_floor=uncertainty_floor,
        )

    def semantic_scale(self) -> torch.Tensor:
        return 4.0 + 16.0 * torch.sigmoid(self.semantic_scale_raw)

    def field_scale(self) -> torch.Tensor:
        return self.max_field_scale * torch.sigmoid(self.field_scale_raw)

    @staticmethod
    def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _topk_mean(
        similarity: torch.Tensor,
        valid_mask: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        valid_idx = torch.where(valid_mask)[0]
        selected = similarity if valid_idx.numel() == 0 else similarity[:, :, valid_idx]
        k = max(1, min(int(topk), selected.shape[-1]))
        return selected.topk(k=k, dim=-1).values.mean(dim=-1)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits.masked_fill(~valid_mask.view(1, 1, -1), -1e4), dim=-1)

    def forward(
        self,
        support_feature: torch.Tensor,
        query_feature: torch.Tensor,
        support_fg_mask: torch.Tensor,
        support_bg_mask: torch.Tensor,
        support_geometry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if query_feature.shape[0] != 1:
            raise ValueError("SIAF-RA expects one query per episode")

        shots, channels, height, width = support_feature.shape
        q_corr = self.query_adapter(query_feature)
        s_corr = self.support_adapter(support_feature)

        q_flat = self._flatten_feature(q_corr)
        s_flat = self._flatten_feature(s_corr).reshape(
            1, shots * height * width, channels
        )
        similarity = torch.bmm(q_flat, s_flat.transpose(1, 2)).clamp(-1.0, 1.0)

        fg_flat = support_fg_mask.reshape(-1)
        bg_flat = support_bg_mask.reshape(-1)
        fg_valid = fg_flat > 0.20
        bg_valid = bg_flat > 0.20
        if fg_valid.sum() == 0:
            fg_valid = fg_flat > 0.05
        if bg_valid.sum() == 0:
            bg_valid = bg_flat > 0.05

        # Strong semantic path: identical logic to the successful semantic-only
        # ablation and always retained as the base decision.
        fg_score = self._topk_mean(similarity, fg_valid, self.topk)
        bg_score = self._topk_mean(similarity, bg_valid, self.topk)
        semantic_margin = fg_score - bg_score
        semantic_logit = self.semantic_scale() * semantic_margin
        semantic_prob = torch.sigmoid(semantic_logit)

        tau = max(self.affinity_temperature, 1e-4)
        sim_tau = similarity / tau
        support_to_query = F.softmax(sim_tau, dim=1)
        base_fg_logits = sim_tau + self.reciprocal_weight * torch.log(
            support_to_query.clamp_min(_EPS)
        )
        fg_weights = self._masked_softmax(base_fg_logits, fg_valid)

        geometry_dim = support_geometry.shape[1]
        geom_flat = support_geometry.flatten(2).transpose(1, 2).reshape(
            1, shots * height * width, geometry_dim
        )
        query_geometry = torch.bmm(fg_weights, geom_flat)

        n_query = similarity.shape[1]
        reciprocal_mass = (fg_weights * support_to_query).sum(dim=-1) * float(n_query)
        reciprocal_conf = torch.sigmoid(
            2.0 * (torch.log(reciprocal_mass.clamp_min(_EPS)) - math.log(2.0))
        )

        geom_delta = query_geometry.unsqueeze(2) - geom_flat.unsqueeze(1)
        geom_var = (fg_weights.unsqueeze(-1) * geom_delta.square()).sum(dim=2).mean(dim=-1)
        coherence = torch.exp(
            -geom_var / max(self.coherence_temperature, 1e-4)
        ).clamp(0.0, 1.0)

        semantic_logit_map = semantic_logit.view(1, 1, height, width)
        semantic_prob_map = semantic_prob.view(1, 1, height, width)
        reciprocal_map = reciprocal_conf.view(1, 1, height, width)
        coherence_map = coherence.view(1, 1, height, width)
        query_geometry_map = query_geometry.transpose(1, 2).view(
            1, geometry_dim, height, width
        )

        adjustment, trust_diag = self.trust_calibrator(
            semantic_prob_map, reciprocal_map, coherence_map
        )
        # Trust/evidence are detached by the calibrator. Geometry can alter the
        # prediction but cannot distort the semantic representation during
        # backpropagation. Only the bounded global intervention amplitude learns.
        final_logit = semantic_logit_map + self.field_scale() * adjustment
        final_logit = final_logit.clamp(-12.0, 12.0)

        diagnostics = {
            "query_corr": q_corr,
            "support_corr": s_corr,
            "similarity": similarity,
            "fg_weights": fg_weights,
            "semantic_margin": semantic_margin.view(1, 1, height, width),
            "semantic_logit": semantic_logit_map,
            "semantic_prob": semantic_prob_map,
            "query_geometry": query_geometry_map,
            "reciprocal_confidence": reciprocal_map,
            "coherence": coherence_map,
            "final_logit": final_logit,
            **trust_diag,
        }
        return final_logit, semantic_logit_map, diagnostics
