from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .anatomical_field_ra import (
    PoissonHarmonicField,
    ReliabilityAdaptiveFieldMatcher,
    build_field_at_resolution,
)
from .backbone.torchvision_backbones import (
    TVDeeplabRes101Encoder,
    TVDeeplabRes50Encoder,
)

_EPS = 1e-6


class FewShotSeg(nn.Module):
    """SIAF-RA: Reliability-Adaptive Support-Induced Anatomical Fields.

    Design after the SIAF-V2 ablation study:
      1. preserve the strong direct FG/BG semantic matcher;
      2. induce Poisson-Harmonic geometry only from support foreground;
      3. do NOT re-rank semantic correspondence and do NOT complete/diffuse it;
      4. self-calibrate anatomical reliability inside each target episode;
      5. let geometry intervene conservatively, mainly on uncertain semantics;
      6. train an explicit semantic anchor objective so geometry cannot destroy
         the strong semantic baseline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained_path: Optional[str] = None,
        cfg: Optional[Dict] = None,
        sdf_criterion=None,
    ) -> None:
        super().__init__()
        del in_channels, sdf_criterion
        self.pretrained_path = pretrained_path
        self.config = cfg or {}
        self.get_encoder()

        ra_cfg = self.config.get("siaf_ra", {})
        feat_dim = int(ra_cfg.get("feat_dim", 256))
        self.field_build_size = int(ra_cfg.get("field_build_size", 64))

        self.field_builder = PoissonHarmonicField(
            poisson_iterations=int(ra_cfg.get("poisson_iterations", 56)),
            harmonic_iterations=int(ra_cfg.get("harmonic_iterations", 56)),
            directional_kappa=float(ra_cfg.get("directional_kappa", 4.0)),
            mask_threshold=float(ra_cfg.get("field_mask_threshold", 0.30)),
        )

        self.matcher = ReliabilityAdaptiveFieldMatcher(
            feat_dim=feat_dim,
            adapter_hidden=int(ra_cfg.get("adapter_hidden", 128)),
            topk=int(ra_cfg.get("topk", 12)),
            semantic_temperature=float(ra_cfg.get("semantic_temperature", 10.0)),
            affinity_temperature=float(ra_cfg.get("affinity_temperature", 0.09)),
            reciprocal_weight=float(ra_cfg.get("reciprocal_weight", 0.35)),
            coherence_temperature=float(ra_cfg.get("coherence_temperature", 0.06)),
            max_field_scale=float(ra_cfg.get("max_field_scale", 2.0)),
            initial_field_scale=float(ra_cfg.get("initial_field_scale", 0.60)),
            fg_anchor=float(ra_cfg.get("fg_anchor", 0.70)),
            bg_anchor=float(ra_cfg.get("bg_anchor", 0.30)),
            anchor_softness=float(ra_cfg.get("anchor_softness", 0.08)),
            separation_margin=float(ra_cfg.get("separation_margin", 0.06)),
            uncertainty_floor=float(ra_cfg.get("uncertainty_floor", 0.10)),
        )

        self._last_siaf_ra = {}
        if self.pretrained_path:
            self._load_checkpoint(self.pretrained_path)

    def get_encoder(self) -> None:
        model_name = self.config.get("which_model", "dlfcn_res101")
        use_coco_init = bool(self.config.get("use_coco_init", True))
        if model_name == "dlfcn_res101":
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif model_name == "dlfcn_res50":
            self.encoder = TVDeeplabRes50Encoder(use_coco_init)
        else:
            raise NotImplementedError("Unsupported backbone: %s" % model_name)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]

        current = self.state_dict()
        cleaned = {}
        skipped = []
        for key, value in checkpoint.items():
            key = key[7:] if key.startswith("module.") else key
            if key in current and current[key].shape == value.shape:
                cleaned[key] = value
            elif key in current:
                skipped.append((key, tuple(value.shape), tuple(current[key].shape)))

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if skipped:
            print("[warn] skipped shape-mismatched keys:", skipped[:10])
        if missing:
            print("[warn] missing keys (%d): %s" % (len(missing), missing[:15]))
        if unexpected:
            print("[warn] unexpected keys (%d): %s" % (len(unexpected), unexpected[:15]))
        print("###### SIAF-RA checkpoint loaded: %s ######" % checkpoint_path)

    @staticmethod
    def _stack_support_masks(mask_nested) -> torch.Tensor:
        masks = []
        for shot_mask in mask_nested[0]:
            if shot_mask.dim() == 3:
                shot_mask = shot_mask.unsqueeze(1)
            masks.append(shot_mask.float())
        return torch.cat(masks, dim=0)

    @staticmethod
    def _resize_class_masks(
        fg_full: torch.Tensor,
        bg_full: torch.Tensor,
        size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fg = F.interpolate(
            fg_full, size=size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        bg = F.interpolate(
            bg_full, size=size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        denom = (fg + bg).clamp_min(_EPS)
        return fg / denom, bg / denom

    def _build_support_geometry(
        self,
        mask_full: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        build_h = min(self.field_build_size, int(mask_full.shape[-2]))
        build_w = min(self.field_build_size, int(mask_full.shape[-1]))
        full_field = build_field_at_resolution(
            self.field_builder,
            mask_full,
            build_size=(build_h, build_w),
            out_size=out_size,
        )
        # occupancy is deliberately excluded from transfer.
        return full_field[:, 1:6]

    @staticmethod
    def _two_class(logit: torch.Tensor) -> torch.Tensor:
        return torch.cat((-0.5 * logit, 0.5 * logit), dim=1)

    def forward(
        self,
        supp_imgs,
        fore_mask,
        back_mask,
        qry_imgs,
        isval=False,
        val_wsize=None,
        sdf_gt=None,
        query_gt=None,
    ):
        del isval, val_wsize, sdf_gt, query_gt

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        if n_ways != 1:
            raise NotImplementedError("SIAF-RA supports 1-way episodes")
        if n_queries != 1:
            raise NotImplementedError("SIAF-RA supports one query per episode")
        if supp_imgs[0][0].shape[0] != 1 or qry_imgs[0].shape[0] != 1:
            raise ValueError("SIAF-RA expects episodic batch size one")

        image_size = qry_imgs[0].shape[-2:]
        all_images = torch.cat(
            [torch.cat(supp_imgs[0], dim=0), qry_imgs[0]], dim=0
        )
        all_features = self.encoder(all_images, low_level=False)
        feature_size = all_features.shape[-2:]
        support_feature = all_features[:n_shots]
        query_feature = all_features[n_shots:n_shots + 1]

        support_fg_full = self._stack_support_masks(fore_mask)
        support_bg_full = self._stack_support_masks(back_mask)
        support_fg, support_bg = self._resize_class_masks(
            support_fg_full, support_bg_full, feature_size
        )
        support_geometry = self._build_support_geometry(
            support_fg_full, feature_size
        )

        final_logit, semantic_logit, diagnostics = self.matcher(
            support_feature=support_feature,
            query_feature=query_feature,
            support_fg_mask=support_fg,
            support_bg_mask=support_bg,
            support_geometry=support_geometry,
        )

        output = F.interpolate(
            self._two_class(final_logit),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        semantic_output = F.interpolate(
            self._two_class(semantic_logit),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

        self._last_query_feat = query_feature.detach()
        self._last_sup_feats = [
            support_feature[s:s + 1].detach() for s in range(n_shots)
        ]
        self._last_siaf_ra = {
            "semantic_logit": semantic_logit.detach(),
            "final_logit": final_logit.detach(),
            "semantic_prob": diagnostics["semantic_prob"].detach(),
            "query_geometry": diagnostics["query_geometry"].detach(),
            "reciprocal_confidence": diagnostics["reciprocal_confidence"].detach(),
            "coherence": diagnostics["coherence"].detach(),
            "quality": diagnostics["quality"].detach(),
            "quality_separation": diagnostics["quality_separation"].detach(),
            "separation_confidence": diagnostics["separation_confidence"].detach(),
            "semantic_uncertainty": diagnostics["semantic_uncertainty"].detach(),
            "trust": diagnostics["trust"].detach(),
            "field_evidence": diagnostics["field_evidence"].detach(),
            "q_fg": diagnostics["q_fg"].detach(),
            "q_bg": diagnostics["q_bg"].detach(),
            "semantic_scale": self.matcher.semantic_scale().detach(),
            "field_scale": self.matcher.field_scale().detach(),
        }

        # Third return is kept for compatibility with the old train/test call
        # signature; it is a zero scalar because SIAF-RA has no query-field loss.
        zero = output.new_tensor(0.0)
        return output, semantic_output, zero
