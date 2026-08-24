"""Train a retrieval-free Sign-JEPA latent generator for native SLRTP-178.

Stage 1 (`train_sign_jepa_slrtp178.py`) learns a predictive sign-motion latent.
This stage freezes that JEPA target encoder and trains:

  gloss + duration + progress -> latent sequence -> SLRTP-178 pose sequence

The latent loss is primary; pose and motion-descriptor losses are secondary.
This keeps the generator anchored to predictive motion structure instead of
plain frame reconstruction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_slrtp178 import (  # noqa: E402
    LENGTH_BUCKETS,
    SLRTPSpanDataset,
    SignJEPAModel,
    build_gloss_vocab,
    compute_stats,
    flat_pose,
    length_bucket_id,
    load_bank,
    resize_seq,
    sample_future_mask,
    sequence_features,
    sinusoidal_positions,
)


class LatentProgramGenerator(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_len: int,
        style_dim: int = 0,
        energy_dim: int = 0,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        self.style_dim = style_dim
        self.energy_dim = energy_dim
        self.gloss_emb = nn.Embedding(num_gloss, hidden)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.style_proj = nn.Linear(style_dim, hidden) if style_dim > 0 else None
        self.energy_proj = nn.Linear(energy_dim, hidden) if energy_dim > 0 else None
        if self.energy_proj is not None:
            nn.init.zeros_(self.energy_proj.weight)
            nn.init.zeros_(self.energy_proj.bias)
        self.progress_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
        T: int,
        style: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = gloss_id.shape[0]
        device = gloss_id.device
        pos = sinusoidal_positions(T, self.hidden, device)[None].expand(B, -1, -1)
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack(
            [p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1
        )
        h = pos + self.progress_mlp(prog)[None]
        h = h + self.gloss_emb(gloss_id)[:, None] + self.len_emb(len_id)[:, None]
        if self.style_proj is not None:
            if style is None:
                style = torch.zeros(B, self.style_dim, device=device)
            h = h + self.style_proj(style)[:, None]
        if self.energy_proj is not None:
            if energy is None:
                energy = torch.zeros(B, self.energy_dim, device=device)
            h = h + self.energy_proj(energy)[:, None]
        return self.norm(self.blocks(h))


class EnergyPredictor(nn.Module):
    def __init__(self, num_gloss: int, hidden: int, energy_dim: int):
        super().__init__()
        self.gloss_emb = nn.Embedding(num_gloss, hidden)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, energy_dim),
        )

    def forward(self, gloss_id: torch.Tensor, len_id: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.gloss_emb(gloss_id), self.len_emb(len_id)], dim=-1)
        return self.net(h)


class SentenceBudgetAllocator(nn.Module):
    def __init__(self, num_gloss: int, hidden: int, energy_dim: int, max_sent_len: int):
        super().__init__()
        self.hidden = hidden
        self.max_sent_len = max_sent_len
        self.gloss_emb = nn.Embedding(num_gloss, hidden)
        self.pos_emb = nn.Embedding(max_sent_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, energy_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        sent_gloss_ids: torch.Tensor,
        sent_mask: torch.Tensor,
        sent_pos: torch.Tensor,
    ) -> torch.Tensor:
        B, S = sent_gloss_ids.shape
        pos = torch.arange(S, device=sent_gloss_ids.device).clamp_max(self.max_sent_len - 1)
        h = self.gloss_emb(sent_gloss_ids) + self.pos_emb(pos)[None]
        h = self.blocks(h, src_key_padding_mask=~sent_mask.bool())
        gather = sent_pos.clamp(0, S - 1).reshape(B, 1, 1).expand(B, 1, self.hidden)
        return self.out(h.gather(1, gather).squeeze(1))


class JEPAPoseDecoder(nn.Module):
    def __init__(
        self,
        hidden: int,
        pose_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, pose_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(z))


class SignJEPAGenerator(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        hidden: int,
        gen_layers: int,
        dec_layers: int,
        heads: int,
        dropout: float,
        max_len: int,
        pose_dim: int = 534,
        style_dim: int = 0,
        use_energy_cond: bool = False,
        energy_dim: int = 16,
        use_sentence_budget: bool = False,
        sentence_max_len: int = 32,
        stochastic_decoder: bool = False,
        init_log_var: float = -4.0,
        latent_noise_std: float = 0.0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.use_energy_cond = use_energy_cond
        self.energy_dim = energy_dim if use_energy_cond else 0
        self.use_sentence_budget = use_sentence_budget
        self.stochastic_decoder = stochastic_decoder
        # v26. Per-clip latent noise vector broadcast over the time axis.
        # Same noise applied at every timestep -> temporally coherent pose
        # variation through the decoder, fixing v25's iid-per-frame problem.
        self.latent_noise_std = float(latent_noise_std)
        self.generator = LatentProgramGenerator(
            num_gloss=num_gloss,
            hidden=hidden,
            layers=gen_layers,
            heads=heads,
            dropout=dropout,
            max_len=max_len,
            style_dim=style_dim,
            energy_dim=self.energy_dim,
        )
        self.decoder = JEPAPoseDecoder(
            hidden=hidden,
            pose_dim=pose_dim,
            layers=dec_layers,
            heads=heads,
            dropout=dropout,
        )
        self.energy_predictor = (
            EnergyPredictor(num_gloss, hidden, self.energy_dim)
            if self.use_energy_cond else None
        )
        self.sentence_budget_allocator = (
            SentenceBudgetAllocator(num_gloss, hidden, self.energy_dim, sentence_max_len)
            if self.use_energy_cond and self.use_sentence_budget else None
        )
        # v25 stochastic decoder. A separate Linear from the generator's latent
        # z to per-element log-variance. Zero-init weights + constant bias
        # gives a tiny isotropic std at init so warm-started v24a poses come
        # out essentially unchanged at temperature=1, then the NLL gradient
        # widens the std exactly where the data demands it.
        if self.stochastic_decoder:
            self.log_var_head = nn.Linear(hidden, pose_dim)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, float(init_log_var))
        else:
            self.log_var_head = None

    def predict_energy(self, gloss_id: torch.Tensor, len_id: torch.Tensor) -> torch.Tensor | None:
        if self.energy_predictor is None:
            return None
        return self.energy_predictor(gloss_id, len_id)

    def predict_condition_energy(
        self,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
        sent_gloss_ids: torch.Tensor | None = None,
        sent_mask: torch.Tensor | None = None,
        sent_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        base = self.predict_energy(gloss_id, len_id)
        if base is None:
            return None
        if self.sentence_budget_allocator is not None and sent_gloss_ids is not None:
            base = base + self.sentence_budget_allocator(sent_gloss_ids, sent_mask, sent_pos)
        return base

    def forward(
        self,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
        T: int,
        style: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.generator(gloss_id, len_id, T, style=style, energy=energy)
        z_decoder_input = z
        if self.latent_noise_std > 0:
            # v26. Temporally coherent latent noise: sample at low frequency
            # (T_low = max(2, T // 8) points), then linearly interpolate up to
            # T frames. Per-clip-broadcast noise (T_low=1) would shift pose
            # statically and preserve velocity; iid per-frame (v25 sample
            # head) explodes jerk. T/8 gives a smooth time-varying
            # perturbation whose decoder response can add coherent motion.
            B = z.shape[0]
            T_dim = z.shape[1]
            T_low = max(2, T_dim // 8)
            noise_low = torch.randn(B, T_low, z.shape[-1], device=z.device, dtype=z.dtype)
            noise = F.interpolate(
                noise_low.transpose(1, 2), size=T_dim,
                mode="linear", align_corners=True,
            ).transpose(1, 2)
            z_decoder_input = z + noise * self.latent_noise_std
        pose = self.decoder(z_decoder_input)
        log_var = self.log_var_head(z_decoder_input) if self.log_var_head is not None else None
        # Return the CLEAN z so latent_loss measures the semantic latent, not
        # the noisy decoder input.
        return z, pose, log_var


def load_frozen_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = SignJEPAModel(
        feat_dim=ckpt["feat_dim"],
        desc_dim=ckpt["desc_dim"],
        num_gloss=len(ckpt["gloss_to_id"]),
        hidden=a["hidden"],
        layers=a["layers"],
        heads=a["heads"],
        pred_layers=a["pred_layers"],
        dropout=0.0,
        max_len=a["seg_len"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def mean_lengths_by_gloss(spans: list[tuple[str, str, int, int]]) -> dict[str, float]:
    vals: dict[str, list[int]] = {}
    for gloss, _sid, s, e in spans:
        vals.setdefault(str(gloss), []).append(max(1, int(e) - int(s)))
    return {g: float(np.mean(v)) for g, v in vals.items()}


class MotionDiscriminator(nn.Module):
    """v29 PatchGAN-style 1D-Conv discriminator on hand+wrist motion.

    Operates in raw pose units on the hand region (joints 8:50) plus the two
    wrist joints (2, 5) — 44 joints x 3 = 132 channels per frame. The conv
    stack downsamples the time axis by 8x and outputs a per-window realness
    logit. Spectral-norm regularization on every conv keeps the
    Lipschitz constant bounded, making hinge-loss training stable without
    R1 weight tuning.

    The carrier is generator G; D is trained against G's raw pose output and
    Phoenix GT pose. Adversarial gradient pushes G's per-frame motion
    distribution toward GT, lifting magnitude and aligning direction
    simultaneously — the supervisory regime no L1/NLL/cosine-only loss can
    deliver.
    """

    HAND_WRIST_IDX = [2, 5] + list(range(8, 50))  # 44 joints

    def __init__(self, hidden: int = 64, dropout: float = 0.1, joint_idx=None):
        super().__init__()
        # Ablation hook: joint_idx=None -> hand+wrist (default); pass a full-body
        # index set for the full-body-critic ablation.
        self.joint_idx = list(self.HAND_WRIST_IDX if joint_idx is None else joint_idx)
        sn = nn.utils.spectral_norm
        in_c = len(self.joint_idx) * 3
        self.blocks = nn.Sequential(
            sn(nn.Conv1d(in_c, hidden, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            sn(nn.Conv1d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            sn(nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = sn(nn.Conv1d(hidden * 4, 1, kernel_size=1))

    def forward(self, pose_raw: torch.Tensor) -> torch.Tensor:
        """Input [B, T, 178, 3] raw pose. Returns [B, n_windows] logits."""
        B, T = pose_raw.shape[:2]
        idx = torch.tensor(self.joint_idx, dtype=torch.long, device=pose_raw.device)
        x = pose_raw.index_select(2, idx).reshape(B, T, -1)
        x = x.transpose(1, 2)                                 # [B, 132, T]
        h = self.blocks(x)
        logits = self.head(h).squeeze(1)                     # [B, n_windows]
        return logits


@torch.no_grad()
def encode_target_latents(jepa: SignJEPAModel, pose_norm, pose_raw, gloss_id, len_id) -> tuple[torch.Tensor, torch.Tensor]:
    feat, desc = sequence_features(pose_norm, pose_raw)
    zero_mask = torch.zeros(pose_norm.shape[:2], dtype=torch.bool, device=pose_norm.device)
    z = jepa.target_encoder(feat, zero_mask, gloss_id, len_id)
    return z.detach(), desc.detach()


def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def hand_jerk_loss(pred_raw: torch.Tensor) -> torch.Tensor:
    """Train-time temporal-smoothness regularizer (R1.7/R2.1): mean L2 magnitude of
    the third temporal difference over the hand joints (8:50) in raw pose units,
    matching the reported hand-jerk metric. Penalizes the high-frequency jitter the
    adversarial critic introduces, without touching body/face."""
    B, T = pred_raw.shape[0], pred_raw.shape[1]
    if T < 4:
        return pred_raw.new_zeros(())
    x = pred_raw.reshape(B, T, 178, 3)[:, :, 8:50]   # [B, T, 42, 3]
    v = x[:, 1:] - x[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]
    return j.norm(dim=-1).mean()


def latent_covariance_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Match the per-dimension covariance (Gram) structure of the generated
    latent to the carrier latent. Effective rank is a function of the latent
    covariance spectrum, so this term directly forces the generator to reproduce
    the carrier's *rank* / richness rather than collapsing z to a low-dimensional
    scaffold (which the magnitude-blind cosine anchor permits). Centered,
    normalized by the number of tokens, compared in Frobenius norm.
    """
    H = z_pred.shape[-1]
    zp = z_pred.reshape(-1, H)
    zt = z_tgt.reshape(-1, H)
    zp = zp - zp.mean(0, keepdim=True)
    zt = zt - zt.mean(0, keepdim=True)
    cov_p = (zp.transpose(0, 1) @ zp) / max(1, zp.shape[0] - 1)
    cov_t = (zt.transpose(0, 1) @ zt) / max(1, zt.shape[0] - 1)
    return (cov_p - cov_t).pow(2).mean()


def latent_anchor_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor,
                       mode: str = "cosine", raw_weight: float = 1.0,
                       cov_weight: float = 0.0) -> torch.Tensor:
    """Anchor the generated latent program to the frozen carrier latent.

    The carrier latent z is the decoder's input (pose = decoder(z)), so what we
    constrain here directly shapes the output.

      cosine (default): smooth_l1 on L2-normalized z -> matches DIRECTION only.
                        This is magnitude-blind and discards the radial/per-dim
                        variance that distinguishes a rich carrier (high
                        effective rank) from an impoverished one. With this loss
                        the generator only needs a coarse directional scaffold,
                        so any non-collapsed carrier suffices and representation
                        richness is not exploited.
      raw   : smooth_l1 on the UNnormalized z -> matches direction AND magnitude,
              forcing the generator to reproduce the carrier's full latent
              structure (the part that carries effective-rank richness).
      both  : cosine + raw_weight * raw -> keeps the well-behaved directional
              signal and adds the magnitude/richness term on top.

    cov_weight>0 adds a latent-covariance (rank/spectrum) matching term on top
    of any mode, the most direct way to make effective-rank richness
    load-bearing.
    """
    if mode == "cosine":
        base = F.smooth_l1_loss(F.normalize(z_pred, dim=-1), F.normalize(z_tgt, dim=-1))
    elif mode == "raw":
        base = F.smooth_l1_loss(z_pred, z_tgt)
    elif mode == "both":
        cos = F.smooth_l1_loss(F.normalize(z_pred, dim=-1), F.normalize(z_tgt, dim=-1))
        raw = F.smooth_l1_loss(z_pred, z_tgt)
        base = cos + raw_weight * raw
    else:
        raise ValueError(f"unknown latent_loss_mode={mode}")
    if cov_weight > 0:
        base = base + cov_weight * latent_covariance_loss(z_pred, z_tgt)
    return base


def jepa_predictive_consistency_loss(
    jepa: SignJEPAModel,
    pose_pred: torch.Tensor,
    pose_pred_raw: torch.Tensor,
    z_tgt: torch.Tensor,
    gloss_id: torch.Tensor,
    len_id: torch.Tensor,
    args,
) -> torch.Tensor:
    """Make the carrier's predictive objective load-bearing downstream.

    The ordinary latent anchor only asks the generated latent z to be close to
    the frozen target encoder. A reconstruction carrier can satisfy that well
    enough because the generator also sees strong pose/velocity/descriptor
    losses. This term adds the missing JEPA contract: when future generated
    frames are masked, the frozen carrier predictor must recover the GT target
    latent for those future frames from the visible generated context.

    For a true JEPA carrier this is the pretraining task. For AE/MAE carriers,
    the same architecture exists but was optimized for reconstruction, so the
    signal is expected to be much less aligned with SLP readability.
    """
    feat_pred, _desc_pred = sequence_features(pose_pred, pose_pred_raw)
    future_mask = sample_future_mask(
        pose_pred.shape[0],
        pose_pred.shape[1],
        args.jepa_pc_min_context,
        args.jepa_pc_target_len_min,
        args.jepa_pc_target_len_max,
        pose_pred.device,
    )
    pc_z, _self_z, _pc_desc = jepa(feat_pred, gloss_id, len_id, future_mask)
    if not future_mask.any():
        return pose_pred.new_zeros(())
    pc_sel = pc_z[future_mask]
    tgt_sel = z_tgt[future_mask]
    return latent_anchor_loss(
        pc_sel,
        tgt_sel,
        args.jepa_pc_loss_mode,
        args.jepa_pc_raw_weight,
        args.jepa_pc_cov_weight,
    )


def vel_direction_loss(pred: torch.Tensor, target: torch.Tensor,
                       wrist_weight: float = 0.5) -> torch.Tensor:
    """Per-frame, per-joint cosine velocity-direction loss, weighted by GT
    velocity magnitude. Magnitude-invariant — does not compete with
    pose_loss / velocity_loss directly; injects only directional information.

    v23 used wrist_weight=0.5 (hands + 0.5 * wrists). That dragged the body
    kinematic chain via the wrist anchors. v24a sets wrist_weight=0.0 to
    isolate intra-hand articulation supervision.

    Inputs are [B, T, 534] normalized poses (the carrier's native space).
    """
    B = pred.shape[0]
    p = pred.reshape(B, pred.shape[1], 178, 3)
    t = target.reshape(B, target.shape[1], 178, 3)
    eps = 1e-6

    def _dir(p_seg: torch.Tensor, t_seg: torch.Tensor) -> torch.Tensor:
        vp = p_seg[:, 1:] - p_seg[:, :-1]            # [B, T-1, J, 3]
        vt = t_seg[:, 1:] - t_seg[:, :-1]
        n_p = vp.norm(dim=-1).clamp_min(eps)
        n_t = vt.norm(dim=-1).clamp_min(eps)
        cos = (vp * vt).sum(dim=-1) / (n_p * n_t)     # [B, T-1, J]
        weight = n_t                                   # weight by GT motion
        num = ((1.0 - cos) * weight).sum()
        den = weight.sum().clamp_min(eps)
        return num / den

    L_hand = _dir(p[:, :, 8:50], t[:, :, 8:50])
    if wrist_weight > 0:
        wrist_idx = torch.tensor([2, 5], dtype=torch.long, device=pred.device)
        L_wrist = _dir(p.index_select(2, wrist_idx), t.index_select(2, wrist_idx))
        return L_hand + wrist_weight * L_wrist
    return L_hand


def std_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2)
    return F.smooth_l1_loss(pred.std(dim=dims), target.std(dim=dims))


def speed_envelope_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_v = pred[:, 1:].reshape(pred.shape[0], pred.shape[1] - 1, 178, 3)
    tgt_v = target[:, 1:].reshape(target.shape[0], target.shape[1] - 1, 178, 3)
    vals = []
    for lo, hi in ((8, 29), (29, 50)):
        ps = pred_v[:, :, lo:hi].norm(dim=-1).mean(dim=-1)
        ts = tgt_v[:, :, lo:hi].norm(dim=-1).mean(dim=-1)
        vals.append(F.smooth_l1_loss(ps, ts))
        vals.append(F.smooth_l1_loss(ps.max(dim=1).values, ts.max(dim=1).values))
    return torch.stack(vals).mean()


def motion_collapse_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """Symmetric motion-density penalty + jerk cap in raw pose units.

    The auxiliary loss the JEPA generator was missing.  An earlier
    asymmetric version (only penalising under-motion) caused the model to
    runaway and over-shoot — jerk reached 5.6, hand_speed 1.2x GT.  This
    version is symmetric on speed (smooth-L1 of `pred_speed / gt_speed`
    against 1.0) so over-motion is penalised, and adds a one-sided jerk
    cap so the model cannot satisfy the speed target by injecting jitter.

    Targets the SLRTP-measured quantities directly in their natural ratio
    form, so the gradient pushes the model toward `ratio = 1`:

      hand_speed_ratio    (gate >= 0.85, ideal 1.0)        joints 8:50
      body_speed_ratio    (total_distance proxy, ideal 1.0) joints 0:8
      hand_jerk_ratio     (gate <= 1.6, ideal <= 1.0)       joints 8:50
      wrist_travel ratio  (= total_distance, ideal 1.0)     joints 2, 5
    """
    B = pred_raw.shape[0]
    pred = pred_raw.reshape(B, pred_raw.shape[1], 178, 3)
    target = target_raw.reshape(B, target_raw.shape[1], 178, 3)

    pv = pred[:, 1:] - pred[:, :-1]                       # [B, T-1, 178, 3]
    tv = target[:, 1:] - target[:, :-1]
    pa = pv[:, 1:] - pv[:, :-1]                           # [B, T-2, 178, 3] (jerk proxy)
    ta = tv[:, 1:] - tv[:, :-1]

    eps = 1e-6
    one = torch.ones((B,), device=pred_raw.device)
    parts = []

    # Symmetric speed match per region.
    for sl, weight in (
        (slice(8, 50),  2.0),    # hands (motion gate)
        (slice(0, 8),   1.0),    # body (total_distance proxy)
        (slice(50, 178), 0.25),  # face (mild signal)
    ):
        sp = pv[:, :, sl].norm(dim=-1).mean(dim=(1, 2))
        st = tv[:, :, sl].norm(dim=-1).mean(dim=(1, 2))
        ratio = sp / (st + eps)
        parts.append(weight * F.smooth_l1_loss(ratio, one))

    # One-sided jerk cap: penalty grows when pred jerk exceeds GT jerk.
    pj = pa[:, :, 8:50].norm(dim=-1).mean(dim=(1, 2))
    tj = ta[:, :, 8:50].norm(dim=-1).mean(dim=(1, 2))
    jratio = pj / (tj + eps)
    parts.append(0.5 * F.relu(jratio - 1.0).mean())

    # Wrist-travel ratio (SLRTP total_distance), eval-rate; symmetric.
    for w_idx in (2, 5):
        pw = pred[:, ::2, w_idx]
        tw = target[:, ::2, w_idx]
        pwd = (pw[:, 1:] - pw[:, :-1]).norm(dim=-1).sum(dim=1)
        twd = (tw[:, 1:] - tw[:, :-1]).norm(dim=-1).sum(dim=1)
        parts.append(0.5 * F.smooth_l1_loss(pwd / (twd + eps), one))

    return torch.stack(parts).sum()


def hand_arc_budget_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """Duration-robust hand arc-length matching with a jerk budget.

    The full-utterance no-leak sampler resizes fixed-length generated gloss
    spans to text-predicted durations, so framewise speed can collapse even
    when the fixed-segment velocity loss looks healthy. This loss matches the
    hand trajectory arc length and burst budget, which survive linear time
    resizing better than per-frame velocity, while keeping acceleration/jerk
    from becoming the v12-style shortcut.
    """
    B = pred_raw.shape[0]
    pred = pred_raw.reshape(B, pred_raw.shape[1], 178, 3)
    target = target_raw.reshape(B, target_raw.shape[1], 178, 3)

    pv = pred[:, 1:, 8:50] - pred[:, :-1, 8:50]
    tv = target[:, 1:, 8:50] - target[:, :-1, 8:50]
    pa = pv[:, 1:] - pv[:, :-1]
    ta = tv[:, 1:] - tv[:, :-1]

    eps = 1e-6
    zero = pred_raw.new_zeros((B,))

    pmag = pv.norm(dim=-1)  # [B, T-1, hands]
    tmag = tv.norm(dim=-1)
    parclen = pmag.sum(dim=1).mean(dim=1)
    tarclen = tmag.sum(dim=1).mean(dim=1)
    arc_loss = F.smooth_l1_loss(torch.log((parclen + eps) / (tarclen + eps)), zero)

    # Burst budget: top 20% hand-motion magnitudes, flattened across hands.
    flat_p = pmag.reshape(B, -1)
    flat_t = tmag.reshape(B, -1)
    k = max(1, int(math.ceil(flat_p.shape[1] * 0.20)))
    pburst = flat_p.topk(k, dim=1).values.mean(dim=1)
    tburst = flat_t.topk(k, dim=1).values.mean(dim=1)
    burst_loss = F.smooth_l1_loss(torch.log((pburst + eps) / (tburst + eps)), zero)

    pjerk = pa.norm(dim=-1).mean(dim=(1, 2))
    tjerk = ta.norm(dim=-1).mean(dim=(1, 2))
    jerk_ratio = pjerk / (tjerk + eps)
    jerk_cap = F.relu(jerk_ratio - 1.05).mean()

    return arc_loss + 0.5 * burst_loss + 0.5 * jerk_cap


def spectral_wrist_hand_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    kernel: int = 7,
    hand_lo: float = 0.90,
    hand_hi: float = 1.10,
    wrist_lo: float = 0.90,
    wrist_hi: float = 1.15,
    jerk_cap: float = 1.20,
) -> torch.Tensor:
    """Low-frequency motion-energy objective with local-shape guardrails.

    v15 showed that low-pass hand amplification can pass the motion gate with
    tolerable, but non-SOTA, recognition. v18 showed that pure wrist boosts
    preserve local hand pose-std but hurt PHIX when bolted on at sample time.
    This loss moves that pressure into training: match low-frequency hand and
    wrist velocities while capping high-frequency/local-hand distortion.
    """
    B = pred_raw.shape[0]
    pred = pred_raw.reshape(B, pred_raw.shape[1], 178, 3)
    target = target_raw.reshape(B, target_raw.shape[1], 178, 3)
    eps = 1e-6

    pred_lp = _wav_smooth_pose_time(pred_raw, kernel).reshape_as(pred)
    tgt_lp = _wav_smooth_pose_time(target_raw, kernel).reshape_as(target)
    pv = pred_lp[:, 1:] - pred_lp[:, :-1]
    tv = tgt_lp[:, 1:] - tgt_lp[:, :-1]

    def band_ratio(x: torch.Tensor, y: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        ratio = x / (y + eps)
        return F.relu(float(lo) - ratio).mean() + F.relu(ratio - float(hi)).mean()

    ph = pv[:, :, 8:50].norm(dim=-1).mean(dim=(1, 2))
    th = tv[:, :, 8:50].norm(dim=-1).mean(dim=(1, 2))
    hand_band = band_ratio(ph, th, hand_lo, hand_hi)

    pw = pv[:, :, [2, 5]].norm(dim=-1).mean(dim=(1, 2))
    tw = tv[:, :, [2, 5]].norm(dim=-1).mean(dim=(1, 2))
    wrist_band = band_ratio(pw, tw, wrist_lo, wrist_hi)

    # Preserve local finger articulation in wrist frames so the carrier cannot
    # satisfy speed by distorting hand shape.
    prh = pred[:, :, 8:29] - pred[:, :, 2:3]
    trh = target[:, :, 8:29] - target[:, :, 2:3]
    plh = pred[:, :, 29:50] - pred[:, :, 5:6]
    tlh = target[:, :, 29:50] - target[:, :, 5:6]
    local_shape = 0.5 * (F.smooth_l1_loss(prh, trh) + F.smooth_l1_loss(plh, tlh))

    # One-sided high-frequency jerk cap on local hands.
    plocal = torch.cat([prh, plh], dim=2)
    tlocal = torch.cat([trh, tlh], dim=2)
    pvel = plocal[:, 1:] - plocal[:, :-1]
    tvel = tlocal[:, 1:] - tlocal[:, :-1]
    pjerk = (pvel[:, 1:] - pvel[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
    tjerk = (tvel[:, 1:] - tvel[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
    jerk = F.relu((pjerk / (tjerk + eps)) - float(jerk_cap)).mean()

    return hand_band + 0.5 * wrist_band + 0.5 * local_shape + 0.25 * jerk


def energy_budget_descriptor(pose_raw: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
    """Text-predictable per-gloss motion budget descriptor.

    Returns 16 dims: for left hand and right hand, speed stats
    (mean/max/p95/top-5-frame integral) followed by the same acceleration
    stats. The descriptor is log-scaled so the MLP predicts relative budget
    rather than tiny raw-coordinate deltas.
    """
    pose = pose_raw.reshape(pose_raw.shape[0], pose_raw.shape[1], 178, 3)
    parts = []
    for sl in (slice(8, 29), slice(29, 50)):
        hand = pose[:, :, sl]
        vel = hand[:, 1:] - hand[:, :-1]
        speed = vel.norm(dim=-1).mean(dim=-1)
        if speed.shape[1] == 0:
            speed = pose_raw.new_zeros((pose_raw.shape[0], 1))
        acc = vel[:, 1:] - vel[:, :-1]
        accel = acc.norm(dim=-1).mean(dim=-1)
        if accel.shape[1] == 0:
            accel = pose_raw.new_zeros((pose_raw.shape[0], 1))
        for x in (speed, accel):
            k = min(5, x.shape[1])
            stat = torch.stack(
                [
                    x.mean(dim=1),
                    x.max(dim=1).values,
                    torch.quantile(x, 0.95, dim=1),
                    x.topk(k, dim=1).values.sum(dim=1),
                ],
                dim=-1,
            )
            parts.append(stat)
    desc = torch.cat(parts, dim=-1)
    if scale > 0:
        desc = torch.log1p(desc / scale)
    return desc.detach()


def _wav_smooth_pose_time(pose_raw: torch.Tensor, kernel: int) -> torch.Tensor:
    """Replicate-padded temporal avg-pool on a (B, T, 534) pose tensor.

    kernel <= 1 is identity. For kernel > 1 this gives a low-pass-filtered
    pose at the requested temporal scale; differences give scale-specific
    hand-velocity magnitudes (the "wavelet" hierarchy used by WAV-KL).
    """
    if kernel <= 1:
        return pose_raw
    y = pose_raw.transpose(1, 2)  # (B, 534, T)
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    return y.transpose(1, 2)


def _wav_hand_vel_magnitudes(pose_raw: torch.Tensor, kernel: int) -> torch.Tensor:
    """Flatten all (B, T-1, 42) hand joint velocity magnitudes at a scale."""
    smooth = _wav_smooth_pose_time(pose_raw, kernel)
    smooth = smooth.reshape(smooth.shape[0], smooth.shape[1], 178, 3)
    hand = smooth[:, :, 8:50]                       # (B, T, 42, 3)
    vel = hand[:, 1:] - hand[:, :-1]                # (B, T-1, 42, 3)
    return vel.norm(dim=-1).reshape(-1)             # (B*(T-1)*42,)


def _soft_histogram(x: torch.Tensor, bin_centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """Differentiable Gaussian-kernel soft histogram.

    x:          (N,) sample magnitudes.
    bin_centers:(K,) bin centers in the same units as x.
    sigma:      kernel width.
    Returns:    (K,) histogram, sums to 1.
    """
    d = (x.unsqueeze(1) - bin_centers.unsqueeze(0)) / sigma
    w = torch.exp(-0.5 * d * d)
    row_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w = w / row_sum                                  # each sample distributes mass 1
    return w.sum(dim=0) / w.shape[0]                 # average over samples


def wavkl_hand_loss(pred_pose_raw: torch.Tensor,
                    gt_hists: dict,
                    bin_centers: torch.Tensor,
                    sigma: float,
                    scales: tuple,
                    eps: float = 1e-8,
                    reverse: bool = False) -> torch.Tensor:
    """Multi-scale forward-KL of hand-velocity magnitude distribution.

    For each temporal smoothing scale k, compute a soft histogram of all
    hand-joint velocity magnitudes from the predicted pose and KL-match it
    against a precomputed GT histogram at the same scale. Identity (no
    motion) puts all mass at the zero-velocity bin where GT has near-zero
    mass — the KL has unbounded penalty for that collapse.
    """
    parts = []
    for k in scales:
        mag = _wav_hand_vel_magnitudes(pred_pose_raw, k)
        pred_h = _soft_histogram(mag, bin_centers, sigma)
        gt_h = gt_hists[k]
        if reverse:  # ablation: reverse (mode-seeking) KL(gt || pred)
            kl = (gt_h * ((gt_h + eps).log() - (pred_h + eps).log())).sum()
        else:        # default: forward (mass-covering) KL(pred || gt)
            kl = (pred_h * ((pred_h + eps).log() - (gt_h + eps).log())).sum()
        parts.append(kl)
    return torch.stack(parts).mean()


@torch.no_grad()
def precompute_gt_wavkl(loader, mean, std, device, scales: tuple,
                        n_bins: int, vmax: float, sigma: float,
                        max_batches: int = 0) -> tuple[dict, torch.Tensor]:
    """One-time pass over the train loader to estimate the GT hand-velocity
    magnitude histogram at each temporal scale. Stop-grad bin centers and
    histograms are returned.
    """
    bin_centers = torch.linspace(0.0, vmax, n_bins, device=device)
    accum = {k: torch.zeros(n_bins, device=device) for k in scales}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        pose_raw = batch["pose_raw"].to(device)
        for k in scales:
            mag = _wav_hand_vel_magnitudes(pose_raw, k)
            accum[k] = accum[k] + _soft_histogram(mag, bin_centers, sigma)
        n += 1
    gt_hists = {k: (accum[k] / max(n, 1)).detach() for k in scales}
    return gt_hists, bin_centers.detach()


def train_epoch(model, jepa, loader, opt, mean, std, device, args,
                wavkl_state=None,
                discriminator=None, d_opt=None,
                anchor_model=None) -> dict:
    model.train()
    vals = []
    lat_vals = []
    pose_vals = []
    vel_vals = []
    desc_vals = []
    std_vals = []
    speed_vals = []
    wavkl_vals = []
    energy_vals = []
    arc_vals = []
    spectral_vals = []
    vel_dir_vals = []
    jepa_pc_vals = []
    adv_g_vals: list[float] = []
    adv_d_vals: list[float] = []
    d_real_vals: list[float] = []
    d_fake_vals: list[float] = []
    anchor_vals: list[float] = []
    for step, batch in enumerate(loader, start=1):
        pose = batch["pose"].to(device)
        pose_raw = batch["pose_raw"].to(device)
        gloss_id = batch["gloss_id"].to(device)
        len_id = batch["len_id"].to(device)
        z_tgt, desc_tgt = encode_target_latents(jepa, pose, pose_raw, gloss_id, len_id)
        style = None
        if args.style_dim > 0:
            style = torch.randn(pose.shape[0], args.style_dim, device=device) * args.style_train_std
        energy_gt = None
        energy_cond = None
        energy_pred_loss = pose.new_zeros(())
        if args.use_energy_cond:
            energy_gt = energy_budget_descriptor(pose_raw, scale=args.energy_desc_scale)
            if args.use_sentence_budget:
                energy_pred = model.predict_condition_energy(
                    gloss_id,
                    len_id,
                    batch["sent_gloss_ids"].to(device),
                    batch["sent_mask"].to(device),
                    batch["sent_pos"].to(device),
                )
            else:
                energy_pred = model.predict_energy(gloss_id, len_id)
            energy_pred_loss = F.smooth_l1_loss(energy_pred, energy_gt)
        energy_cond = energy_pred if args.use_sentence_budget else energy_gt
        z_pred, pose_pred, pose_log_var = model(gloss_id, len_id, pose.shape[1], style=style, energy=energy_cond)

        latent = latent_anchor_loss(z_pred, z_tgt, args.latent_loss_mode, args.latent_raw_weight, args.latent_cov_weight)
        if pose_log_var is not None and args.lambda_nll > 0:
            # v25 heteroscedastic Gaussian NLL replaces the deterministic L1.
            # log_var is clamped to keep training stable.
            lv = pose_log_var.clamp(min=-8.0, max=2.0)
            inv_var = (-lv).exp()
            diff2 = (pose_pred - pose).pow(2)
            nll = 0.5 * (diff2 * inv_var + lv)            # const term dropped
            pose_loss = nll.mean()
        else:
            pose_loss = F.smooth_l1_loss(pose_pred, pose)
        vel = velocity_loss(pose_pred, pose)
        pose_pred_raw = pose_pred * std.to(device) + mean.to(device)
        _feat_pred, desc_pred = sequence_features(pose_pred, pose_pred_raw)
        desc = F.smooth_l1_loss(desc_pred, desc_tgt)
        std_loss = std_match_loss(pose_pred, pose)
        speed_loss = speed_envelope_loss(pose_pred, pose)
        mcoll = motion_collapse_loss(pose_pred_raw, pose_raw) if args.lambda_motion_collapse > 0 else pose_pred_raw.new_zeros(())
        arc_budget = hand_arc_budget_loss(pose_pred_raw, pose_raw) if args.lambda_arc_budget > 0 else pose_pred_raw.new_zeros(())
        spectral = spectral_wrist_hand_loss(
            pose_pred_raw,
            pose_raw,
            kernel=args.spectral_kernel,
            hand_lo=args.spectral_hand_lo,
            hand_hi=args.spectral_hand_hi,
            wrist_lo=args.spectral_wrist_lo,
            wrist_hi=args.spectral_wrist_hi,
            jerk_cap=args.spectral_jerk_cap,
        ) if args.lambda_spectral_wrist > 0 else pose_pred_raw.new_zeros(())
        if wavkl_state is not None and args.lambda_wavkl > 0:
            wavkl = wavkl_hand_loss(
                pose_pred_raw,
                wavkl_state["gt_hists"],
                wavkl_state["bin_centers"],
                wavkl_state["sigma"],
                wavkl_state["scales"],
                reverse=wavkl_state.get("reverse", False),
            )
        else:
            wavkl = pose_pred_raw.new_zeros(())
        if args.lambda_vel_dir > 0:
            vel_dir = vel_direction_loss(pose_pred, pose,
                                         wrist_weight=args.vel_dir_wrist_weight)
        else:
            vel_dir = pose_pred.new_zeros(())
        if args.lambda_jepa_pc > 0:
            jepa_pc = jepa_predictive_consistency_loss(
                jepa,
                pose_pred,
                pose_pred_raw,
                z_tgt,
                gloss_id,
                len_id,
                args,
            )
        else:
            jepa_pc = pose_pred.new_zeros(())

        # v29 adversarial step. D operates on raw pose; train D first on
        # (real GT, fake G output detached), then compute G's adversarial
        # loss from the same fake batch with D parameters frozen.
        adv_g = pose_pred.new_zeros(())
        adv_d = pose_pred.new_zeros(())
        d_real_mean = 0.0
        d_fake_mean = 0.0
        # v33 anchor: body+face L1 distillation toward frozen anchor carrier.
        L_anchor = pose_pred.new_zeros(())
        if anchor_model is not None and args.lambda_anchor > 0:
            with torch.no_grad():
                # Same conditioning as the training step
                a_z, a_pose, _a_lv = anchor_model(
                    gloss_id, len_id, pose.shape[1],
                    style=None, energy=energy_cond,
                )
            # Body (0:8) + Face (50:178) regions, in normalized pose space
            B_, T_, F_ = pose_pred.shape
            pose_pred_r = pose_pred.reshape(B_, T_, 178, 3)
            a_pose_r = a_pose.reshape(B_, T_, 178, 3)
            body = pose_pred_r[:, :, 0:8]
            face = pose_pred_r[:, :, 50:178]
            a_body = a_pose_r[:, :, 0:8]
            a_face = a_pose_r[:, :, 50:178]
            L_anchor = (
                F.smooth_l1_loss(body, a_body) + F.smooth_l1_loss(face, a_face)
            )

        if discriminator is not None and args.lambda_adv > 0:
            B, Tseg = pose_pred.shape[:2]
            sd = std.to(device)
            mn = mean.to(device)
            pose_pred_raw_for_d = (pose_pred.detach() * sd + mn).reshape(B, Tseg, 178, 3)
            pose_real_for_d = pose_raw.reshape(B, Tseg, 178, 3)
            logits_real = discriminator(pose_real_for_d)
            logits_fake = discriminator(pose_pred_raw_for_d)
            d_loss = F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            adv_d = d_loss.detach()
            d_real_mean = float(logits_real.detach().mean().cpu())
            d_fake_mean = float(logits_fake.detach().mean().cpu())
            # G adversarial signal. Pose_pred is the live graph from G.
            pose_pred_raw_for_g = (pose_pred * sd + mn).reshape(B, Tseg, 178, 3)
            adv_g = -discriminator(pose_pred_raw_for_g).mean()

        loss = (
            args.lambda_latent * latent
            + args.lambda_pose * pose_loss
            + args.lambda_vel * vel
            + args.lambda_desc * desc
            + args.lambda_std * std_loss
            + args.lambda_speed * speed_loss
            + args.lambda_motion_collapse * mcoll
            + args.lambda_arc_budget * arc_budget
            + args.lambda_spectral_wrist * spectral
            + args.lambda_wavkl * wavkl
            + args.lambda_energy_pred * energy_pred_loss
            + args.lambda_vel_dir * vel_dir
            + args.lambda_jepa_pc * jepa_pc
            + args.lambda_adv * adv_g
            + args.lambda_anchor * L_anchor
            + args.lambda_jerk * (hand_jerk_loss(pose_pred_raw)
                                  if args.lambda_jerk > 0 else pose_pred_raw.new_zeros(()))
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        vals.append(float(loss.detach().cpu()))
        lat_vals.append(float(latent.detach().cpu()))
        pose_vals.append(float(pose_loss.detach().cpu()))
        vel_vals.append(float(vel.detach().cpu()))
        desc_vals.append(float(desc.detach().cpu()))
        std_vals.append(float(std_loss.detach().cpu()))
        speed_vals.append(float(speed_loss.detach().cpu()))
        wavkl_vals.append(float(wavkl.detach().cpu()))
        energy_vals.append(float(energy_pred_loss.detach().cpu()))
        arc_vals.append(float(arc_budget.detach().cpu()))
        spectral_vals.append(float(spectral.detach().cpu()))
        vel_dir_vals.append(float(vel_dir.detach().cpu()))
        jepa_pc_vals.append(float(jepa_pc.detach().cpu()))
        adv_g_vals.append(float(adv_g.detach().cpu()) if torch.is_tensor(adv_g) else float(adv_g))
        adv_d_vals.append(float(adv_d.detach().cpu()) if torch.is_tensor(adv_d) else float(adv_d))
        d_real_vals.append(d_real_mean)
        d_fake_vals.append(d_fake_mean)
        anchor_vals.append(float(L_anchor.detach().cpu()) if torch.is_tensor(L_anchor) else float(L_anchor))
        if args.log_every and (step == 1 or step % args.log_every == 0):
            print(
                json.dumps({
                    "step": step,
                    "loss": vals[-1],
                    "latent": lat_vals[-1],
                    "pose": pose_vals[-1],
                    "vel": vel_vals[-1],
                    "desc": desc_vals[-1],
                    "std": std_vals[-1],
                    "speed": speed_vals[-1],
                    "wavkl": wavkl_vals[-1],
                    "energy_pred": energy_vals[-1],
                    "arc_budget": arc_vals[-1],
                    "spectral_wrist": spectral_vals[-1],
                    "vel_dir": vel_dir_vals[-1],
                    "jepa_pc": jepa_pc_vals[-1],
                    "adv_g": adv_g_vals[-1],
                    "adv_d": adv_d_vals[-1],
                    "d_real": d_real_vals[-1],
                    "d_fake": d_fake_vals[-1],
                    "L_anchor": anchor_vals[-1],
                    "grad_norm": float(gnorm),
                    "pose_pred_std": float(pose_pred.detach().std().cpu()),
                    "pose_tgt_std": float(pose.detach().std().cpu()),
                }),
                flush=True,
            )
        if args.max_steps and step >= args.max_steps:
            break
    return {
        "train_loss": float(np.mean(vals)),
        "train_latent": float(np.mean(lat_vals)),
        "train_pose": float(np.mean(pose_vals)),
        "train_vel": float(np.mean(vel_vals)),
        "train_desc": float(np.mean(desc_vals)),
        "train_std": float(np.mean(std_vals)),
        "train_speed": float(np.mean(speed_vals)),
        "train_wavkl": float(np.mean(wavkl_vals)) if wavkl_vals else 0.0,
        "train_energy_pred": float(np.mean(energy_vals)) if energy_vals else 0.0,
        "train_arc_budget": float(np.mean(arc_vals)) if arc_vals else 0.0,
        "train_spectral_wrist": float(np.mean(spectral_vals)) if spectral_vals else 0.0,
        "train_vel_dir": float(np.mean(vel_dir_vals)) if vel_dir_vals else 0.0,
        "train_jepa_pc": float(np.mean(jepa_pc_vals)) if jepa_pc_vals else 0.0,
        "train_adv_g": float(np.mean(adv_g_vals)) if adv_g_vals else 0.0,
        "train_adv_d": float(np.mean(adv_d_vals)) if adv_d_vals else 0.0,
        "train_d_real": float(np.mean(d_real_vals)) if d_real_vals else 0.0,
        "train_d_fake": float(np.mean(d_fake_vals)) if d_fake_vals else 0.0,
        "train_L_anchor": float(np.mean(anchor_vals)) if anchor_vals else 0.0,
    }


@torch.no_grad()
def eval_epoch(model, jepa, loader, mean, std, device, args,
               wavkl_state=None) -> dict:
    model.eval()
    vals = []
    cos_vals = []
    pose_vals = []
    desc_vals = []
    wavkl_vals = []
    energy_vals = []
    arc_vals = []
    spectral_vals = []
    vel_dir_vals = []
    jepa_pc_vals = []
    hand_speed_ratios = []
    hand_jerk_ratios = []
    predcond_hand_speed_ratios = []
    predcond_hand_jerk_ratios = []
    for bi, batch in enumerate(loader):
        if args.val_batches and bi >= args.val_batches:
            break
        pose = batch["pose"].to(device)
        pose_raw = batch["pose_raw"].to(device)
        gloss_id = batch["gloss_id"].to(device)
        len_id = batch["len_id"].to(device)
        z_tgt, desc_tgt = encode_target_latents(jepa, pose, pose_raw, gloss_id, len_id)
        energy_gt = None
        energy_cond = None
        energy_pred_loss = pose.new_zeros(())
        if args.use_energy_cond:
            energy_gt = energy_budget_descriptor(pose_raw, scale=args.energy_desc_scale)
            if args.use_sentence_budget:
                energy_pred = model.predict_condition_energy(
                    gloss_id,
                    len_id,
                    batch["sent_gloss_ids"].to(device),
                    batch["sent_mask"].to(device),
                    batch["sent_pos"].to(device),
                )
            else:
                energy_pred = model.predict_energy(gloss_id, len_id)
            energy_pred_loss = F.smooth_l1_loss(energy_pred, energy_gt)
        energy_cond = energy_pred if args.use_sentence_budget else energy_gt
        z_pred, pose_pred, pose_log_var = model(gloss_id, len_id, pose.shape[1], energy=energy_cond)
        latent = latent_anchor_loss(z_pred, z_tgt, args.latent_loss_mode, args.latent_raw_weight, args.latent_cov_weight)
        pred_n = F.normalize(z_pred, dim=-1)   # retained for the cosine diagnostic below
        tgt_n = F.normalize(z_tgt, dim=-1)
        pose_loss = F.smooth_l1_loss(pose_pred, pose)
        pose_pred_raw = pose_pred * std.to(device) + mean.to(device)
        _feat_pred, desc_pred = sequence_features(pose_pred, pose_pred_raw)
        desc = F.smooth_l1_loss(desc_pred, desc_tgt)
        std_loss = std_match_loss(pose_pred, pose)
        speed_loss = speed_envelope_loss(pose_pred, pose)
        mcoll = motion_collapse_loss(pose_pred_raw, pose_raw) if args.lambda_motion_collapse > 0 else pose_pred_raw.new_zeros(())
        arc_budget = hand_arc_budget_loss(pose_pred_raw, pose_raw) if args.lambda_arc_budget > 0 else pose_pred_raw.new_zeros(())
        spectral = spectral_wrist_hand_loss(
            pose_pred_raw,
            pose_raw,
            kernel=args.spectral_kernel,
            hand_lo=args.spectral_hand_lo,
            hand_hi=args.spectral_hand_hi,
            wrist_lo=args.spectral_wrist_lo,
            wrist_hi=args.spectral_wrist_hi,
            jerk_cap=args.spectral_jerk_cap,
        ) if args.lambda_spectral_wrist > 0 else pose_pred_raw.new_zeros(())
        if wavkl_state is not None and args.lambda_wavkl > 0:
            wavkl = wavkl_hand_loss(
                pose_pred_raw,
                wavkl_state["gt_hists"],
                wavkl_state["bin_centers"],
                wavkl_state["sigma"],
                wavkl_state["scales"],
                reverse=wavkl_state.get("reverse", False),
            )
        else:
            wavkl = pose_pred_raw.new_zeros(())
        if args.lambda_vel_dir > 0:
            vel_dir = vel_direction_loss(pose_pred, pose,
                                         wrist_weight=args.vel_dir_wrist_weight)
        else:
            vel_dir = pose_pred.new_zeros(())
        if args.lambda_jepa_pc > 0:
            jepa_pc = jepa_predictive_consistency_loss(
                jepa,
                pose_pred,
                pose_pred_raw,
                z_tgt,
                gloss_id,
                len_id,
                args,
            )
        else:
            jepa_pc = pose_pred.new_zeros(())
        loss = (
            args.lambda_latent * latent
            + args.lambda_pose * pose_loss
            + args.lambda_desc * desc
            + args.lambda_std * std_loss
            + args.lambda_speed * speed_loss
            + args.lambda_motion_collapse * mcoll
            + args.lambda_arc_budget * arc_budget
            + args.lambda_spectral_wrist * spectral
            + args.lambda_wavkl * wavkl
            + args.lambda_energy_pred * energy_pred_loss
            + args.lambda_vel_dir * vel_dir
            + args.lambda_jepa_pc * jepa_pc
        )
        vel_dir_vals.append(float(vel_dir.detach().cpu()))
        jepa_pc_vals.append(float(jepa_pc.detach().cpu()))
        # Motion ratios on raw poses for direct read of hand_speed gate.
        pred_v = (pose_pred_raw[:, 1:] - pose_pred_raw[:, :-1]).reshape(
            pose_pred_raw.shape[0], pose_pred_raw.shape[1] - 1, 178, 3)
        gt_v = (pose_raw[:, 1:] - pose_raw[:, :-1]).reshape(
            pose_raw.shape[0], pose_raw.shape[1] - 1, 178, 3)
        ps = pred_v[:, :, 8:50].norm(dim=-1).mean()
        ts = gt_v[:, :, 8:50].norm(dim=-1).mean()
        hand_speed_ratios.append(float((ps / ts.clamp_min(1e-9)).cpu()))
        pred_a = pred_v[:, 1:] - pred_v[:, :-1]
        gt_a = gt_v[:, 1:] - gt_v[:, :-1]
        pj = pred_a[:, :, 8:50].norm(dim=-1).mean()
        tj = gt_a[:, :, 8:50].norm(dim=-1).mean()
        hand_jerk_ratios.append(float((pj / tj.clamp_min(1e-9)).cpu()))
        if args.use_energy_cond:
            if args.use_sentence_budget:
                energy_inf = model.predict_condition_energy(
                    gloss_id,
                    len_id,
                    batch["sent_gloss_ids"].to(device),
                    batch["sent_mask"].to(device),
                    batch["sent_pos"].to(device),
                ).detach().clamp_min(0.0)
            else:
                energy_inf = model.predict_energy(gloss_id, len_id).detach().clamp_min(0.0)
            _z_pc, pose_pc, _lv_pc = model(gloss_id, len_id, pose.shape[1], energy=energy_inf)
            pose_pc_raw = pose_pc * std.to(device) + mean.to(device)
            pc_v = (pose_pc_raw[:, 1:] - pose_pc_raw[:, :-1]).reshape(
                pose_pc_raw.shape[0], pose_pc_raw.shape[1] - 1, 178, 3)
            pc_s = pc_v[:, :, 8:50].norm(dim=-1).mean()
            predcond_hand_speed_ratios.append(float((pc_s / ts.clamp_min(1e-9)).cpu()))
            pc_a = pc_v[:, 1:] - pc_v[:, :-1]
            pc_j = pc_a[:, :, 8:50].norm(dim=-1).mean()
            predcond_hand_jerk_ratios.append(float((pc_j / tj.clamp_min(1e-9)).cpu()))
        vals.append(float(loss.cpu()))
        cos_vals.append(float((pred_n * tgt_n).sum(dim=-1).mean().cpu()))
        pose_vals.append(float(pose_loss.cpu()))
        desc_vals.append(float(desc.cpu()))
        wavkl_vals.append(float(wavkl.cpu()))
        energy_vals.append(float(energy_pred_loss.cpu()))
        arc_vals.append(float(arc_budget.cpu()))
        spectral_vals.append(float(spectral.cpu()))
    return {
        "val_loss": float(np.mean(vals)),
        "val_cos": float(np.mean(cos_vals)),
        "val_pose": float(np.mean(pose_vals)),
        "val_desc": float(np.mean(desc_vals)),
        "val_wavkl": float(np.mean(wavkl_vals)) if wavkl_vals else 0.0,
        "val_energy_pred": float(np.mean(energy_vals)) if energy_vals else 0.0,
        "val_arc_budget": float(np.mean(arc_vals)) if arc_vals else 0.0,
        "val_spectral_wrist": float(np.mean(spectral_vals)) if spectral_vals else 0.0,
        "val_vel_dir": float(np.mean(vel_dir_vals)) if vel_dir_vals else 0.0,
        "val_jepa_pc": float(np.mean(jepa_pc_vals)) if jepa_pc_vals else 0.0,
        "val_hand_speed_ratio": float(np.mean(hand_speed_ratios)) if hand_speed_ratios else 0.0,
        "val_hand_jerk_ratio": float(np.mean(hand_jerk_ratios)) if hand_jerk_ratios else 0.0,
        "val_predcond_hand_speed_ratio": float(np.mean(predcond_hand_speed_ratios)) if predcond_hand_speed_ratios else 0.0,
        "val_predcond_hand_jerk_ratio": float(np.mean(predcond_hand_jerk_ratios)) if predcond_hand_jerk_ratios else 0.0,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jepa, jepa_ckpt = load_frozen_jepa(ROOT / args.jepa_ckpt, device)
    bank, spans = load_bank(ROOT / args.bank)
    if getattr(args, "exclude_sids_json", ""):
        banned = set(json.load(open(ROOT / args.exclude_sids_json)))
        n_before = len(spans)
        spans = [sp for sp in spans if sp[1] not in banned]
        print(f"[jepa-gen] excluded {n_before - len(spans)}/{n_before} spans from "
              f"{len(banned)} banned train clips", flush=True)
    gloss_to_id = jepa_ckpt["gloss_to_id"]
    if gloss_to_id != build_gloss_vocab(bank):
        print("[jepa-gen] warning: bank gloss vocab differs from JEPA checkpoint", flush=True)
    mean = jepa_ckpt["mean"].float()
    std = jepa_ckpt["std"].float()
    gloss_mean_lengths = mean_lengths_by_gloss(spans)
    row_by_sid = {}
    if args.use_sentence_budget and args.train_manifest_json:
        rows = json.load(open(ROOT / args.train_manifest_json))
        row_by_sid = {str(r["id"]): r for r in rows}

    random.Random(args.seed).shuffle(spans)
    n_val = max(args.min_val, int(args.val_frac * len(spans)))
    val_spans = spans[:n_val]
    train_spans = spans[n_val:]
    if args.smoke:
        train_spans = train_spans[:args.smoke_train]
        val_spans = val_spans[:args.smoke_val]

    seg_len = int(jepa_ckpt["args"]["seg_len"])
    if getattr(args, "seg_len_override", 0) > 0:
        # v28. Override the per-segment training/inference time-axis length.
        # The default seg_len=64 (from JEPA) upsamples typical Phoenix
        # 8-frame spans 8x, smoothing out high-frequency motion. A smaller
        # seg_len keeps fine motion content intact.
        seg_len = int(args.seg_len_override)
        print(f"[jepa-gen] seg_len overridden to {seg_len}", flush=True)
    train_ds = SLRTPSpanDataset(
        bank, train_spans, gloss_to_id, mean, std, seg_len,
        max_items=args.max_train_items, seed=args.seed,
        row_by_sid=row_by_sid, sent_max_len=args.sentence_max_len,
    )
    val_ds = SLRTPSpanDataset(
        bank, val_spans, gloss_to_id, mean, std, seg_len,
        max_items=args.max_val_items, seed=args.seed + 7,
        row_by_sid=row_by_sid, sent_max_len=args.sentence_max_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    hidden = int(jepa_ckpt["args"]["hidden"])
    model = SignJEPAGenerator(
        num_gloss=len(gloss_to_id),
        hidden=hidden,
        gen_layers=args.gen_layers,
        dec_layers=args.dec_layers,
        heads=args.heads,
        dropout=args.dropout,
        max_len=seg_len,
        style_dim=args.style_dim,
        use_energy_cond=args.use_energy_cond,
        energy_dim=args.energy_dim,
        use_sentence_budget=args.use_sentence_budget,
        sentence_max_len=args.sentence_max_len,
        stochastic_decoder=args.stochastic_decoder,
        init_log_var=args.init_log_var,
        latent_noise_std=args.latent_noise_std,
    ).to(device)
    if args.init_from:
        ckpt_path = ROOT / args.init_from
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[jepa-gen] init_from {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.freeze_gen:
        for p in model.generator.parameters():
            p.requires_grad_(False)
        print("[jepa-gen] freeze_gen=True; training decoder only", flush=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"[jepa-gen] device={device} train={len(train_ds)} val={len(val_ds)} "
        f"seg_len={seg_len} hidden={hidden} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
        f"trainable={sum(p.numel() for p in trainable_params)/1e6:.2f}M",
        flush=True,
    )

    discriminator = None
    d_opt = None
    if args.lambda_adv > 0:
        _crit_idx = list(range(178)) if getattr(args, "critic_full_body", False) else None
        discriminator = MotionDiscriminator(hidden=args.d_hidden, dropout=args.d_dropout,
                                            joint_idx=_crit_idx).to(device)
        d_opt = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr,
                                  weight_decay=0.0, betas=(0.0, 0.99))
        print(
            f"[jepa-gen] v29 discriminator hidden={args.d_hidden} "
            f"params={sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M "
            f"lambda_adv={args.lambda_adv} d_lr={args.d_lr}",
            flush=True,
        )

    anchor_model = None
    if args.lambda_anchor > 0 and args.anchor_ckpt:
        # v33. Load a frozen anchor carrier (typically v25) whose body+face
        # output is the L1 distillation target for v33's body+face. Hands
        # remain free for adversarial pressure.
        a_ck = torch.load(ROOT / args.anchor_ckpt, map_location=device, weights_only=False)
        anchor_model = SignJEPAGenerator(
            num_gloss=len(gloss_to_id),
            hidden=a_ck["hidden"],
            gen_layers=a_ck["args"]["gen_layers"],
            dec_layers=a_ck["args"]["dec_layers"],
            heads=a_ck["args"]["heads"],
            dropout=0.0,
            max_len=a_ck["seg_len"],
            style_dim=a_ck["args"].get("style_dim", 0),
            use_energy_cond=a_ck["args"].get("use_energy_cond", False),
            energy_dim=a_ck["args"].get("energy_dim", 16),
            use_sentence_budget=a_ck["args"].get("use_sentence_budget", False),
            sentence_max_len=a_ck["args"].get("sentence_max_len", 32),
            stochastic_decoder=a_ck["args"].get("stochastic_decoder", False),
            init_log_var=a_ck["args"].get("init_log_var", -4.0),
            latent_noise_std=a_ck["args"].get("latent_noise_std", 0.0),
        ).to(device)
        anchor_model.load_state_dict(a_ck["model"])
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad_(False)
        print(
            f"[jepa-gen] v33 anchor from {args.anchor_ckpt} "
            f"lambda_anchor={args.lambda_anchor}", flush=True,
        )

    wavkl_state = None
    if args.lambda_wavkl > 0:
        wavkl_scales = tuple(int(k) for k in str(args.wavkl_scales).split(",") if k.strip())
        wavkl_bins = int(args.wavkl_bins)
        wavkl_vmax = float(args.wavkl_vmax)
        wavkl_sigma = float(args.wavkl_sigma) if args.wavkl_sigma > 0 else (wavkl_vmax / wavkl_bins) * 0.6
        print(f"[jepa-gen] WAV-KL precomputing GT hand-velocity histograms "
              f"scales={wavkl_scales} bins={wavkl_bins} vmax={wavkl_vmax} sigma={wavkl_sigma}",
              flush=True)
        # Use a no-shuffle iterator over a fresh DataLoader so we don't disturb training order.
        hist_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        gt_hists, bin_centers = precompute_gt_wavkl(
            hist_loader, mean, std, device,
            scales=wavkl_scales, n_bins=wavkl_bins, vmax=wavkl_vmax,
            sigma=wavkl_sigma, max_batches=int(args.wavkl_max_batches),
        )
        wavkl_state = {
            "gt_hists": gt_hists,
            "bin_centers": bin_centers,
            "sigma": wavkl_sigma,
            "scales": wavkl_scales,
            "reverse": bool(getattr(args, "wavkl_reverse", False)),
        }
        # Quick summary of GT histograms (mass at first bin = how often GT is near-zero).
        for k, h in gt_hists.items():
            print(f"[jepa-gen] WAV-KL GT k={k}: zero-bin={h[0].item():.4f} "
                  f"peak-bin={h.argmax().item()} peak-mass={h.max().item():.4f}",
                  flush=True)

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, jepa, train_loader, opt, mean, std, device, args,
                         wavkl_state=wavkl_state,
                         discriminator=discriminator, d_opt=d_opt,
                         anchor_model=anchor_model)
        va = eval_epoch(model, jepa, val_loader, mean, std, device, args, wavkl_state=wavkl_state)
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **tr, **va}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best": best,
                    "jepa_ckpt": str(args.jepa_ckpt),
                    "gloss_to_id": gloss_to_id,
                    "mean": mean,
                    "std": std,
                    "seg_len": seg_len,
                    "hidden": hidden,
                    "gloss_mean_lengths": gloss_mean_lengths,
                    "discriminator": discriminator.state_dict() if discriminator is not None else None,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[jepa-gen] saved best -> {out_dir / 'best.pt'}", flush=True)
    print(f"[jepa-gen] done best={best}", flush=True)


def allocate_lengths(
    glosses: list[str],
    total_len: int,
    mean_lengths: dict[str, float] | None = None,
    fallback: int = 8,
) -> list[int]:
    if not glosses:
        return []
    mean_lengths = mean_lengths or {}
    base = np.asarray(
        [max(1.0, float(mean_lengths.get(g, fallback))) for g in glosses],
        dtype=np.float64,
    )
    raw = base / base.sum() * max(len(glosses), int(total_len))
    lens = np.maximum(1, np.floor(raw).astype(np.int64))
    diff = int(total_len) - int(lens.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    j = 0
    while diff > 0:
        lens[order[j % len(lens)]] += 1
        diff -= 1
        j += 1
    while diff < 0:
        idx = int(np.argmax(lens))
        if lens[idx] <= 1:
            break
        lens[idx] -= 1
        diff += 1
    return [int(x) for x in lens]


def duration_policy_features(row: dict, policy: dict) -> list[float]:
    glosses = [g for g in str(row.get("gloss", "")).split() if g]
    text_words = [w for w in str(row.get("text", "")).split() if w]
    n_gloss = max(1, len(glosses))
    gloss_counts = policy.get("gloss_counts", {})
    signers = list(policy.get("signers", []))
    lexicon = list(policy.get("lexicon", []))
    signer = str(row.get("signer", ""))
    gloss_set = set(glosses)
    freq_sum = sum(float(gloss_counts.get(g, 0.0)) for g in glosses)
    feature_mode = str(policy.get("feature_mode", "legacy_gt_length"))
    feats = [
        1.0,
        float(len(glosses)),
        float(len(text_words)),
        float(len(str(row.get("text", "")))),
        float(len(text_words)) / n_gloss,
        freq_sum / n_gloss,
        float(sum(1 for g in glosses if float(gloss_counts.get(g, 0.0)) < 10.0)),
        float(sum(1 for g in glosses if "-" in g)),
        float(row.get("fps", 25) or 25),
    ]
    if feature_mode == "legacy_gt_length":
        base_len = float(row.get("length", 0) or 0)
        feats[4:4] = [base_len, base_len / n_gloss]
    feats.extend(1.0 if signer == s else 0.0 for s in signers)
    feats.extend(1.0 if g in gloss_set else 0.0 for g in lexicon)
    return feats


def load_duration_policy(path: str) -> dict | None:
    if not path:
        return None
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Loading --duration_policy requires joblib/sklearn in the active env") from exc
    policy = joblib.load(ROOT / path)
    if not isinstance(policy, dict) or "model" not in policy or "scales" not in policy:
        raise ValueError(f"Invalid duration policy file: {path}")
    return policy


def predict_duration_scale(row: dict, policy: dict) -> float:
    x = np.asarray([duration_policy_features(row, policy)], dtype=np.float32)
    pred = np.asarray(policy["model"].predict(x))[0]
    scales = [float(s) for s in policy["scales"]]
    if pred.ndim == 0:
        return float(np.clip(pred.item(), min(scales), max(scales)))
    best_idx = int(np.argmin(pred))
    base_scale = float(policy.get("base_scale", min(scales)))
    margin = float(policy.get("switch_margin", 0.0))
    if margin > 0.0 and base_scale in scales:
        base_idx = scales.index(base_scale)
        if float(pred[base_idx] - pred[best_idx]) < margin:
            return base_scale
    return scales[best_idx]


def predict_duration_total_len(row: dict, policy: dict) -> int | None:
    """Predict deployable total frames from text/gloss-only policy features."""
    length_model = policy.get("length_model")
    if length_model is None:
        return None
    length_policy = {**policy, "feature_mode": str(policy.get("length_feature_mode", "text_only"))}
    x = np.asarray([duration_policy_features(row, length_policy)], dtype=np.float32)
    pred_log_len = float(np.asarray(length_model.predict(x)).reshape(-1)[0])
    pred_len = int(round(math.exp(pred_log_len)))
    min_len = int(policy.get("length_min", 4))
    max_len = int(policy.get("length_max", 512))
    gloss_count = len([g for g in str(row.get("gloss", "")).split() if g])
    return int(np.clip(pred_len, max(1, gloss_count, min_len), max_len))


def sentence_tensors_from_glosses(
    glosses: list[str],
    gloss_to_id: dict[str, int],
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sent_ids = torch.zeros(1, max_len, dtype=torch.long, device=device)
    sent_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    for j, g in enumerate(glosses[:max_len]):
        sent_ids[0, j] = int(gloss_to_id.get(g, 1))
        sent_mask[0, j] = True
    if not glosses:
        sent_mask[0, 0] = True
    return sent_ids, sent_mask


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ckpt["gloss_to_id"]
    model = SignJEPAGenerator(
        num_gloss=len(gloss_to_id),
        hidden=ckpt["hidden"],
        gen_layers=ckpt["args"]["gen_layers"],
        dec_layers=ckpt["args"]["dec_layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
        max_len=ckpt["seg_len"],
        style_dim=ckpt["args"].get("style_dim", 0),
        use_energy_cond=ckpt["args"].get("use_energy_cond", False),
        energy_dim=ckpt["args"].get("energy_dim", 16),
        use_sentence_budget=ckpt["args"].get("use_sentence_budget", False),
        sentence_max_len=ckpt["args"].get("sentence_max_len", 32),
        stochastic_decoder=ckpt["args"].get("stochastic_decoder", False),
        init_log_var=ckpt["args"].get("init_log_var", -4.0),
        latent_noise_std=ckpt["args"].get("latent_noise_std", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    mean_lengths = ckpt.get("gloss_mean_lengths", {})
    trace = json.load(open(ROOT / args.trace_json))
    items = list(trace.items())
    if args.max_clips:
        items = items[:args.max_clips]
    out = {}
    for i, (sid, row) in enumerate(items, start=1):
        chunks = []
        for seg in row.get("segments", []):
            glosses = [str(g) for g in seg.get("glosses", [])]
            sent_ids, sent_mask = sentence_tensors_from_glosses(
                glosses, gloss_to_id, ckpt["args"].get("sentence_max_len", 32), device)
            total_len = int(seg.get("out_len", 0))
            if total_len <= 0 or not glosses:
                continue
            lengths = allocate_lengths(glosses, total_len, mean_lengths)
            for pos_i, (gloss, L) in enumerate(zip(glosses, lengths)):
                gid = torch.tensor([gloss_to_id.get(gloss, 1)], dtype=torch.long, device=device)
                lid = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
                style = None
                style_dim = ckpt["args"].get("style_dim", 0)
                if style_dim > 0 and args.style_sample_std > 0:
                    style = torch.randn(1, style_dim, device=device) * args.style_sample_std
                energy = None
                if ckpt["args"].get("use_energy_cond", False):
                    if ckpt["args"].get("use_sentence_budget", False):
                        spos = torch.tensor([min(pos_i, sent_ids.shape[1] - 1)], dtype=torch.long, device=device)
                        energy = model.predict_condition_energy(gid, lid, sent_ids, sent_mask, spos)
                    else:
                        energy = model.predict_energy(gid, lid)
                    energy = energy.detach().clamp_min(0.0)
                _z, pose, log_var = model(gid, lid, ckpt["seg_len"], style=style, energy=energy)
                if log_var is not None and getattr(args, "sample_temperature", 0.0) > 0:
                    std_per = (0.5 * log_var.clamp(min=-8.0, max=2.0)).exp()
                    pose = pose + std_per * torch.randn_like(pose) * float(args.sample_temperature)
                pose = pose[0] * std + mean
                pose = resize_seq(pose.cpu(), L).reshape(L, 178, 3).float()
                chunks.append(pose)
        if chunks:
            out[sid] = torch.cat(chunks, dim=0).contiguous()
        else:
            out[sid] = torch.zeros(4, 178, 3)
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[jepa-gen-sample] {i}/{len(items)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    lens = np.asarray([v.shape[0] for v in out.values()])
    print(f"saved -> {out_path}")
    print(f"T mean={lens.mean():.1f} p10/50/90={np.percentile(lens, [10,50,90]).tolist()}")


@torch.no_grad()
def sample_manifest(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ckpt["gloss_to_id"]
    model = SignJEPAGenerator(
        num_gloss=len(gloss_to_id),
        hidden=ckpt["hidden"],
        gen_layers=ckpt["args"]["gen_layers"],
        dec_layers=ckpt["args"]["dec_layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
        max_len=ckpt["seg_len"],
        style_dim=ckpt["args"].get("style_dim", 0),
        use_energy_cond=ckpt["args"].get("use_energy_cond", False),
        energy_dim=ckpt["args"].get("energy_dim", 16),
        use_sentence_budget=ckpt["args"].get("use_sentence_budget", False),
        sentence_max_len=ckpt["args"].get("sentence_max_len", 32),
        stochastic_decoder=ckpt["args"].get("stochastic_decoder", False),
        init_log_var=ckpt["args"].get("init_log_var", -4.0),
        latent_noise_std=ckpt["args"].get("latent_noise_std", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # v26. Allow inference-time override of the trained latent noise scale.
    if getattr(args, "latent_noise_scale", -1.0) >= 0:
        model.latent_noise_std = float(args.latent_noise_scale)
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    mean_lengths = ckpt.get("gloss_mean_lengths", {})
    duration_policy = load_duration_policy(getattr(args, "duration_policy", ""))
    if not getattr(args, "allow_gt_length_base", False):
        if duration_policy is None or duration_policy.get("length_model") is None:
            raise ValueError(
                "sample_manifest no longer uses manifest row['length'] by default. "
                "Pass a duration policy containing a text-only length_model, or use "
                "--allow_gt_length_base for an explicit legacy/leaky diagnostic run."
            )

    rows = json.load(open(ROOT / args.manifest_json))
    ref_ids = None
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=True)
        ref_ids = list(ref.keys())
        row_by_id = {str(r["id"]): r for r in rows}
        missing = [sid for sid in ref_ids if sid not in row_by_id]
        if missing:
            raise KeyError(f"{len(missing)} reference ids missing from manifest, first={missing[:3]}")
        rows = [row_by_id[sid] for sid in ref_ids]
    if args.max_clips:
        rows = rows[:args.max_clips]
    out = {}
    scale_rows = []
    skipped = 0
    for i, row in enumerate(rows, start=1):
        sid = str(row["id"])
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        scale = 1.0
        len_source = "text_policy"
        pred_total_len = predict_duration_total_len(row, duration_policy) if duration_policy is not None else None
        if pred_total_len is not None:
            total_len = pred_total_len
            scale = float(args.length_scale)
        else:
            len_source = "manifest_length_legacy"
            total_len = int(row.get("length", 0) or 0)
            scale = float(args.length_scale)
            if duration_policy is not None:
                scale = predict_duration_scale(row, duration_policy)
        if scale != 1.0 and total_len > 0:
            total_len = max(len(glosses), int(round(total_len * scale)))
        if not glosses or total_len <= 0:
            skipped += 1
            continue
        scale_rows.append({"id": sid, "scale": scale, "length": total_len, "length_source": len_source})
        chunks = []
        lengths = allocate_lengths(glosses, total_len, mean_lengths)
        sent_ids, sent_mask = sentence_tensors_from_glosses(
            glosses, gloss_to_id, ckpt["args"].get("sentence_max_len", 32), device)
        for pos_i, (gloss, L) in enumerate(zip(glosses, lengths)):
            gid = torch.tensor([gloss_to_id.get(gloss, 1)], dtype=torch.long, device=device)
            lid = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
            style = None
            style_dim = ckpt["args"].get("style_dim", 0)
            if style_dim > 0 and args.style_sample_std > 0:
                style = torch.randn(1, style_dim, device=device) * args.style_sample_std
            energy = None
            if ckpt["args"].get("use_energy_cond", False):
                if ckpt["args"].get("use_sentence_budget", False):
                    spos = torch.tensor([min(pos_i, sent_ids.shape[1] - 1)], dtype=torch.long, device=device)
                    energy = model.predict_condition_energy(gid, lid, sent_ids, sent_mask, spos)
                else:
                    energy = model.predict_energy(gid, lid)
                energy = energy.detach().clamp_min(0.0)
                if abs(args.energy_lift - 1.0) > 1e-6:
                    # v24b. Lift the predicted motion budget in raw motion-energy
                    # space, then re-log1p to keep the carrier's input
                    # distribution shape.
                    desc_scale = float(ckpt["args"].get("energy_desc_scale", 0.01))
                    energy_raw = (torch.expm1(energy) * desc_scale).clamp_min(0.0)
                    energy_raw = energy_raw * float(args.energy_lift)
                    energy = torch.log1p(energy_raw / max(desc_scale, 1e-9))
            # v27 fix. Default behaviour generates each segment at T=seg_len
            # then resize_seq downsamples by 5-15x to the allocated length L,
            # which low-passes hand velocity and collapses hand_speed at the
            # full-utterance level. --gen_at_target_T generates at T=L
            # directly so motion magnitude is preserved.
            if getattr(args, "gen_at_target_T", False):
                gen_T = max(2, int(L))
                # Clip to seg_len to stay inside the carrier's trained
                # positional-encoding range.
                gen_T = min(gen_T, int(ckpt["seg_len"]))
            else:
                gen_T = int(ckpt["seg_len"])
            _z, pose, log_var = model(gid, lid, gen_T, style=style, energy=energy)
            if log_var is not None and getattr(args, "sample_temperature", 0.0) > 0:
                std_per = (0.5 * log_var.clamp(min=-8.0, max=2.0)).exp()
                pose = pose + std_per * torch.randn_like(pose) * float(args.sample_temperature)
            pose = pose[0] * std + mean
            if pose.shape[0] != L:
                pose = resize_seq(pose.cpu(), L).reshape(L, 178, 3).float()
            else:
                pose = pose.cpu().reshape(L, 178, 3).float()
            chunks.append(pose)
        out[sid] = torch.cat(chunks, dim=0).contiguous()
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[jepa-gen-manifest] {i}/{len(rows)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    lens = np.asarray([v.shape[0] for v in out.values()])
    print(f"saved -> {out_path}")
    print(f"clips={len(out)} skipped={skipped}")
    print(f"T mean={lens.mean():.1f} p10/50/90={np.percentile(lens, [10,50,90]).tolist()}")
    if args.dump_duration_scales:
        dump_path = ROOT / args.dump_duration_scales
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(scale_rows, indent=2))
        print(f"duration scales -> {dump_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--jepa_ckpt", default="outputs/sota_chase/sign_jepa_slrtp178/best.pt")
    tr.add_argument("--bank", default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_generator_slrtp178")
    tr.add_argument("--gen_layers", type=int, default=4)
    tr.add_argument("--dec_layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_latent", type=float, default=1.0)
    tr.add_argument("--latent_loss_mode", choices=["cosine", "raw", "both"], default="cosine",
                    help="rich-latent SLP: cosine=direction-only (original, magnitude-blind); "
                         "raw=full unnormalized z (matches carrier richness); both=cosine+raw.")
    tr.add_argument("--latent_raw_weight", type=float, default=1.0,
                    help="weight on the raw (magnitude) term when --latent_loss_mode both.")
    tr.add_argument("--latent_cov_weight", type=float, default=0.0,
                    help="rich-latent SLP: weight on the latent-covariance (rank/spectrum) "
                         "matching term; forces z_pred to reproduce the carrier's effective rank.")
    tr.add_argument("--lambda_jepa_pc", type=float, default=0.0,
                    help="Weight for JEPA predictive-consistency loss. Future generated "
                         "frames are masked and the frozen carrier predictor must recover "
                         "the GT target latent from visible generated context. This makes "
                         "the latent-prediction pretraining objective load-bearing.")
    tr.add_argument("--jepa_pc_loss_mode", choices=["cosine", "raw", "both"], default="cosine",
                    help="Latent distance used inside --lambda_jepa_pc.")
    tr.add_argument("--jepa_pc_raw_weight", type=float, default=1.0,
                    help="Raw-latent weight when --jepa_pc_loss_mode both.")
    tr.add_argument("--jepa_pc_cov_weight", type=float, default=0.0,
                    help="Optional covariance/rank matching weight inside --lambda_jepa_pc.")
    tr.add_argument("--jepa_pc_min_context", type=int, default=16,
                    help="Minimum visible context for predictive-consistency masks.")
    tr.add_argument("--jepa_pc_target_len_min", type=int, default=8,
                    help="Minimum held-out future span length for predictive consistency.")
    tr.add_argument("--jepa_pc_target_len_max", type=int, default=24,
                    help="Maximum held-out future span length for predictive consistency.")
    tr.add_argument("--lambda_pose", type=float, default=0.2)
    tr.add_argument("--lambda_vel", type=float, default=0.2)
    tr.add_argument("--lambda_desc", type=float, default=0.3)
    tr.add_argument("--lambda_std", type=float, default=0.0)
    tr.add_argument("--lambda_speed", type=float, default=0.0)
    tr.add_argument("--lambda_motion_collapse", type=float, default=0.0,
                    help="weight for the asymmetric motion-density auxiliary loss "
                         "(the fix for JEPA motion collapse — see motion_collapse_loss)")
    tr.add_argument("--lambda_arc_budget", type=float, default=0.0,
                    help="Weight for duration-robust hand arc-length/burst matching with a jerk cap.")
    tr.add_argument("--lambda_spectral_wrist", type=float, default=0.0,
                    help="Weight for low-frequency wrist/hand velocity band loss with local-shape guardrails.")
    tr.add_argument("--spectral_kernel", type=int, default=7)
    tr.add_argument("--spectral_hand_lo", type=float, default=0.90)
    tr.add_argument("--spectral_hand_hi", type=float, default=1.10)
    tr.add_argument("--spectral_wrist_lo", type=float, default=0.90)
    tr.add_argument("--spectral_wrist_hi", type=float, default=1.15)
    tr.add_argument("--spectral_jerk_cap", type=float, default=1.20)
    tr.add_argument("--lambda_vel_dir", type=float, default=0.0,
                    help="v23. Weight for per-frame velocity-direction cosine loss "
                         "on hands and wrists vs GT (magnitude-invariant). 0 disables.")
    tr.add_argument("--vel_dir_wrist_weight", type=float, default=0.5,
                    help="v24a. Relative weight of wrist {2,5} contribution within "
                         "vel_direction_loss (v23 used 0.5; set 0.0 for fingers-only).")
    tr.add_argument("--critic_full_body", action="store_true",
                    help="Ablation: discriminate over all 178 joints instead of "
                         "the hand+wrist region (hand-vs-full-body critic).")
    tr.add_argument("--stochastic_decoder", action="store_true",
                    help="v25. Predict heteroscedastic Gaussian (mean, log_var) per pose "
                         "element instead of deterministic point. Replaces L1 pose loss "
                         "with Gaussian NLL when --lambda_nll > 0.")
    tr.add_argument("--init_log_var", type=float, default=-4.0,
                    help="v25. Initial bias of the log-variance head. -4 = std≈0.135.")
    tr.add_argument("--lambda_nll", type=float, default=0.0,
                    help="v25. If > 0 and --stochastic_decoder, Gaussian NLL replaces "
                         "the deterministic L1 pose target inside lambda_pose.")
    tr.add_argument("--latent_noise_std", type=float, default=0.0,
                    help="v26. Std of the per-clip coherent latent noise injected into "
                         "the generator's z before the pose decoder. 0 disables.")
    tr.add_argument("--seg_len_override", type=int, default=0,
                    help="v28. Override the training/inference per-segment T (default "
                         "uses JEPA's seg_len=64). Smaller seg_len means less data upsample "
                         "during training, preserving high-frequency motion. 0 disables.")
    tr.add_argument("--lambda_adv", type=float, default=0.0,
                    help="v29. Weight of generator's adversarial loss against the motion "
                         "discriminator. 0 disables (no D trained either).")
    tr.add_argument("--d_lr", type=float, default=2e-4,
                    help="v29. Discriminator learning rate (typically 2-4x G's lr).")
    tr.add_argument("--d_hidden", type=int, default=64,
                    help="v29. Discriminator base channel count.")
    tr.add_argument("--d_dropout", type=float, default=0.1,
                    help="v29. Discriminator dropout.")
    tr.add_argument("--anchor_ckpt", default="",
                    help="v33. Path to frozen anchor generator checkpoint (e.g. v25). "
                         "If set + --lambda_anchor > 0, body+face regions of pose_pred are "
                         "L1-distilled to the anchor's output at every batch.")
    tr.add_argument("--lambda_anchor", type=float, default=0.0,
                    help="v33. Weight of the body+face L1 distillation toward anchor_ckpt's "
                         "output. Hands (joints 8:50) are NOT anchored — adversarial only.")
    tr.add_argument("--lambda_jerk", type=float, default=0.0,
                    help="Train-time temporal-smoothness regularizer (R1.7/R2.1): "
                         "penalize mean third-difference magnitude of hand joints 8:50 "
                         "(raw units). 0 disables.")
    tr.add_argument("--lambda_wavkl", type=float, default=0.0,
                    help="Weight for multi-scale WAV-KL hand-velocity-magnitude distribution loss (0 disables).")
    tr.add_argument("--wavkl_scales", default="1,3,7,15",
                    help="Comma-separated temporal smoothing kernels defining the wavelet-like scale hierarchy.")
    tr.add_argument("--wavkl_bins", type=int, default=64,
                    help="Number of bins in the velocity-magnitude histograms.")
    tr.add_argument("--wavkl_vmax", type=float, default=0.15,
                    help="Maximum velocity-magnitude bin (raw pose units). Should cover the 99th percentile of GT.")
    tr.add_argument("--wavkl_sigma", type=float, default=0.0,
                    help="Soft-bin kernel width. 0 = auto: 0.6 * (vmax / n_bins).")
    tr.add_argument("--wavkl_max_batches", type=int, default=0,
                    help="Max train batches used to estimate the GT histogram (0 = all).")
    tr.add_argument("--wavkl_reverse", action="store_true",
                    help="Ablation: use reverse (mode-seeking) KL(gt||pred) instead of "
                         "the default forward (mass-covering) KL(pred||gt).")
    tr.add_argument("--use_energy_cond", action="store_true",
                    help="Condition the latent carrier on a 16-D hand energy-budget descriptor.")
    tr.add_argument("--energy_dim", type=int, default=16,
                    help="Energy descriptor dimension; keep at 16 for the built-in descriptor.")
    tr.add_argument("--energy_desc_scale", type=float, default=0.01,
                    help="Raw-unit scale for log1p energy descriptor compression.")
    tr.add_argument("--lambda_energy_pred", type=float, default=0.0,
                    help="Weight for gloss-id + length-bucket -> energy descriptor predictor loss.")
    tr.add_argument("--use_sentence_budget", action="store_true",
                    help="Add a sentence-context residual budget allocator on top of the per-gloss energy predictor.")
    tr.add_argument("--sentence_max_len", type=int, default=32)
    tr.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--exclude_sids_json", type=str, default="",
                    help="JSON list of train clip ids to exclude from every training span "
                         "(leave-source-out counterfactual retraining)")
    tr.add_argument("--init_from", type=str, default="",
                    help="path to an existing generator checkpoint to fine-tune from")
    tr.add_argument("--freeze_gen", action="store_true",
                    help="freeze the latent program generator and train only the pose decoder")
    tr.add_argument("--style_dim", type=int, default=0)
    tr.add_argument("--style_train_std", type=float, default=1.0)
    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--min_val", type=int, default=512)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--max_train_items", type=int, default=0)
    tr.add_argument("--max_val_items", type=int, default=2048)
    tr.add_argument("--drop_last", action="store_true")
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=256)
    tr.add_argument("--smoke_val", type=int, default=96)

    sp = sub.add_parser("sample")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--trace_json", required=True)
    sp.add_argument("--out_pt", required=True)
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sp.add_argument("--style_sample_std", type=float, default=0.0)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--max_clips", type=int, default=0)

    sm = sub.add_parser("sample_manifest")
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--manifest_json", required=True)
    sm.add_argument("--out_pt", required=True)
    sm.add_argument(
        "--reference_pt",
        default="",
        help="Optional organizer GT .pt; if set, filter/order manifest rows to these keys.",
    )
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--style_sample_std", type=float, default=0.0)
    sm.add_argument("--length_scale", type=float, default=1.0)
    sm.add_argument("--duration_policy", default="")
    sm.add_argument(
        "--allow_gt_length_base",
        action="store_true",
        help="Legacy diagnostic only: allow manifest row['length'] as the duration base.",
    )
    sm.add_argument("--dump_duration_scales", default="")
    sm.add_argument("--energy_lift", type=float, default=1.0,
                    help="v24b. Multiply the predicted hand energy descriptor by this "
                         "factor in raw (motion-energy) space before re-log1p. Lifts the "
                         "carrier's motion budget at inference. 1.0 = no change.")
    sm.add_argument("--sample_temperature", type=float, default=0.0,
                    help="v25. Stochastic-decoder sampling temperature; 0 disables noise "
                         "injection (deterministic mean). 1.0 draws from N(mean, var).")
    sm.add_argument("--latent_noise_scale", type=float, default=-1.0,
                    help="v26. Override the model's trained latent_noise_std at inference. "
                         "Negative = use the trained value as-is. 0 = deterministic.")
    sm.add_argument("--gen_at_target_T", action="store_true",
                    help="v27. Generate each gloss segment at T=L (the allocated frame "
                         "count) instead of T=seg_len followed by resize_seq downsample. "
                         "Preserves full-utterance hand_speed (resize 5-15x downsample "
                         "low-passes velocity).")
    sm.add_argument("--log_every", type=int, default=50)
    sm.add_argument("--max_clips", type=int, default=0)

    args = ap.parse_args()
    if args.mode == "train":
        if args.jepa_pc_min_context + args.jepa_pc_target_len_min > int(args.seg_len_override or 64):
            raise ValueError(
                "--jepa_pc_min_context + --jepa_pc_target_len_min must fit within "
                "the training segment length"
            )
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
