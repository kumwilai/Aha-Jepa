"""JRT-DMB: JEPA Residual Transport with Dominance-Margin Barrier.

A learned bounded residual transport field on top of a frozen JEPA carrier:

    X = C_jepa + R_theta(C_jepa, v_C, t, region, text_ctx)

with `R_theta` a small Transformer that outputs a per-frame per-joint
displacement, bounded in two ways:

  (a) tanh * region_max_scale  (hard per-axis bound at the output head)
  (b) JEPA-dominance margin barrier:
        ||R_clip||_F / ||T_clip - C_clip||_F  <=  rho
      enforced softly during training (squared hinge) against a frozen
      teacher reference T (the public CAC arm). At inference T is used
      ONLY to compute an audit certificate; the deployed pose X = C + R
      depends only on the learned model and the text-side inputs. Gloss
      conditioning is retained as an explicit oracle/diagnostic mode.

This is NOT teacher interpolation: the deployed X = C + R_theta(.)
never directly depends on the teacher pose. The teacher pose only enters
training (a) as the dominance budget reference and (b) optionally as a
weak directional guidance term added to the GT-pose imitation loss.

Loss
====
  L_pose         : Huber(X, GT_resized_to_T_C) — primary supervision
  L_dir          : Huber direction term —
                   small Huber on R vs alpha_train * (T_resized - C);
                   alpha_train is a SINGLE fixed scalar inherited from
                   the previously-locked dev protocol (default 0.40),
                   never tuned on test.
  L_dom          : JEPA-dominance soft barrier
                   relu(||R||_F / ||T - C||_F - rho)^2 per clip
  L_pres_body    : MSE on body residual (region preservation)
  L_pres_face    : MSE on face residual (region preservation)
  L_smooth       : residual acceleration L2
  L_bone         : per-frame hand finger bone-length consistency on X
                   (bones from R-disturbed pose match the carrier bones)
  L_wdist        : wrist total-distance ratio toward 1.0
                   relu(|ratio - 1.0| - dist_tol)^2
  L_lex          : optional frozen official back-translator lexical coverage
                   loss on X; by default this is anchor-relative, penalizing
                   only tokens whose teacher-forced CE is worse than the
                   frozen JEPA carrier. Gradients flow into R_theta only.

Subcommands
===========
  train    : fit R_theta on a dev-split inner-train subset, pick by inner-val
  generate : run R_theta to produce X for a held-out carrier .pt (no teacher)
  certify  : compute per-clip ||R||/||T-C|| dominance certificate and region shares

The deployed test artifact passes "JEPA-majority" iff the certificate has
mean per-clip ratio < 0.5 AND the per-clip pass rate at rho < 0.5 is high.

Honesty
=======
- alpha_train is FIXED before any test eval. If you change it after seeing
  test metrics, that's test-tuning and forfeits publishability — flag it.
- Inner-val selection happens on a strict held-out 20 % of dev clip ids,
  invisible to optimisation. Test is touched only for final inference.
- Total Distance and Average Duration are scored by closeness to 1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
SLRTP_ROOT = ROOT / "external/SLRTP-Sign-Production-Evaluation"
sys.path.insert(0, str(SLRTP_ROOT))


# Joint regions in SLRTP-178 layout (body 0:8, RH 8:29, LH 29:50, face 50:178).
BODY = slice(0, 8)
RH = slice(8, 29)
LH = slice(29, 50)
HAND = slice(8, 50)
FACE = slice(50, 178)
RWRIST, LWRIST = 2, 5

# 24 hand finger / palm bones in MediaPipe-style indexing.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (2, 5),
]


# ---------------------------------------------------------------------------
# Pose loading helpers
# ---------------------------------------------------------------------------


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    pose = torch.as_tensor(x).float()
    if pose.ndim == 2:
        pose = pose.reshape(pose.shape[0], 178, 3)
    if pose.ndim != 3 or pose.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(pose.shape)}")
    return pose


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def load_ref_meta(path: Path) -> dict[str, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(k): v for k, v in data.items()}


def resize_pose(pose: torch.Tensor, length: int) -> torch.Tensor:
    if pose.shape[0] == length:
        return pose
    y = pose.reshape(pose.shape[0], -1).T[None]
    y = F.interpolate(y, size=int(length), mode="linear", align_corners=True)
    return y[0].T.reshape(int(length), 178, 3).contiguous()


# ---------------------------------------------------------------------------
# Text / gloss conditioning vocab
# ---------------------------------------------------------------------------


def condition_tokens(meta: dict, field: str) -> list[str]:
    """Return conditioning tokens available at inference.

    `text` is the clean task input for text-to-pose. `gloss` is retained for
    backwards-compatible oracle/diagnostic runs only.
    """
    if field == "none":
        return []
    raw = str(meta.get(field, ""))
    if field == "text":
        return re.findall(r"\w+", raw.lower(), flags=re.UNICODE)
    return [g for g in raw.split() if g]


def build_condition_vocab(ref_metas: list[dict], field: str, min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in ref_metas:
        for tok in condition_tokens(r, field):
            counts[tok] = counts.get(tok, 0) + 1
    voc = {"<pad>": 0, "<unk>": 1}
    for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c < min_count:
            continue
        voc[tok] = len(voc)
    return voc


def text_words(meta: dict) -> list[str]:
    return re.findall(r"\w+", str(meta.get("text", "")).lower(), flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class JRTDataset(Dataset):
    """Per-clip iterator yielding (carrier, teacher_resized, GT, ctx)."""

    def __init__(
        self,
        ids: list[str],
        carrier: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
        ref: dict[str, dict],
        token_to_id: dict[str, int],
        condition_field: str,
        sent_max_len: int,
    ):
        self.ids = list(ids)
        self.carrier = carrier
        self.teacher = teacher
        self.ref = ref
        self.token_to_id = token_to_id
        self.condition_field = condition_field
        self.sent_max_len = int(sent_max_len)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        C = self.carrier[sid].float()
        T_pose = resize_pose(self.teacher[sid].float(), C.shape[0])
        meta = self.ref[sid]
        GT = as_pose(meta).float()
        if GT.shape[0] != C.shape[0]:
            GT = resize_pose(GT, C.shape[0])
        tokens = condition_tokens(meta, self.condition_field)
        sent_ids = torch.zeros(self.sent_max_len, dtype=torch.long)
        sent_mask = torch.zeros(self.sent_max_len, dtype=torch.bool)
        for j, tok in enumerate(tokens[: self.sent_max_len]):
            sent_ids[j] = int(self.token_to_id.get(tok, 1))
            sent_mask[j] = True
        if sent_mask.sum() == 0:
            sent_mask[0] = True
        return {
            "sid": sid,
            "C": C.contiguous(),
            "T": T_pose.contiguous(),
            "GT": GT.contiguous(),
            "Tlen": int(C.shape[0]),
            "sent_ids": sent_ids,
            "sent_mask": sent_mask,
            "text_tokens": text_words(meta),
        }


def collate_jrt(batch: list[dict]) -> dict:
    B = len(batch)
    Tmax = max(b["Tlen"] for b in batch)
    C = torch.zeros(B, Tmax, 178, 3)
    Tt = torch.zeros(B, Tmax, 178, 3)
    GT = torch.zeros(B, Tmax, 178, 3)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        Ti = b["Tlen"]
        C[i, :Ti] = b["C"]
        Tt[i, :Ti] = b["T"]
        GT[i, :Ti] = b["GT"]
        mask[i, :Ti] = True
    return {
        "C": C,
        "T": Tt,
        "GT": GT,
        "mask": mask,
        "sent_ids": torch.stack([b["sent_ids"] for b in batch], dim=0),
        "sent_mask": torch.stack([b["sent_mask"] for b in batch], dim=0),
        "sids": [b["sid"] for b in batch],
        "text_tokens": [b["text_tokens"] for b in batch],
        "Tlen": torch.tensor([b["Tlen"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def sinusoidal_positions(T: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    emb = torch.cat([torch.sin(pos[:, None] * freqs), torch.cos(pos[:, None] * freqs)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class JRTTransport(nn.Module):
    """Per-frame Transformer producing wrist-coupled bounded residuals.

    Optional richer text conditioning: a 2-layer Transformer encoder over the
    gloss sentence (vs the plain mean-pool default).

    Optional **HyperJRT-DMB** mode (`use_hyper_rho=True`): the JEPA-dominance
    threshold rho becomes a runtime input. The network ingests a 4-D rho
    fingerprint (rho, 1-rho, log(rho), 1-2*rho) via a small MLP whose output
    is added to the per-frame token features, and the final tanh-bounded
    residual is multiplied by rho. One trained model exposes a continuous
    family of certified operating points X(C, rho) for rho in [rho_min,rho_max]
    without retraining. The "Pareto curve" becomes a property of the network,
    not an N-run sweep.

    Optional **Region-HyperJRT-DMB** mode (`use_region_hyper_rho=True`): the
    runtime control is a 3-vector (rho_body, rho_hand, rho_face). The network
    conditions on all three budgets and scales each residual head by its own
    region budget, giving a single model separate certified controls for body,
    hand, and face motion.
    """

    def __init__(
        self,
        gloss_vocab: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        sent_max_len: int,
        max_res_body: float,
        max_res_hand: float,
        max_res_face: float,
        use_sentence_transformer: bool = False,
        sent_tx_layers: int = 2,
        sent_tx_heads: int = 4,
        use_hyper_rho: bool = False,
        use_region_hyper_rho: bool = False,
        use_tarp: bool = False,
        tarp_a_max: float = 0.49,
        tarp_max_perp: float = 0.02,
        tarp_a_init: float = 0.30,
        tarp_constant_env: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.sent_max_len = sent_max_len
        self.max_res_body = max_res_body
        self.max_res_hand = max_res_hand
        self.max_res_face = max_res_face
        self.use_sentence_transformer = bool(use_sentence_transformer)
        self.use_hyper_rho = bool(use_hyper_rho)
        self.use_region_hyper_rho = bool(use_region_hyper_rho)
        self.use_tarp = bool(use_tarp)
        self.tarp_a_max = float(tarp_a_max)
        self.tarp_max_perp = float(tarp_max_perp)
        self.tarp_a_init = float(tarp_a_init)
        self.tarp_constant_env = bool(tarp_constant_env)

        self.pose_in = nn.Linear(178 * 3, hidden)
        self.vel_in = nn.Linear(178 * 3, hidden)
        self.gloss_emb = nn.Embedding(gloss_vocab, hidden, padding_idx=0)
        self.sent_pos_emb = nn.Embedding(sent_max_len, hidden)
        self.progress_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if self.use_region_hyper_rho:
            # Fingerprint(rho_body, rho_hand, rho_face) -> conditioning vector.
            self.rho_mlp = nn.Sequential(
                nn.Linear(12, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        elif self.use_hyper_rho:
            # Fingerprint(rho) -> per-clip conditioning vector.
            self.rho_mlp = nn.Sequential(
                nn.Linear(4, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.rho_mlp = None
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
        if self.use_sentence_transformer:
            sent_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=sent_tx_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sent_blocks = nn.TransformerEncoder(sent_layer, num_layers=sent_tx_layers)
            self.sent_norm = nn.LayerNorm(hidden)
        else:
            self.sent_blocks = None
            self.sent_norm = None
        self.body_head = nn.Linear(hidden, 8 * 3)
        self.hand_head = nn.Linear(hidden, 42 * 3)
        self.face_head = nn.Linear(hidden, 128 * 3)
        for h in (self.body_head, self.hand_head, self.face_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        if self.use_tarp:
            # Stroke-envelope head: per-frame region gains (body, hand, face).
            self.env_head = nn.Linear(hidden, 3)
            nn.init.zeros_(self.env_head.weight)
            # bias so initial sigmoid maps to tarp_a_init / tarp_a_max.
            frac = min(max(self.tarp_a_init / max(self.tarp_a_max, 1e-6), 1e-3), 0.999)
            nn.init.constant_(self.env_head.bias, math.log(frac / (1.0 - frac)))

    def encode_sentence(self, sent_ids: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        S = sent_ids.shape[1]
        device = sent_ids.device
        sp = torch.arange(S, device=device)
        h = self.gloss_emb(sent_ids) + self.sent_pos_emb(sp)[None]
        if self.sent_blocks is not None:
            h = self.sent_norm(self.sent_blocks(h, src_key_padding_mask=~sent_mask.bool()))
        denom = sent_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (h * sent_mask.float().unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        C: torch.Tensor,
        mask: torch.Tensor,
        sent_ids: torch.Tensor,
        sent_mask: torch.Tensor,
        rho_in: torch.Tensor | None = None,
        T_pose: torch.Tensor | None = None,
    ) -> dict:
        B, T = C.shape[:2]
        device = C.device
        x = C.reshape(B, T, -1)
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        h = self.pose_in(x) + self.vel_in(v)
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack(
            [p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1
        )
        h = h + self.progress_mlp(prog)[None]
        h = h + sinusoidal_positions(T, self.hidden, device)[None]
        ctx = self.encode_sentence(sent_ids, sent_mask)
        h = h + ctx[:, None]
        if self.use_tarp:
            z = self.norm(self.blocks(h, src_key_padding_mask=~mask))
            return self._tarp_forward(z, C, T_pose)
        if self.rho_mlp is not None:
            if self.use_region_hyper_rho:
                if rho_in is None:
                    rho_in = torch.full((B, 3), 0.25, device=device)
                if rho_in.ndim == 1:
                    rho_in = rho_in[:, None].expand(-1, 3)
                r = rho_in[:, :3].clamp(1e-3, 0.99)
                fp = torch.cat([r, 1.0 - r, torch.log(r), 1.0 - 2.0 * r], dim=-1)
            else:
                if rho_in is None:
                    rho_in = torch.full((B,), 0.25, device=device)
                r = rho_in.clamp(1e-3, 0.99).reshape(B, 1)
                fp = torch.cat([r, 1.0 - r, torch.log(r), 1.0 - 2.0 * r], dim=-1)
            h = h + self.rho_mlp(fp)[:, None]
        z = self.norm(self.blocks(h, src_key_padding_mask=~mask))

        if self.use_region_hyper_rho and rho_in is not None:
            if rho_in.ndim == 1:
                rho_in = rho_in[:, None].expand(-1, 3)
            scales = rho_in[:, :3].clamp(1e-3, 0.99).reshape(B, 1, 3, 1, 1)
            body_scale = scales[:, :, 0]
            hand_scale = scales[:, :, 1]
            face_scale = scales[:, :, 2]
        elif self.rho_mlp is not None and rho_in is not None:
            scale = rho_in.reshape(B, 1, 1, 1).clamp(1e-3, 0.99)
            body_scale = hand_scale = face_scale = scale
        else:
            body_scale = hand_scale = face_scale = 1.0
        db = torch.tanh(self.body_head(z)).reshape(B, T, 8, 3) * self.max_res_body * body_scale
        dh = torch.tanh(self.hand_head(z)).reshape(B, T, 42, 3) * self.max_res_hand * hand_scale
        df = torch.tanh(self.face_head(z)).reshape(B, T, 128, 3) * self.max_res_face * face_scale

        body_new = C[:, :, BODY] + db
        rh_local = C[:, :, RH] - C[:, :, RWRIST : RWRIST + 1]
        rh_new = body_new[:, :, RWRIST : RWRIST + 1] + rh_local + dh[:, :, 0:21]
        lh_local = C[:, :, LH] - C[:, :, LWRIST : LWRIST + 1]
        lh_new = body_new[:, :, LWRIST : LWRIST + 1] + lh_local + dh[:, :, 21:42]
        face_new = C[:, :, FACE] + df
        X = torch.cat([body_new, rh_new, lh_new, face_new], dim=2)
        R = X - C
        return {"X": X, "R": R, "db": db, "dh": dh, "df": df}

    def _tarp_forward(self, z: torch.Tensor, C: torch.Tensor, T_pose: torch.Tensor) -> dict:
        """Teacher-Aligned Ray Projection with Stroke Envelope.

        R = a_region(t) * D_region  +  e_perp   (e_perp orthogonal to D per
        region/frame). D = T_pose - C. a in [0, a_max] (smooth, nonneg) so the
        certificate rho_region ~= a_region <= a_max < 0.5 by construction.
        """
        if T_pose is None:
            raise ValueError("TARP mode requires T_pose (teacher direction).")
        B, T = C.shape[:2]
        eps = 1e-8
        D = T_pose - C                                          # [B,T,178,3]
        env = self.tarp_a_max * torch.sigmoid(self.env_head(z))  # [B,T,3] in [0,a_max]
        if getattr(self, "tarp_constant_env", False):
            # Ablation: remove the stroke (time) routing -> single per-region gain.
            env = env.mean(dim=1, keepdim=True).expand(-1, env.shape[1], -1)

        e_body = torch.tanh(self.body_head(z)).reshape(B, T, 8, 3) * self.tarp_max_perp
        e_hand = torch.tanh(self.hand_head(z)).reshape(B, T, 42, 3) * self.tarp_max_perp
        e_face = torch.tanh(self.face_head(z)).reshape(B, T, 128, 3) * self.tarp_max_perp
        e = torch.cat([e_body, e_hand, e_face], dim=2)          # [B,T,178,3]

        R_par = torch.zeros_like(D)
        R_perp = torch.zeros_like(D)
        regions = [(BODY, 0), (HAND, 1), (FACE, 2)]
        for sl, gi in regions:
            Dr = D[:, :, sl]                                    # [B,T,J,3]
            er = e[:, :, sl]
            ar = env[:, :, gi].reshape(B, T, 1, 1)
            R_par[:, :, sl] = ar * Dr
            num = (er * Dr).sum(dim=(2, 3), keepdim=True)
            den = (Dr * Dr).sum(dim=(2, 3), keepdim=True).clamp_min(eps)
            R_perp[:, :, sl] = er - (num / den) * Dr            # orthogonalise off-ray
        R = R_par + R_perp
        X = C + R
        return {"X": X, "R": R, "R_par": R_par, "R_perp": R_perp, "env": env,
                "db": R_par[:, :, BODY], "dh": R_par[:, :, HAND], "df": R_par[:, :, FACE]}


# ---------------------------------------------------------------------------
# Losses + dominance certificate
# ---------------------------------------------------------------------------


def per_clip_frobenius(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """||x||_F per clip, accounting for the time mask."""
    m = mask.float().unsqueeze(-1).unsqueeze(-1)
    return (x * m).pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-12).sqrt()


def hand_bones_lengths(pose: torch.Tensor) -> torch.Tensor:
    rh = pose[:, :, RH]
    lh = pose[:, :, LH]
    parts = []
    for (i, j) in HAND_BONES:
        parts.append((rh[:, :, j] - rh[:, :, i]).norm(dim=-1))
        parts.append((lh[:, :, j] - lh[:, :, i]).norm(dim=-1))
    return torch.stack(parts, dim=-1)


def wrist_total_distance(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of frame-to-frame wrist displacements per clip."""
    v = pose[:, 1:, [RWRIST, LWRIST]] - pose[:, :-1, [RWRIST, LWRIST]]
    nv = v.norm(dim=-1).mean(dim=-1)
    m = (mask[:, 1:] & mask[:, :-1]).float()
    return (nv * m).sum(dim=1)


def hand_speed_mass(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-clip mean hand-joint velocity magnitude (unnormalised energy mass)."""
    v = pose[:, 1:, HAND] - pose[:, :-1, HAND]            # [B, T-1, 42, 3]
    sp = v.norm(dim=-1).mean(dim=-1)                      # [B, T-1]
    m = (mask[:, 1:] & mask[:, :-1]).float()
    return (sp * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def hand_speed_frames(pose: torch.Tensor) -> torch.Tensor:
    """Per-frame hand speed [B, T-1] (for percentile matching)."""
    v = pose[:, 1:, HAND] - pose[:, :-1, HAND]
    return v.norm(dim=-1).mean(dim=-1)


class ELARecogEmbedLoss(nn.Module):
    """Clean lexical-grounding loss: align the generated pose's frozen
    pose->gloss recognizer embedding to the GT pose's embedding (frame-masked
    cosine + MSE). The recognizer is independent train-only (NOT the official
    evaluator). Honestly: this makes ELA-JRT recognizer-assisted, not
    pose-only. Embedding alignment (not the recognizer's CE/decision) keeps it
    a representational target rather than a recognizer exploit.
    """

    def __init__(self, recog_ckpt: Path, device: torch.device):
        super().__init__()
        from scripts.train_pose_gloss_ctc import load_frozen  # noqa: E402
        self.recog = load_frozen(recog_ckpt, device)

    def forward(self, X: torch.Tensor, GT: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hx = self.recog.encode(X, mask)
        with torch.no_grad():
            hg = self.recog.encode(GT, mask)
        m = mask.float().unsqueeze(-1)
        mse = ((hx - hg).pow(2) * m).sum() / (m.sum() * hx.shape[-1] + 1e-8)
        cos = 1.0 - (F.cosine_similarity(hx, hg, dim=-1) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return mse + cos


class OfficialBTLexicalLoss(nn.Module):
    """Frozen SLRTP back-translator loss used as lexical coverage signal.

    The model is the same public PHIX back-translator used by the official
    harness. We run it in teacher-forced mode against organizer text, freeze
    every recognizer parameter, and by default compare X against the frozen
    JEPA carrier's own token CE. This makes the lexical term a conservative
    coverage-improvement constraint instead of a free recognizer exploit.
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        max_txt_len: int = 64,
        anchor_mode: str = "carrier_margin",
        margin: float = 0.0,
        clip: float = 1.0,
        eos_weight: float = 0.25,
    ):
        super().__init__()
        try:
            from back_translation.back_translate import EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, UNK_TOKEN, make_back_translation_model
        except Exception as exc:  # pragma: no cover - import guard for CLI clarity.
            raise RuntimeError(f"could not import SLRTP back-translator: {exc}") from exc
        self.bt = make_back_translation_model(model_dir)
        self.bt.to(device)
        self.bt.eval()
        self.bt.do_translation = True
        self.bt.do_recognition = False
        for p in self.bt.parameters():
            p.requires_grad_(False)
        self.max_txt_len = int(max_txt_len)
        self.bos_id = int(self.bt.txt_vocab.stoi[BOS_TOKEN])
        self.eos_id = int(self.bt.txt_vocab.stoi[EOS_TOKEN])
        self.pad_id = int(self.bt.txt_vocab.stoi[PAD_TOKEN])
        self.unk_id = int(self.bt.txt_vocab.stoi[UNK_TOKEN])
        self.device = device
        if anchor_mode not in {"none", "carrier_margin"}:
            raise ValueError(f"bad lexical anchor_mode={anchor_mode}")
        self.anchor_mode = anchor_mode
        self.margin = float(margin)
        self.clip = float(clip)
        self.eos_weight = float(eos_weight)

    def _encode_text(self, token_lists: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = min(
            self.max_txt_len,
            max((len(toks) + 1 for toks in token_lists), default=1),
        )
        B = len(token_lists)
        txt_in = torch.full((B, max_len), self.pad_id, dtype=torch.long, device=self.device)
        txt_tgt = torch.full((B, max_len), self.pad_id, dtype=torch.long, device=self.device)
        txt_mask = torch.zeros((B, 1, max_len), dtype=torch.bool, device=self.device)
        txt_weight = torch.zeros((B, max_len), dtype=torch.float32, device=self.device)
        for i, toks in enumerate(token_lists):
            ids = [int(self.bt.txt_vocab.stoi.get(t, self.unk_id)) for t in toks[: max_len - 1]]
            inp = [self.bos_id] + ids
            tgt = ids + [self.eos_id]
            L = min(len(inp), max_len)
            txt_in[i, :L] = torch.tensor(inp[:L], dtype=torch.long, device=self.device)
            txt_tgt[i, :L] = torch.tensor(tgt[:L], dtype=torch.long, device=self.device)
            txt_mask[i, 0, :L] = True
            txt_weight[i, :L] = 1.0
            if L > 0:
                txt_weight[i, L - 1] = self.eos_weight
        return txt_in, txt_tgt, txt_mask, txt_weight

    def _sample_ce(
        self,
        pose: torch.Tensor,
        mask: torch.Tensor,
        txt_in: torch.Tensor,
        txt_tgt: torch.Tensor,
        txt_mask: torch.Tensor,
        txt_weight: torch.Tensor,
    ) -> torch.Tensor:
        B, T = pose.shape[:2]
        sgn = pose.reshape(B, T, -1)
        sgn_mask = mask[:, None, :].bool()
        sgn_lengths = mask.long().sum(dim=1).to(torch.float32)
        decoder_outputs, _ = self.bt.forward(
            sgn=sgn,
            sgn_mask=sgn_mask,
            sgn_lengths=sgn_lengths,
            txt_input=txt_in,
            txt_mask=txt_mask,
        )
        word_logits, _, _, _ = decoder_outputs
        flat_ce = F.cross_entropy(
            word_logits.reshape(-1, word_logits.shape[-1]),
            txt_tgt.reshape(-1),
            ignore_index=self.pad_id,
            reduction="none",
        ).reshape(B, -1)
        denom = txt_weight.sum(dim=1).clamp_min(1.0)
        return (flat_ce * txt_weight).sum(dim=1) / denom

    def forward(
        self,
        pose: torch.Tensor,
        mask: torch.Tensor,
        token_lists: list[list[str]],
        anchor_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        txt_in, txt_tgt, txt_mask, txt_weight = self._encode_text(token_lists)
        pose_ce = self._sample_ce(pose, mask, txt_in, txt_tgt, txt_mask, txt_weight)
        if self.anchor_mode == "none":
            loss = pose_ce
        else:
            if anchor_pose is None:
                raise ValueError("anchor_pose is required for carrier_margin lexical loss")
            with torch.no_grad():
                anchor_ce = self._sample_ce(anchor_pose.detach(), mask, txt_in, txt_tgt, txt_mask, txt_weight)
            loss = F.relu(pose_ce - anchor_ce + self.margin)
        if self.clip > 0.0:
            loss = loss.clamp_max(self.clip)
        return loss.mean()


class CleanUnitCoverageLoss(nn.Module):
    """LA-JRT clean lexical-coverage loss (no official evaluator).

    Couples a frozen, independently-trained text->motion-unit predictor with a
    differentiable soft assignment of the GENERATED pose to the discovered
    motion-unit codebook. The loss penalises *under-expression*: units the text
    is expected to require but the generated pose does not cover.

      p_text = softmax(text_to_unit(text))           # frozen clean teacher
      p_gen  = soft_unit_bag(X, codebook, temp)       # differentiable in X
      L      = sum_k relu(p_text - p_gen)              # coverage deficit

    Region selection ("hands"/"body"/"all") is recorded for provenance; the
    discovered codebook is hand-motion-defined, so "hands" is the operative
    semantics and other values fall back to the same hand-unit space.
    """

    def __init__(self, codebook_path: Path, teacher_ckpt: Path, device: torch.device,
                 temp: float, region: str):
        super().__init__()
        from scripts.motion_units import soft_unit_bag  # noqa: E402
        from scripts.train_text_to_unit import TextToUnit  # noqa: E402
        self._soft_unit_bag = soft_unit_bag
        self.region = region
        cb = torch.load(codebook_path, map_location="cpu", weights_only=False)
        self.register_buffer("centroids", cb["centroids"].to(device))
        self.register_buffer("feat_mean", cb["feat_mean"].to(device))
        self.register_buffer("feat_std", cb["feat_std"].to(device))
        self.temp = float(temp) if temp > 0 else float(cb.get("temp", 1.0))
        self.K = int(cb["K"])
        tk = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        self.vocab = tk["vocab"]
        self.max_len = int(tk["max_len"])
        self.teacher = TextToUnit(len(self.vocab), tk["hidden"], tk["layers"],
                                  tk["heads"], 0.0, tk["K"], tk["max_len"]).to(device)
        self.teacher.load_state_dict(tk["model"])
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if int(tk["K"]) != self.K:
            raise SystemExit(f"codebook K={self.K} != teacher K={tk['K']}")

    @torch.no_grad()
    def _text_target(self, token_lists: list[list[str]], device) -> torch.Tensor:
        B = len(token_lists)
        ids = torch.zeros(B, self.max_len, dtype=torch.long, device=device)
        msk = torch.zeros(B, self.max_len, dtype=torch.bool, device=device)
        for i, toks in enumerate(token_lists):
            for j, t in enumerate(toks[: self.max_len]):
                ids[i, j] = self.vocab.get(t, 1)
                msk[i, j] = True
            if msk[i].sum() == 0:
                msk[i, 0] = True
        return torch.softmax(self.teacher(ids, msk), dim=-1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor,
                token_lists: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        p_text = self._text_target(token_lists, X.device)
        p_gen = self._soft_unit_bag(X, mask, self.centroids, self.feat_mean,
                                    self.feat_std, self.temp)
        deficit = F.relu(p_text - p_gen).sum(dim=-1).mean()
        # Symmetric coverage diagnostic (not optimised): cosine to text target.
        cov_cos = F.cosine_similarity(p_gen, p_text, dim=-1).mean().detach()
        return deficit, cov_cos


def compute_losses(
    out: dict,
    C: torch.Tensor,
    T_pose: torch.Tensor,
    GT: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    rho_in: torch.Tensor | None = None,
    lex_loss: torch.Tensor | None = None,
    lex_weight: float = 0.0,
    unit_loss: torch.Tensor | None = None,
    unit_weight: float = 0.0,
    unit_cov_cos: float = 0.0,
    recog_loss: torch.Tensor | None = None,
    recog_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    X = out["X"]
    R = out["R"]
    eps = 1e-8
    m = mask.float().unsqueeze(-1).unsqueeze(-1)
    n_frames = m.sum().clamp_min(1.0)

    L_pose = (F.smooth_l1_loss(X, GT, reduction="none") * m).sum() / (n_frames * 178 * 3)

    # In hyper-rho mode, the directional target follows the runtime budget;
    # in region-hyper mode each region receives its own target scale.
    delta_teacher = (T_pose - C) * m
    if rho_in is not None and rho_in.ndim == 2:
        alpha_eff = torch.zeros_like(delta_teacher)
        alpha_eff[:, :, BODY] = rho_in[:, 0].reshape(-1, 1, 1, 1)
        alpha_eff[:, :, HAND] = rho_in[:, 1].reshape(-1, 1, 1, 1)
        alpha_eff[:, :, FACE] = rho_in[:, 2].reshape(-1, 1, 1, 1)
    elif rho_in is not None:
        alpha_eff = rho_in.reshape(-1, 1, 1, 1)
    else:
        alpha_eff = torch.full_like(delta_teacher, float(args.alpha_train))
        alpha_eff = alpha_eff[:, :1, :1, :1]
    R_target = alpha_eff * delta_teacher
    L_dir = (F.smooth_l1_loss(R * m, R_target, reduction="none") * m).sum() / (n_frames * 178 * 3)

    R_norm = per_clip_frobenius(R, mask)
    Tdelta_norm = per_clip_frobenius(delta_teacher, mask)
    rho_clip = R_norm / Tdelta_norm.clamp_min(eps)
    # In scalar hyper mode the threshold is per-sample = rho_in. In region
    # hyper mode the global threshold stays the declared certificate cap.
    if rho_in is not None and rho_in.ndim == 1:
        rho_thr = rho_in.reshape(-1)
    else:
        rho_thr = torch.full_like(rho_clip, float(args.rho))
    L_dom_global = F.relu(rho_clip - rho_thr).pow(2).mean()

    # Per-region dominance: ||R_region|| / ||T_region - C_region|| per clip.
    R_body = (X[:, :, BODY] - C[:, :, BODY]) * m
    R_hand = (X[:, :, HAND] - C[:, :, HAND]) * m
    R_face = (X[:, :, FACE] - C[:, :, FACE]) * m
    T_body = (T_pose[:, :, BODY] - C[:, :, BODY]) * m
    T_hand = (T_pose[:, :, HAND] - C[:, :, HAND]) * m
    T_face = (T_pose[:, :, FACE] - C[:, :, FACE]) * m
    rho_body = per_clip_frobenius(R_body, mask) / per_clip_frobenius(T_body, mask).clamp_min(eps)
    rho_hand = per_clip_frobenius(R_hand, mask) / per_clip_frobenius(T_hand, mask).clamp_min(eps)
    rho_face = per_clip_frobenius(R_face, mask) / per_clip_frobenius(T_face, mask).clamp_min(eps)
    if rho_in is not None and rho_in.ndim == 2:
        rho_body_thr = rho_in[:, 0].reshape(-1)
        rho_hand_thr = rho_in[:, 1].reshape(-1)
        rho_face_thr = rho_in[:, 2].reshape(-1)
    else:
        rho_body_thr = torch.full_like(rho_body, float(args.rho_body))
        rho_hand_thr = torch.full_like(rho_hand, float(args.rho_hand))
        rho_face_thr = torch.full_like(rho_face, float(args.rho_face))
    L_dom_body = F.relu(rho_body - rho_body_thr).pow(2).mean()
    L_dom_hand = F.relu(rho_hand - rho_hand_thr).pow(2).mean()
    L_dom_face = F.relu(rho_face - rho_face_thr).pow(2).mean()
    L_dom = (
        args.lambda_dom * L_dom_global
        + args.lambda_dom_body * L_dom_body
        + args.lambda_dom_hand * L_dom_hand
        + args.lambda_dom_face * L_dom_face
    )

    L_pres_body = (out["db"].pow(2) * m).sum() / (n_frames * 8 * 3)
    L_pres_face = (out["df"].pow(2) * m).sum() / (n_frames * 128 * 3)

    R_acc = R[:, 2:] - 2 * R[:, 1:-1] + R[:, :-2]
    sm = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1).unsqueeze(-1)
    L_smooth = (R_acc.pow(2) * sm).sum() / (sm.sum().clamp_min(1.0) * 178 * 3 + eps)

    bone_x = hand_bones_lengths(X)
    bone_c = hand_bones_lengths(C)
    bm = mask.float().unsqueeze(-1)
    L_bone = ((bone_x - bone_c).pow(2) * bm).sum() / (bm.sum() * bone_x.shape[-1] + eps)

    td_x = wrist_total_distance(X, mask)
    td_g = wrist_total_distance(GT, mask).clamp_min(eps)
    ratio = td_x / td_g
    L_wdist = F.relu((ratio - 1.0).abs() - args.dist_tol).pow(2).mean()

    # --- ELA energy-calibration / anti-shrink (unnormalised, asymmetric) -----
    if args.lambda_energy_floor > 0.0:
        gen_hs = hand_speed_mass(X, mask)
        gt_hs = hand_speed_mass(GT, mask).clamp_min(eps)
        car_hs = hand_speed_mass(C, mask).clamp_min(eps)
        r_gt = gen_hs / gt_hs
        r_car = gen_hs / car_hs
        L_under = F.relu(1.0 - r_gt).pow(2).mean()                       # below GT energy (hard)
        L_over = F.relu(r_gt - args.energy_over_tol).pow(2).mean()       # above GT*tol (mild)
        L_noshrink = F.relu(args.energy_floor_frac - r_car).pow(2).mean()  # below frac*carrier (hard)
        gen_f = hand_speed_frames(X)
        gt_f = hand_speed_frames(GT)
        valid = (mask[:, 1:] & mask[:, :-1])
        gen_burst = gen_f.masked_fill(~valid, float("-inf")).max(dim=1).values
        gt_burst = gt_f.masked_fill(~valid, float("-inf")).max(dim=1).values.clamp_min(eps)
        L_burst = F.relu(1.0 - gen_burst / gt_burst).pow(2).mean()       # under-burst (hard)
        # Total-distance anti-shrink toward 1.0 (asymmetric: under punished more).
        L_td_under = F.relu(1.0 - ratio).pow(2).mean()
        L_energy = (
            L_under
            + args.energy_overshoot_weight * L_over
            + L_noshrink
            + args.energy_burst_weight * L_burst
            + args.energy_td_weight * L_td_under
        )
        energy_logs = {
            "L_energy_under": float(L_under.detach().cpu()),
            "L_energy_over": float(L_over.detach().cpu()),
            "L_energy_noshrink": float(L_noshrink.detach().cpu()),
            "L_energy_burst": float(L_burst.detach().cpu()),
            "gen_hs_over_gt": float(r_gt.mean().detach().cpu()),
            "gen_hs_over_car": float(r_car.mean().detach().cpu()),
        }
    else:
        L_energy = X.new_zeros(())
        energy_logs = {}

    # --- TARP: cone / perp / envelope-smoothness + perp certificate ---------
    tarp_logs = {}
    L_tarp = X.new_zeros(())
    if args.use_tarp and "R_par" in out:
        R_par = out["R_par"]
        R_perp = out["R_perp"]
        env = out["env"]                                          # [B,T,3]
        par_norm = per_clip_frobenius(R_par, mask)
        perp_norm = per_clip_frobenius(R_perp, mask)
        # off-ray must stay a small fraction of the on-ray motion (cone).
        L_cone = F.relu(perp_norm - args.tarp_cone_ratio * par_norm).pow(2).mean()
        # absolute off-ray penalty.
        L_perp = (R_perp.pow(2) * m).sum() / (n_frames * 178 * 3)
        # envelope temporal smoothness (acceleration of the per-region gains).
        env_acc = env[:, 2:] - 2 * env[:, 1:-1] + env[:, :-2]      # [B,T-2,3]
        em = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1)
        L_env_smooth = (env_acc.pow(2) * em).sum() / (em.sum().clamp_min(1.0) * 3 + eps)
        L_env_oracle = X.new_zeros(())
        if args.lambda_env_oracle > 0.0 and T_pose is not None:
            D = T_pose - C
            g = GT - C
            a_star = torch.zeros_like(env)                        # [B,T,3]
            for sl, gi in ((BODY, 0), (HAND, 1), (FACE, 2)):
                num = (g[:, :, sl] * D[:, :, sl]).sum(dim=(2, 3))
                den = (D[:, :, sl] * D[:, :, sl]).sum(dim=(2, 3)).clamp_min(eps)
                a_star[:, :, gi] = (num / den).clamp(0.0, float(args.tarp_a_max))
            # smooth the oracle target over time (moving average).
            k = max(int(args.env_oracle_smooth_kernel), 1)
            if k > 1:
                pad = k // 2
                a_s = F.pad(a_star.transpose(1, 2), (pad, pad), mode="replicate")
                a_star = F.avg_pool1d(a_s, kernel_size=k, stride=1).transpose(1, 2)
            mm = mask.float().unsqueeze(-1)
            L_env_oracle = ((env - a_star).pow(2) * mm).sum() / (mm.sum() * 3 + eps)
        L_tarp = (
            args.lambda_cone * L_cone
            + args.lambda_perp * L_perp
            + args.lambda_env_smooth * L_env_smooth
            + args.lambda_env_oracle * L_env_oracle
        )
        perp_share = (perp_norm / (par_norm + perp_norm).clamp_min(eps))
        tarp_logs = {
            "L_cone": float(L_cone.detach().cpu()),
            "L_perp": float(L_perp.detach().cpu()),
            "L_env_smooth": float(L_env_smooth.detach().cpu()),
            "L_env_oracle": float(L_env_oracle.detach().cpu()),
            "env_body_mean": float(env[..., 0].mean().detach().cpu()),
            "env_hand_mean": float(env[..., 1].mean().detach().cpu()),
            "env_face_mean": float(env[..., 2].mean().detach().cpu()),
            "perp_share_mean": float(perp_share.mean().detach().cpu()),
        }

    if rho_in is not None and rho_in.ndim == 2 and args.lambda_rho_use > 0.0:
        frac = float(args.rho_use_frac)
        L_use_body = F.relu(frac * rho_body_thr - rho_body).pow(2).mean()
        L_use_hand = F.relu(frac * rho_hand_thr - rho_hand).pow(2).mean()
        L_use_face = F.relu(frac * rho_face_thr - rho_face).pow(2).mean()
        L_rho_use = (
            args.rho_use_body_weight * L_use_body
            + args.rho_use_hand_weight * L_use_hand
            + args.rho_use_face_weight * L_use_face
        )
    elif rho_in is not None and rho_in.ndim == 1 and args.lambda_rho_use > 0.0:
        L_use_body = torch.zeros((), device=X.device)
        L_use_hand = torch.zeros((), device=X.device)
        L_use_face = torch.zeros((), device=X.device)
        L_rho_use = F.relu(float(args.rho_use_frac) * rho_thr - rho_clip).pow(2).mean()
    else:
        L_use_body = torch.zeros((), device=X.device)
        L_use_hand = torch.zeros((), device=X.device)
        L_use_face = torch.zeros((), device=X.device)
        L_rho_use = torch.zeros((), device=X.device)

    loss = (
        args.lambda_pose * L_pose
        + args.lambda_dir * L_dir
        + L_dom
        + args.lambda_rho_use * L_rho_use
        + args.lambda_pres_body * L_pres_body
        + args.lambda_pres_face * L_pres_face
        + args.lambda_smooth * L_smooth
        + args.lambda_bone * L_bone
        + args.lambda_wdist * L_wdist
    )
    if lex_loss is not None and lex_weight > 0.0:
        loss = loss + float(lex_weight) * lex_loss
    if unit_loss is not None and unit_weight > 0.0:
        loss = loss + float(unit_weight) * unit_loss
    if args.lambda_energy_floor > 0.0:
        loss = loss + float(args.lambda_energy_floor) * L_energy
    if recog_loss is not None and recog_weight > 0.0:
        loss = loss + float(recog_weight) * recog_loss
    if args.use_tarp:
        loss = loss + L_tarp

    if rho_in is not None:
        rho_pass = (rho_clip < rho_thr).float().mean()
    else:
        rho_pass = (rho_clip < args.rho).float().mean()
    rho_pass_body = (rho_body < rho_body_thr).float().mean()
    rho_pass_hand = (rho_hand < rho_hand_thr).float().mean()
    rho_pass_face = (rho_face < rho_face_thr).float().mean()
    logs = {
        "loss": float(loss.detach().cpu()),
        "L_pose": float(L_pose.detach().cpu()),
        "L_dir": float(L_dir.detach().cpu()),
        "L_dom": float(L_dom.detach().cpu()),
        "L_dom_body": float(L_dom_body.detach().cpu()),
        "L_dom_hand": float(L_dom_hand.detach().cpu()),
        "L_dom_face": float(L_dom_face.detach().cpu()),
        "L_rho_use": float(L_rho_use.detach().cpu()),
        "L_use_body": float(L_use_body.detach().cpu()),
        "L_use_hand": float(L_use_hand.detach().cpu()),
        "L_use_face": float(L_use_face.detach().cpu()),
        "L_pres_body": float(L_pres_body.detach().cpu()),
        "L_pres_face": float(L_pres_face.detach().cpu()),
        "L_smooth": float(L_smooth.detach().cpu()),
        "L_bone": float(L_bone.detach().cpu()),
        "L_wdist": float(L_wdist.detach().cpu()),
        "L_lex": float(lex_loss.detach().cpu()) if lex_loss is not None else 0.0,
        "lex_weight": float(lex_weight),
        "L_unit_cov": float(unit_loss.detach().cpu()) if unit_loss is not None else 0.0,
        "unit_weight": float(unit_weight),
        "unit_cov_cos": float(unit_cov_cos),
        "L_energy": float(L_energy.detach().cpu()),
        "L_recog": float(recog_loss.detach().cpu()) if recog_loss is not None else 0.0,
        "recog_weight": float(recog_weight),
        "L_tarp": float(L_tarp.detach().cpu()),
        **energy_logs,
        **tarp_logs,
        "rho_mean": float(rho_clip.mean().detach().cpu()),
        "rho_p90": float(torch.quantile(rho_clip, 0.9).detach().cpu()),
        "rho_pass": float(rho_pass.detach().cpu()),
        "rho_body_mean": float(rho_body.mean().detach().cpu()),
        "rho_hand_mean": float(rho_hand.mean().detach().cpu()),
        "rho_face_mean": float(rho_face.mean().detach().cpu()),
        "rho_pass_body": float(rho_pass_body.detach().cpu()),
        "rho_pass_hand": float(rho_pass_hand.detach().cpu()),
        "rho_pass_face": float(rho_pass_face.detach().cpu()),
        "td_ratio_mean": float(ratio.mean().detach().cpu()),
        "td_ratio_p10": float(torch.quantile(ratio, 0.1).detach().cpu()),
        "td_ratio_p90": float(torch.quantile(ratio, 0.9).detach().cpu()),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Train driver
# ---------------------------------------------------------------------------


def split_ids(ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rnd = random.Random(seed)
    shuffled = list(ids)
    rnd.shuffle(shuffled)
    n_val = max(16, int(round(val_frac * len(shuffled))))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def lexical_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if float(args.lambda_lex) <= 0.0:
        return 0.0
    warm = max(int(args.lex_warmup_epochs), 0)
    if epoch <= warm:
        return 0.0
    ramp = max(int(args.lex_ramp_epochs), 1)
    return float(args.lambda_lex) * min(1.0, float(epoch - warm) / float(ramp))


def train(args: argparse.Namespace) -> None:
    if args.use_hyper_rho and args.use_region_hyper_rho:
        raise SystemExit("choose only one of --use_hyper_rho or --use_region_hyper_rho")
    if args.use_tarp and (args.use_hyper_rho or args.use_region_hyper_rho):
        raise SystemExit("--use_tarp is exclusive with --use_hyper_rho / --use_region_hyper_rho")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[jrt-dmb] loading carrier {args.carrier_dev_pt}", flush=True)
    carrier = load_pose_map(ROOT / args.carrier_dev_pt)
    print(f"[jrt-dmb] loading teacher {args.teacher_dev_pt}", flush=True)
    teacher = load_pose_map(ROOT / args.teacher_dev_pt)
    print(f"[jrt-dmb] loading reference {args.ref_dev_pt}", flush=True)
    ref = load_ref_meta(ROOT / args.ref_dev_pt)
    common_ids = sorted(set(carrier) & set(teacher) & set(ref))
    print(f"[jrt-dmb] dev intersection ids = {len(common_ids)}", flush=True)
    if not common_ids:
        raise SystemExit("no overlap between carrier/teacher/ref")

    train_ids, val_ids = split_ids(common_ids, args.val_frac, args.seed)
    print(f"[jrt-dmb] inner-train={len(train_ids)} inner-val={len(val_ids)}", flush=True)

    gloss_voc = build_condition_vocab([ref[sid] for sid in train_ids], args.condition_field)
    (out_dir / "gloss_vocab.json").write_text(json.dumps(gloss_voc, indent=2))
    print(
        f"[jrt-dmb] condition_field={args.condition_field} vocab={len(gloss_voc)} (train-only)",
        flush=True,
    )

    train_ds = JRTDataset(train_ids, carrier, teacher, ref, gloss_voc, args.condition_field, args.sent_max_len)
    val_ds = JRTDataset(val_ids, carrier, teacher, ref, gloss_voc, args.condition_field, args.sent_max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_jrt, drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_jrt,
        pin_memory=(device.type == "cuda"),
    )

    model = JRTTransport(
        gloss_vocab=len(gloss_voc),
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        sent_max_len=args.sent_max_len,
        max_res_body=args.max_res_body,
        max_res_hand=args.max_res_hand,
        max_res_face=args.max_res_face,
        use_sentence_transformer=args.use_sentence_transformer,
        sent_tx_layers=args.sent_tx_layers,
        sent_tx_heads=args.sent_tx_heads,
        use_hyper_rho=args.use_hyper_rho,
        use_region_hyper_rho=args.use_region_hyper_rho,
        use_tarp=args.use_tarp,
        tarp_a_max=args.tarp_a_max,
        tarp_max_perp=args.tarp_max_perp,
        tarp_a_init=args.tarp_a_init,
        tarp_constant_env=args.tarp_constant_env,
    ).to(device)
    if args.use_tarp:
        print(
            f"[jrt-dmb] TARP-SE-JRT enabled: a_max={args.tarp_a_max} max_perp={args.tarp_max_perp} "
            f"a_init={args.tarp_a_init} cone_ratio={args.tarp_cone_ratio} "
            f"lambda_cone={args.lambda_cone} lambda_perp={args.lambda_perp} "
            f"lambda_env_smooth={args.lambda_env_smooth} "
            f"(uses teacher direction at inference under JEPA-majority certificate)",
            flush=True,
        )
    print(
        f"[jrt-dmb] device={device} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )
    lex_criterion = None
    if args.lambda_lex > 0.0:
        if not args.allow_evaluator_aware_loss:
            raise SystemExit(
                "lambda_lex uses the official SLRTP back-translator as a train-time loss. "
                "That is evaluator-aware and must not be used for clean benchmark claims. "
                "Pass --allow_evaluator_aware_loss only for diagnostics clearly labeled as such."
            )
        if args.lex_loss_type != "official":
            raise SystemExit(f"unsupported lex_loss_type={args.lex_loss_type}")
        lex_criterion = OfficialBTLexicalLoss(
            ROOT / args.lex_model_dir,
            device,
            args.lex_max_txt_len,
            anchor_mode=args.lex_anchor_mode,
            margin=args.lex_margin,
            clip=args.lex_clip,
            eos_weight=args.lex_eos_weight,
        )
        print(
            f"[jrt-dmb] lexical coverage enabled: type=official "
            f"lambda_lex={args.lambda_lex} warmup={args.lex_warmup_epochs} "
            f"ramp={args.lex_ramp_epochs} anchor={args.lex_anchor_mode} "
            f"margin={args.lex_margin} clip={args.lex_clip} model={args.lex_model_dir}",
            flush=True,
        )
    unit_criterion = None
    if args.lambda_unit_cov > 0.0:
        if not args.unit_codebook_path or not args.unit_teacher_ckpt:
            raise SystemExit("--lambda_unit_cov requires --unit_codebook_path and --unit_teacher_ckpt")
        unit_criterion = CleanUnitCoverageLoss(
            ROOT / args.unit_codebook_path,
            ROOT / args.unit_teacher_ckpt,
            device,
            temp=args.unit_cov_temp,
            region=args.unit_cov_region,
        )
        print(
            f"[jrt-dmb] LA-JRT clean unit coverage enabled: lambda_unit_cov={args.lambda_unit_cov} "
            f"K={unit_criterion.K} temp={unit_criterion.temp} region={args.unit_cov_region} "
            f"codebook={args.unit_codebook_path} teacher={args.unit_teacher_ckpt} "
            f"(NO official evaluator)",
            flush=True,
        )
    recog_criterion = None
    if args.lambda_recog_embed > 0.0:
        if not args.recog_ckpt:
            raise SystemExit("--lambda_recog_embed requires --recog_ckpt")
        recog_criterion = ELARecogEmbedLoss(ROOT / args.recog_ckpt, device)
        print(
            f"[jrt-dmb] ELA recog-embed grounding enabled: lambda_recog_embed={args.lambda_recog_embed} "
            f"recog={args.recog_ckpt} (independent train-only recognizer; recognizer-assisted, "
            f"NOT official evaluator)",
            flush=True,
        )
    if args.lambda_energy_floor > 0.0:
        print(
            f"[jrt-dmb] ELA energy calibration enabled: lambda_energy_floor={args.lambda_energy_floor} "
            f"floor_frac={args.energy_floor_frac} over_tol={args.energy_over_tol} "
            f"overshoot_w={args.energy_overshoot_weight} burst_w={args.energy_burst_weight} "
            f"td_w={args.energy_td_weight}",
            flush=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_acc: dict[str, float] = {}
        n_tr = 0
        for step, batch in enumerate(train_loader, start=1):
            C = batch["C"].to(device)
            T_pose = batch["T"].to(device)
            GT = batch["GT"].to(device)
            mask = batch["mask"].to(device)
            sent_ids = batch["sent_ids"].to(device)
            sent_mask = batch["sent_mask"].to(device)
            if args.use_region_hyper_rho:
                rb = (
                    torch.rand(C.shape[0], device=device)
                    * (args.region_rho_body_max - args.region_rho_body_min)
                    + args.region_rho_body_min
                )
                rh = (
                    torch.rand(C.shape[0], device=device)
                    * (args.region_rho_hand_max - args.region_rho_hand_min)
                    + args.region_rho_hand_min
                )
                rf = (
                    torch.rand(C.shape[0], device=device)
                    * (args.region_rho_face_max - args.region_rho_face_min)
                    + args.region_rho_face_min
                )
                rho_in = torch.stack([rb, rh, rf], dim=1)
            elif args.use_hyper_rho:
                rho_in = (
                    torch.rand(C.shape[0], device=device)
                    * (args.hyper_rho_max - args.hyper_rho_min)
                    + args.hyper_rho_min
                )
            else:
                rho_in = None
            out = model(C, mask, sent_ids, sent_mask, rho_in=rho_in,
                        T_pose=(T_pose if args.use_tarp else None))
            lex_w = lexical_weight_for_epoch(args, ep)
            lex_loss = None
            if lex_criterion is not None and lex_w > 0.0:
                lex_loss = lex_criterion(out["X"], mask, batch["text_tokens"], anchor_pose=C)
            unit_loss = None
            unit_cos = 0.0
            if unit_criterion is not None:
                unit_loss, unit_cos_t = unit_criterion(out["X"], mask, batch["text_tokens"])
                unit_cos = float(unit_cos_t)
            recog_loss = None
            if recog_criterion is not None:
                recog_in = (C + out["R_par"]) if (args.use_tarp and "R_par" in out) else out["X"]
                recog_loss = recog_criterion(recog_in, GT, mask)
            loss, logs = compute_losses(
                out, C, T_pose, GT, mask, args, rho_in=rho_in,
                lex_loss=lex_loss, lex_weight=lex_w,
                unit_loss=unit_loss, unit_weight=args.lambda_unit_cov, unit_cov_cos=unit_cos,
                recog_loss=recog_loss, recog_weight=args.lambda_recog_embed,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in logs.items():
                train_acc[k] = train_acc.get(k, 0.0) + v
            n_tr += 1
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "grad_norm": float(gn), **logs}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        train_log = {f"train_{k}": v / max(n_tr, 1) for k, v in train_acc.items()}

        model.eval()
        val_acc: dict[str, float] = {}
        n_vb = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.val_batches and bi >= args.val_batches:
                    break
                C = batch["C"].to(device)
                T_pose = batch["T"].to(device)
                GT = batch["GT"].to(device)
                mask = batch["mask"].to(device)
                sent_ids = batch["sent_ids"].to(device)
                sent_mask = batch["sent_mask"].to(device)
                if args.use_region_hyper_rho:
                    rho_in = torch.tensor(
                        [
                            0.5 * (args.region_rho_body_min + args.region_rho_body_max),
                            0.5 * (args.region_rho_hand_min + args.region_rho_hand_max),
                            0.5 * (args.region_rho_face_min + args.region_rho_face_max),
                        ],
                        dtype=torch.float32,
                        device=device,
                    )[None].expand(C.shape[0], -1)
                elif args.use_hyper_rho:
                    rho_in = torch.full(
                        (C.shape[0],),
                        0.5 * (args.hyper_rho_min + args.hyper_rho_max),
                        device=device,
                    )
                else:
                    rho_in = None
                out = model(C, mask, sent_ids, sent_mask, rho_in=rho_in,
                            T_pose=(T_pose if args.use_tarp else None))
                lex_w = lexical_weight_for_epoch(args, ep)
                lex_loss = None
                if lex_criterion is not None and lex_w > 0.0:
                    lex_loss = lex_criterion(out["X"], mask, batch["text_tokens"], anchor_pose=C)
                unit_loss = None
                unit_cos = 0.0
                if unit_criterion is not None:
                    unit_loss, unit_cos_t = unit_criterion(out["X"], mask, batch["text_tokens"])
                    unit_cos = float(unit_cos_t)
                recog_loss = None
                if recog_criterion is not None:
                    recog_in = (C + out["R_par"]) if (args.use_tarp and "R_par" in out) else out["X"]
                    recog_loss = recog_criterion(recog_in, GT, mask)
                _, logs = compute_losses(
                    out, C, T_pose, GT, mask, args, rho_in=rho_in,
                    lex_loss=lex_loss, lex_weight=lex_w,
                    unit_loss=unit_loss, unit_weight=args.lambda_unit_cov, unit_cov_cos=unit_cos,
                    recog_loss=recog_loss, recog_weight=args.lambda_recog_embed,
                )
                for k, v in logs.items():
                    val_acc[k] = val_acc.get(k, 0.0) + v
                n_vb += 1
        val_log = {f"val_{k}": v / max(n_vb, 1) for k, v in val_acc.items()}

        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **train_log, **val_log}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)

        score = (
            rec["val_L_pose"]
            + max(0.0, rec["val_rho_mean"] - args.rho) * 5.0
            + 0.05 * (1.0 - rec["val_rho_pass"])
            + 0.05 * (1.0 - rec.get("val_rho_pass_body", 1.0))
            + 0.05 * (1.0 - rec.get("val_rho_pass_hand", 1.0))
            + 0.05 * (1.0 - rec.get("val_rho_pass_face", 1.0))
            + 0.05 * abs(rec["val_td_ratio_mean"] - 1.0)
            + float(args.lex_select_weight) * rec.get("val_L_lex", 0.0)
            - float(args.unit_select_weight) * rec.get("val_unit_cov_cos", 0.0)
            + float(args.recog_select_weight) * rec.get("val_L_recog", 0.0)
        )
        if best is None or score < best["score"]:
            best = {**rec, "score": float(score)}
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best": best,
                    "gloss_to_id": gloss_voc,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[jrt-dmb] saved best -> {out_dir / 'best.pt'} score={score:.4f}", flush=True)

    selected = {
        "alpha_train": args.alpha_train,
        "rho": args.rho,
        "use_hyper_rho": args.use_hyper_rho,
        "use_region_hyper_rho": args.use_region_hyper_rho,
        "hyper_rho_min": args.hyper_rho_min,
        "hyper_rho_max": args.hyper_rho_max,
        "region_rho_body_min": args.region_rho_body_min,
        "region_rho_body_max": args.region_rho_body_max,
        "region_rho_hand_min": args.region_rho_hand_min,
        "region_rho_hand_max": args.region_rho_hand_max,
        "region_rho_face_min": args.region_rho_face_min,
        "region_rho_face_max": args.region_rho_face_max,
        "lambda_rho_use": args.lambda_rho_use,
        "rho_use_frac": args.rho_use_frac,
        "rho_use_body_weight": args.rho_use_body_weight,
        "rho_use_hand_weight": args.rho_use_hand_weight,
        "rho_use_face_weight": args.rho_use_face_weight,
        "lambda_pose": args.lambda_pose,
        "lambda_dir": args.lambda_dir,
        "lambda_dom": args.lambda_dom,
        "lambda_pres_body": args.lambda_pres_body,
        "lambda_pres_face": args.lambda_pres_face,
        "lambda_smooth": args.lambda_smooth,
        "lambda_bone": args.lambda_bone,
        "lambda_wdist": args.lambda_wdist,
        "lambda_unit_cov": args.lambda_unit_cov,
        "unit_codebook_path": args.unit_codebook_path,
        "unit_teacher_ckpt": args.unit_teacher_ckpt,
        "unit_cov_region": args.unit_cov_region,
        "unit_cov_temp": args.unit_cov_temp,
        "lambda_energy_floor": args.lambda_energy_floor,
        "energy_floor_frac": args.energy_floor_frac,
        "energy_over_tol": args.energy_over_tol,
        "energy_overshoot_weight": args.energy_overshoot_weight,
        "energy_burst_weight": args.energy_burst_weight,
        "energy_td_weight": args.energy_td_weight,
        "lambda_recog_embed": args.lambda_recog_embed,
        "recog_ckpt": args.recog_ckpt,
        "recog_select_weight": args.recog_select_weight,
        "use_tarp": args.use_tarp,
        "tarp_a_max": args.tarp_a_max,
        "tarp_max_perp": args.tarp_max_perp,
        "tarp_a_init": args.tarp_a_init,
        "tarp_cone_ratio": args.tarp_cone_ratio,
        "lambda_cone": args.lambda_cone,
        "lambda_perp": args.lambda_perp,
        "lambda_env_smooth": args.lambda_env_smooth,
        "tarp_constant_env": args.tarp_constant_env,
        "lambda_lex": args.lambda_lex,
        "lex_loss_type": args.lex_loss_type,
        "lex_warmup_epochs": args.lex_warmup_epochs,
        "lex_ramp_epochs": args.lex_ramp_epochs,
        "lex_select_weight": args.lex_select_weight,
        "lex_model_dir": args.lex_model_dir,
        "lex_anchor_mode": args.lex_anchor_mode,
        "lex_margin": args.lex_margin,
        "lex_clip": args.lex_clip,
        "lex_eos_weight": args.lex_eos_weight,
        "allow_evaluator_aware_loss": args.allow_evaluator_aware_loss,
        "dist_tol": args.dist_tol,
        "max_res_body": args.max_res_body,
        "max_res_hand": args.max_res_hand,
        "max_res_face": args.max_res_face,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "layers": args.layers,
        "condition_field": args.condition_field,
        "best_epoch": best["epoch"] if best else None,
        "best_val_L_pose": best["val_L_pose"] if best else None,
        "best_val_rho_mean": best["val_rho_mean"] if best else None,
        "best_val_rho_pass": best["val_rho_pass"] if best else None,
        "best_val_td_ratio_mean": best["val_td_ratio_mean"] if best else None,
        "selection_protocol": (
            "Inner-val 20% of dev clip ids, seed=0, never seen by optimizer. "
            "alpha_train and rho are fixed BEFORE any test inference. No test-set "
            "metric was consulted to pick these values."
        ),
    }
    (out_dir / "selected_config.json").write_text(json.dumps(selected, indent=2))
    print(f"[jrt-dmb] selected_config -> {out_dir / 'selected_config.json'}", flush=True)
    print(f"[jrt-dmb] done best={best}", flush=True)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ck["args"])
    gloss_voc = ck["gloss_to_id"]
    model = JRTTransport(
        gloss_vocab=len(gloss_voc),
        hidden=cfg.hidden,
        layers=cfg.layers,
        heads=cfg.heads,
        dropout=0.0,
        sent_max_len=cfg.sent_max_len,
        max_res_body=cfg.max_res_body,
        max_res_hand=cfg.max_res_hand,
        max_res_face=cfg.max_res_face,
        use_sentence_transformer=getattr(cfg, "use_sentence_transformer", False),
        sent_tx_layers=getattr(cfg, "sent_tx_layers", 2),
        sent_tx_heads=getattr(cfg, "sent_tx_heads", 4),
        use_hyper_rho=getattr(cfg, "use_hyper_rho", False),
        use_region_hyper_rho=getattr(cfg, "use_region_hyper_rho", False),
        use_tarp=getattr(cfg, "use_tarp", False),
        tarp_a_max=getattr(cfg, "tarp_a_max", 0.49),
        tarp_max_perp=getattr(cfg, "tarp_max_perp", 0.02),
        tarp_a_init=getattr(cfg, "tarp_a_init", 0.30),
        tarp_constant_env=getattr(cfg, "tarp_constant_env", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    carrier = load_pose_map(ROOT / args.carrier_pt)
    ref = load_ref_meta(ROOT / args.ref_pt)
    use_tarp = getattr(cfg, "use_tarp", False)
    teacher = None
    if use_tarp:
        if not args.teacher_pt:
            raise SystemExit("TARP checkpoint requires --teacher_pt at generation.")
        teacher = load_pose_map(ROOT / args.teacher_pt)
    common_ids = [sid for sid in carrier.keys() if sid in ref and (teacher is None or sid in teacher)]
    sent_max_len = int(cfg.sent_max_len)
    out: dict[str, torch.Tensor] = {}
    n = len(common_ids)
    for i, sid in enumerate(common_ids, start=1):
        C = carrier[sid].float().to(device)
        T = C.shape[0]
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        condition_field = getattr(cfg, "condition_field", "gloss")
        tokens = condition_tokens(ref[sid], condition_field)
        sent_ids = torch.zeros(1, sent_max_len, dtype=torch.long, device=device)
        sent_mask = torch.zeros(1, sent_max_len, dtype=torch.bool, device=device)
        for j, tok in enumerate(tokens[:sent_max_len]):
            sent_ids[0, j] = int(gloss_voc.get(tok, 1))
            sent_mask[0, j] = True
        if sent_mask.sum() == 0:
            sent_mask[0, 0] = True
        rho_in = None
        if getattr(cfg, "use_region_hyper_rho", False):
            rho_in = torch.tensor(
                [[float(args.rho_body), float(args.rho_hand), float(args.rho_face)]],
                dtype=torch.float32,
                device=device,
            )
        elif getattr(cfg, "use_hyper_rho", False):
            rho_in = torch.full((1,), float(args.rho), device=device)
        T_in = None
        if use_tarp:
            Tp = teacher[sid].float().to(device)
            if Tp.shape[0] != T:
                Tp = resize_pose(Tp, T)
            T_in = Tp.unsqueeze(0)
        res = model(C.unsqueeze(0), mask, sent_ids, sent_mask, rho_in=rho_in, T_pose=T_in)
        X = res["X"][0].detach().cpu().contiguous()
        out[sid] = X
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[jrt-dmb-gen] {i}/{n}", flush=True)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}", flush=True)


# ---------------------------------------------------------------------------
# Certify (post-hoc audit)
# ---------------------------------------------------------------------------


def certify(args: argparse.Namespace) -> None:
    carrier = load_pose_map(ROOT / args.carrier_pt)
    pred = load_pose_map(ROOT / args.pred_pt)
    teacher = load_pose_map(ROOT / args.teacher_pt)
    common_ids = [sid for sid in pred.keys() if sid in carrier and sid in teacher]

    per_clip = {}
    rhos = []
    rhos_body, rhos_hand, rhos_face = [], [], []
    out_shares = []
    region_shares = {"body": [], "hand": [], "face": []}
    rho_threshold = float(args.rho)
    rho_b_thr = float(args.rho_body)
    rho_h_thr = float(args.rho_hand)
    rho_f_thr = float(args.rho_face)
    for sid in common_ids:
        C = carrier[sid].float()
        X = pred[sid].float()
        if X.shape[0] != C.shape[0]:
            X = resize_pose(X, C.shape[0])
        T_pose = resize_pose(teacher[sid].float(), C.shape[0])
        R = X - C
        Td = T_pose - C
        r_norm = float(R.pow(2).sum().clamp_min(1e-12).sqrt())
        td_norm = float(Td.pow(2).sum().clamp_min(1e-12).sqrt())
        x_norm = float(X.pow(2).sum().clamp_min(1e-12).sqrt())
        c_norm = float(C.pow(2).sum().clamp_min(1e-12).sqrt())
        rho_v = r_norm / max(td_norm, 1e-12)
        share_in_out = r_norm / max(x_norm, 1e-12)
        share_in_c = r_norm / max(c_norm, 1e-12)
        body_r = float(R[:, BODY].pow(2).sum().clamp_min(1e-12).sqrt())
        hand_r = float(R[:, HAND].pow(2).sum().clamp_min(1e-12).sqrt())
        face_r = float(R[:, FACE].pow(2).sum().clamp_min(1e-12).sqrt())
        body_td = float(Td[:, BODY].pow(2).sum().clamp_min(1e-12).sqrt())
        hand_td = float(Td[:, HAND].pow(2).sum().clamp_min(1e-12).sqrt())
        face_td = float(Td[:, FACE].pow(2).sum().clamp_min(1e-12).sqrt())
        rho_b = body_r / max(body_td, 1e-12)
        rho_h = hand_r / max(hand_td, 1e-12)
        rho_f = face_r / max(face_td, 1e-12)
        rsum = body_r + hand_r + face_r + 1e-12
        per_clip[sid] = {
            "rho_vs_teacher_delta": rho_v,
            "rho_body": rho_b,
            "rho_hand": rho_h,
            "rho_face": rho_f,
            "residual_norm": r_norm,
            "teacher_delta_norm": td_norm,
            "carrier_norm": c_norm,
            "output_norm": x_norm,
            "residual_share_in_output": share_in_out,
            "residual_share_in_carrier": share_in_c,
            "region_share_body": body_r / rsum,
            "region_share_hand": hand_r / rsum,
            "region_share_face": face_r / rsum,
            "pass_rho": rho_v < rho_threshold,
            "pass_rho_body": rho_b < rho_b_thr,
            "pass_rho_hand": rho_h < rho_h_thr,
            "pass_rho_face": rho_f < rho_f_thr,
        }
        rhos.append(rho_v)
        rhos_body.append(rho_b)
        rhos_hand.append(rho_h)
        rhos_face.append(rho_f)
        out_shares.append(share_in_out)
        region_shares["body"].append(body_r / rsum)
        region_shares["hand"].append(hand_r / rsum)
        region_shares["face"].append(face_r / rsum)

    rhos_a = np.asarray(rhos, dtype=np.float64)
    rhos_b = np.asarray(rhos_body, dtype=np.float64)
    rhos_h = np.asarray(rhos_hand, dtype=np.float64)
    rhos_f = np.asarray(rhos_face, dtype=np.float64)
    summary = {
        "rho_threshold": rho_threshold,
        "rho_body_threshold": rho_b_thr,
        "rho_hand_threshold": rho_h_thr,
        "rho_face_threshold": rho_f_thr,
        "clips": len(common_ids),
        "rho_mean": float(rhos_a.mean()),
        "rho_p50": float(np.percentile(rhos_a, 50)),
        "rho_p90": float(np.percentile(rhos_a, 90)),
        "pct_pass_rho": float((rhos_a < rho_threshold).mean()),
        "rho_body_mean": float(rhos_b.mean()),
        "rho_hand_mean": float(rhos_h.mean()),
        "rho_face_mean": float(rhos_f.mean()),
        "pct_pass_rho_body": float((rhos_b < rho_b_thr).mean()),
        "pct_pass_rho_hand": float((rhos_h < rho_h_thr).mean()),
        "pct_pass_rho_face": float((rhos_f < rho_f_thr).mean()),
        "residual_share_in_output_mean": float(np.mean(out_shares)),
        "region_share_body_mean": float(np.mean(region_shares["body"])),
        "region_share_hand_mean": float(np.mean(region_shares["hand"])),
        "region_share_face_mean": float(np.mean(region_shares["face"])),
        "interpretation": (
            "rho_vs_teacher_delta = ||R||_F / ||T - C||_F per clip; below "
            "rho_threshold means the learned residual uses less than that "
            "fraction of the teacher delta magnitude. Region rhos compare "
            "per-region norms (body 0:8, hand 8:50, face 50:178)."
        ),
    }
    out_path = ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"certificate -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--out_dir", default="outputs/sota_chase/jrt_dmb")
    tr.add_argument("--carrier_dev_pt",
                    default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_generator_manifest_dev.pt")
    tr.add_argument("--teacher_dev_pt",
                    default="external/SLRTP-Sign-Production-Evaluation/results/phase58_b40_CAC_a0_00_dev.pt")
    tr.add_argument("--ref_dev_pt",
                    default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--sent_max_len", type=int, default=32)
    tr.add_argument("--condition_field", choices=["text", "gloss", "none"], default="text",
                    help="'text' is the publishable T2P input. 'gloss' is an oracle/diagnostic mode.")
    tr.add_argument("--max_res_body", type=float, default=0.020)
    tr.add_argument("--max_res_hand", type=float, default=0.060)
    tr.add_argument("--max_res_face", type=float, default=0.008)
    tr.add_argument("--alpha_train", type=float, default=0.40,
                    help="FIXED scalar guiding the directional Huber term; do NOT pick by test.")
    tr.add_argument("--rho", type=float, default=0.50,
                    help="Global JEPA-majority dominance threshold on ||R||/||T-C|| per clip.")
    tr.add_argument("--rho_body", type=float, default=0.50,
                    help="Per-region dominance threshold for body joints (0..7).")
    tr.add_argument("--rho_hand", type=float, default=0.50,
                    help="Per-region dominance threshold for hand joints (8..49).")
    tr.add_argument("--rho_face", type=float, default=0.50,
                    help="Per-region dominance threshold for face joints (50..177).")
    tr.add_argument("--use_sentence_transformer", action="store_true",
                    help="Replace mean-pool gloss context with a 2-layer Transformer over the sentence.")
    tr.add_argument("--sent_tx_layers", type=int, default=2)
    tr.add_argument("--sent_tx_heads", type=int, default=4)
    tr.add_argument("--use_hyper_rho", action="store_true",
                    help="HyperJRT-DMB: condition the model on rho as a runtime input; "
                         "sample rho per-clip during training from U[hyper_rho_min, hyper_rho_max]. "
                         "One trained model exposes a continuous family X(C, rho).")
    tr.add_argument("--hyper_rho_min", type=float, default=0.05)
    tr.add_argument("--hyper_rho_max", type=float, default=0.50)
    tr.add_argument("--use_region_hyper_rho", action="store_true",
                    help="Region-HyperJRT-DMB: condition on runtime "
                         "(rho_body, rho_hand, rho_face) and scale each residual head separately.")
    tr.add_argument("--region_rho_body_min", type=float, default=0.03)
    tr.add_argument("--region_rho_body_max", type=float, default=0.20)
    tr.add_argument("--region_rho_hand_min", type=float, default=0.10)
    tr.add_argument("--region_rho_hand_max", type=float, default=0.50)
    tr.add_argument("--region_rho_face_min", type=float, default=0.005)
    tr.add_argument("--region_rho_face_max", type=float, default=0.08)
    tr.add_argument("--lambda_unit_cov", type=float, default=0.0,
                    help="LA-JRT clean lexical coverage weight. Penalises motion-unit "
                         "under-expression vs a frozen text->unit teacher. No official evaluator.")
    tr.add_argument("--unit_codebook_path", type=str, default="",
                    help="Path to motion-unit codebook.pt from extract_motion_units.py.")
    tr.add_argument("--unit_teacher_ckpt", type=str, default="",
                    help="Path to frozen text->unit predictor best.pt from train_text_to_unit.py.")
    tr.add_argument("--unit_cov_region", choices=["hands", "body", "all"], default="hands",
                    help="Region provenance for coverage; codebook is hand-motion-defined.")
    tr.add_argument("--unit_cov_temp", type=float, default=0.0,
                    help="Soft-assignment temperature (0 = use codebook's stored temp).")
    tr.add_argument("--unit_select_weight", type=float, default=0.10,
                    help="Clean checkpoint-selection reward for val unit-coverage cosine "
                         "(rewards higher coverage subject to certificate; no official evaluator).")
    # --- ELA-JRT: energy calibration / anti-shrink (unnormalised) -----------
    tr.add_argument("--lambda_energy_floor", type=float, default=0.0,
                    help="ELA-JRT energy-calibration weight (anti-shrink hand-speed/burst/TotalDist).")
    tr.add_argument("--energy_floor_frac", type=float, default=1.0,
                    help="Generated hand-speed must reach >= frac * carrier hand-speed (anti-shrink floor).")
    tr.add_argument("--energy_over_tol", type=float, default=1.10,
                    help="Hand-speed up to this multiple of GT is unpenalised; above is mildly penalised.")
    tr.add_argument("--energy_overshoot_weight", type=float, default=0.25,
                    help="Asymmetry: overshoot penalty weight (< under-articulation, which is weight 1).")
    tr.add_argument("--energy_burst_weight", type=float, default=0.5)
    tr.add_argument("--energy_td_weight", type=float, default=1.0,
                    help="Weight on one-sided TotalDist-under-1 penalty inside the energy term.")
    # --- ELA-JRT: independent recognizer-embedding grounding ----------------
    tr.add_argument("--lambda_recog_embed", type=float, default=0.0,
                    help="ELA-JRT recognizer-embedding alignment weight (independent train-only "
                         "pose->gloss CTC recognizer; recognizer-assisted, NOT official evaluator).")
    tr.add_argument("--recog_ckpt", type=str, default="",
                    help="Path to frozen pose->gloss CTC recognizer best.pt.")
    tr.add_argument("--recog_select_weight", type=float, default=0.0,
                    help="Clean checkpoint-selection reward for lower val recog-embed loss.")
    # --- TARP-SE-JRT: teacher-aligned ray projection with stroke envelope ---
    tr.add_argument("--use_tarp", action="store_true",
                    help="TARP-SE-JRT: R = a_region(t)*(T-C) + e_perp. Learns a smooth nonneg "
                         "stroke envelope on the teacher ray plus a bounded orthogonal residual. "
                         "Uses teacher direction at inference under a JEPA-majority certificate.")
    tr.add_argument("--tarp_a_max", type=float, default=0.49,
                    help="Max envelope gain (== max certificate rho along the ray; <0.5 keeps JEPA-majority).")
    tr.add_argument("--tarp_max_perp", type=float, default=0.02,
                    help="Bound on the per-axis off-ray (orthogonal) residual.")
    tr.add_argument("--tarp_a_init", type=float, default=0.30,
                    help="Initial envelope gain (bias init).")
    tr.add_argument("--lambda_cone", type=float, default=1.0,
                    help="Penalty when off-ray norm exceeds tarp_cone_ratio * on-ray norm.")
    tr.add_argument("--tarp_cone_ratio", type=float, default=0.25,
                    help="Allowed ratio ||R_perp|| / ||R_parallel|| before L_cone fires.")
    tr.add_argument("--lambda_perp", type=float, default=5.0,
                    help="Absolute off-ray residual penalty.")
    tr.add_argument("--lambda_env_smooth", type=float, default=10.0,
                    help="Stroke-envelope temporal smoothness (acceleration) penalty.")
    tr.add_argument("--lambda_env_oracle", type=float, default=0.0,
                    help="TARP-SE v3: distil the envelope toward the clean oracle ray projection "
                         "a*=clamp(<GT-C,D>/||D||^2, 0, a_max) from train GT (no official evaluator).")
    tr.add_argument("--env_oracle_smooth_kernel", type=int, default=5,
                    help="Temporal moving-average kernel applied to the oracle envelope target.")
    tr.add_argument("--tarp_constant_env", action="store_true",
                    help="Ablation: time-average the envelope to a single per-region gain (no stroke routing).")
    tr.add_argument("--dist_tol", type=float, default=0.10)
    tr.add_argument("--lambda_pose", type=float, default=1.0)
    tr.add_argument("--lambda_dir", type=float, default=0.5)
    tr.add_argument("--lambda_dom", type=float, default=4.0)
    tr.add_argument("--lambda_dom_body", type=float, default=0.0)
    tr.add_argument("--lambda_dom_hand", type=float, default=0.0)
    tr.add_argument("--lambda_dom_face", type=float, default=0.0)
    tr.add_argument("--lambda_rho_use", type=float, default=0.0,
                    help="One-sided budget-utilization loss for hyper-rho modes.")
    tr.add_argument("--rho_use_frac", type=float, default=0.5,
                    help="Require observed rho to reach at least this fraction of runtime rho.")
    tr.add_argument("--rho_use_body_weight", type=float, default=1.0)
    tr.add_argument("--rho_use_hand_weight", type=float, default=2.0)
    tr.add_argument("--rho_use_face_weight", type=float, default=0.2)
    tr.add_argument("--lambda_pres_body", type=float, default=1.0)
    tr.add_argument("--lambda_pres_face", type=float, default=2.0)
    tr.add_argument("--lambda_smooth", type=float, default=100.0)
    tr.add_argument("--lambda_bone", type=float, default=10.0)
    tr.add_argument("--lambda_wdist", type=float, default=1.0)
    tr.add_argument("--lambda_lex", type=float, default=0.0,
                    help="Recognizer-aware lexical coverage weight. Default 0 preserves old JRT-DMB.")
    tr.add_argument("--allow_evaluator_aware_loss", action="store_true",
                    help="Required to train with lambda_lex. This uses the official SLRTP back-translator "
                         "inside the loss and is diagnostic/evaluator-aware, not a clean benchmark setting.")
    tr.add_argument("--lex_loss_type", choices=["official"], default="official",
                    help="Frozen lexical critic. 'official' uses the public SLRTP PHIX back-translator CE.")
    tr.add_argument("--lex_model_dir",
                    default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model")
    tr.add_argument("--lex_max_txt_len", type=int, default=64)
    tr.add_argument("--lex_anchor_mode", choices=["carrier_margin", "none"], default="carrier_margin",
                    help="carrier_margin penalizes only text CE that is worse than the frozen JEPA carrier.")
    tr.add_argument("--lex_margin", type=float, default=0.0,
                    help="Allowed CE margin above the carrier before lexical loss is charged.")
    tr.add_argument("--lex_clip", type=float, default=1.0,
                    help="Per-sample lexical penalty cap; <=0 disables clipping.")
    tr.add_argument("--lex_eos_weight", type=float, default=0.25,
                    help="Down-weight EOS so lexical coverage favors content tokens over early stopping.")
    tr.add_argument("--lex_warmup_epochs", type=int, default=3,
                    help="Epochs before applying lexical loss.")
    tr.add_argument("--lex_ramp_epochs", type=int, default=4,
                    help="Linear ramp length after warmup for lambda_lex.")
    tr.add_argument("--lex_select_weight", type=float, default=0.0,
                    help="Optional dev-only lexical CE term in checkpoint selection score.")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--eval_batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--val_frac", type=float, default=0.20)
    tr.add_argument("--val_batches", type=int, default=0)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--max_steps", type=int, default=0)

    gn = sub.add_parser("generate")
    gn.add_argument("--ckpt", required=True)
    gn.add_argument("--carrier_pt", required=True)
    gn.add_argument("--ref_pt", required=True,
                    help="Organizer GT .pt (used ONLY for gloss-side conditioning, not for pose targets).")
    gn.add_argument("--teacher_pt", type=str, default="",
                    help="Teacher pose .pt (required for TARP checkpoints; provides the ray D=T-C).")
    gn.add_argument("--out_pt", required=True)
    gn.add_argument("--rho", type=float, default=0.30,
                    help="Inference-time rho for HyperJRT-DMB checkpoints. "
                         "Ignored for non-hyper checkpoints.")
    gn.add_argument("--rho_body", type=float, default=0.10,
                    help="Inference-time body rho for Region-HyperJRT-DMB checkpoints.")
    gn.add_argument("--rho_hand", type=float, default=0.45,
                    help="Inference-time hand rho for Region-HyperJRT-DMB checkpoints.")
    gn.add_argument("--rho_face", type=float, default=0.03,
                    help="Inference-time face rho for Region-HyperJRT-DMB checkpoints.")
    gn.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    gn.add_argument("--log_every", type=int, default=200)

    cr = sub.add_parser("certify")
    cr.add_argument("--carrier_pt", required=True)
    cr.add_argument("--teacher_pt", required=True)
    cr.add_argument("--pred_pt", required=True)
    cr.add_argument("--out_json", required=True)
    cr.add_argument("--rho", type=float, default=0.50)
    cr.add_argument("--rho_body", type=float, default=0.50)
    cr.add_argument("--rho_hand", type=float, default=0.50)
    cr.add_argument("--rho_face", type=float, default=0.50)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    else:
        certify(args)


if __name__ == "__main__":
    main()
