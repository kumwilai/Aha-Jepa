"""v22 = v21 velocity-field parameterisation + per-frame velocity-direction
supervision against linearly-resampled GT (v22).

Motivation
----------
v21 confirmed that a bounded velocity-delta field with cumsum integration and
endpoint+zero-mean drift removal DOES add real intra-utterance arc length
(arc_gain 1.07-1.19, hand_pose_std → 0.99, no smoothing collapse) but at a
substantial BLEU cost: the added energy is not semantically aligned because
supervision was aggregate-only (speed band + jerk cap + arc-gain + geometry).

v22 keeps the v21 parameterisation exactly and adds a single supervision
term: per-frame cosine velocity-direction loss between predicted hand/wrist
velocity and the corresponding GT hand/wrist velocity after linearly
resampling GT to the source clip length. The cosine signal is translation
and scale invariant — it only injects directional information — so it cannot
collapse the refiner into a pure source-of-GT regressor and it does not
require the signer's absolute pose coordinates to match.

Constraints satisfied
---------------------
- frozen v11+governor source pose at inference
- no GT length at inference (source comes from rebuilt text-only duration
  policy)
- no retrieval / no real-clip splicing / no train-corpus stats at inference
- GT poses are used only at TRAINING time, not at inference. They never enter
  the inference path.
- no SLRTP back-translation evaluator distillation.

Reuses
------
- VelocityFieldRefiner (exact v21 model)
- amp_clip source augmentation (so train/inference src_speed distribution
  match is preserved)
- length-bucket + sentence-gloss-mean conditioning

Adds
----
- AlignedResidualFlowDataset: same as v21's ResidualFlowDataset but also
  yields a `gt_aligned` tensor (GT linearly resampled to the (possibly
  cropped) source length).
- L_vel_dir: per-frame, per-joint cosine direction loss on hand and wrist
  velocity, masked by GT velocity magnitude to emphasise frames where the
  GT signer's hand is actually moving.

Loss aggregation:

  L_vel_dir = ( sum_{t, j} |v_gt[t, j]| * (1 - cos(v_pred[t, j], v_gt[t, j])) )
              / ( sum_{t, j} |v_gt[t, j]| + eps )

This loss is bounded in [0, 2] per frame-joint, weighted by GT motion energy
so that low-motion / hold frames do not dominate.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_constrained_residual_flow import (  # noqa: E402
    BODY,
    RH,
    LH,
    FACE,
    HAND,
    RWRIST,
    LWRIST,
    HAND_BONES,
    hand_speed_scalar,
    hand_jerk_scalar,
    load_pose_map,
    hand_bones_lengths,
    masked_hand_speed,
    masked_hand_jerk,
)
from scripts.sign_jepa_motion_amplify import amplify_clip  # noqa: E402
from scripts.train_sign_jepa_velocity_field_refiner import (  # noqa: E402
    VelocityFieldRefiner,
    masked_arc_length,
)


# ---------------------------------------------------------------------------
# Aligned dataset (GT linearly resampled to source length)
# ---------------------------------------------------------------------------


def linear_resample_pose(pose: torch.Tensor, T_target: int) -> torch.Tensor:
    """Linearly resample a [T, 178, 3] pose tensor along the time axis."""
    T_src = pose.shape[0]
    if T_src == T_target:
        return pose
    if T_src <= 1:
        return pose.repeat(T_target, 1, 1)
    # Use interpolate on a [1, 1, T_src, F] image.
    flat = pose.reshape(T_src, -1).t().unsqueeze(0)        # [1, F, T_src]
    out = F.interpolate(flat, size=T_target, mode="linear", align_corners=True)
    return out.squeeze(0).t().reshape(T_target, 178, 3).contiguous()


class AlignedResidualFlowDataset(Dataset):
    """Like ResidualFlowDataset but also yields linearly-resampled GT."""

    def __init__(
        self,
        ids: list[str],
        src_pose_map: dict[str, torch.Tensor],
        gt_pose_map: dict[str, torch.Tensor],
        manifest_rows: dict[str, dict],
        gloss_to_id: dict[str, int],
        max_t: int,
        max_sent_len: int,
        is_train: bool,
        crop_p: float,
        source_amp_aug: bool,
        source_amp_min: float,
        source_amp_max: float,
        seed: int = 0,
    ):
        self.ids = list(ids)
        self.src_pose_map = src_pose_map
        self.gt_pose_map = gt_pose_map
        self.manifest_rows = manifest_rows
        self.gloss_to_id = gloss_to_id
        self.max_t = int(max_t)
        self.max_sent_len = int(max_sent_len)
        self.is_train = bool(is_train)
        self.crop_p = float(crop_p)
        self.source_amp_aug = bool(source_amp_aug)
        self.source_amp_min = float(source_amp_min)
        self.source_amp_max = float(source_amp_max)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        src = self.src_pose_map[sid].float()                      # [Ts, 178, 3]
        gt_full = self.gt_pose_map[sid].float()                   # [Tg, 178, 3]
        row = self.manifest_rows.get(sid, {"text": "", "gloss": ""})
        sent_glosses = [g for g in str(row.get("gloss", "")).split() if g]
        sent_ids = torch.zeros(self.max_sent_len, dtype=torch.long)
        sent_mask = torch.zeros(self.max_sent_len, dtype=torch.bool)
        for j, g in enumerate(sent_glosses[: self.max_sent_len]):
            sent_ids[j] = int(self.gloss_to_id.get(g, 1))
            sent_mask[j] = True
        if sent_mask.sum() == 0:
            sent_mask[0] = True

        T = src.shape[0]
        if self.is_train and T > self.max_t and self._rng.random() < self.crop_p:
            start = self._rng.randint(0, T - self.max_t)
            src_clip = src[start : start + self.max_t]
            # Map this src crop to the proportional GT crop, then resample.
            T_g = gt_full.shape[0]
            g_start = int(round(start * T_g / max(T, 1)))
            g_end = int(round((start + self.max_t) * T_g / max(T, 1)))
            g_start = max(0, min(T_g - 1, g_start))
            g_end = max(g_start + 1, min(T_g, g_end))
            gt_clip_pre = gt_full[g_start:g_end]
        elif T > self.max_t:
            src_clip = src[: self.max_t]
            g_end = int(round(self.max_t * gt_full.shape[0] / max(T, 1)))
            g_end = max(1, min(gt_full.shape[0], g_end))
            gt_clip_pre = gt_full[:g_end]
        else:
            src_clip = src
            gt_clip_pre = gt_full

        if self.is_train and self.source_amp_aug:
            lo = min(self.source_amp_min, self.source_amp_max)
            hi = max(self.source_amp_min, self.source_amp_max)
            gain = self._rng.uniform(lo, hi)
            src_clip = amplify_clip(
                src_clip,
                g_body=1.0,
                g_hand=gain,
                g_face=1.0,
                lp_kernel=1,
                smooth_kernel=1,
            )

        gt_aligned = linear_resample_pose(gt_clip_pre, src_clip.shape[0])
        gt_speed = hand_speed_scalar(gt_full)
        gt_jerk = hand_jerk_scalar(gt_full)
        return {
            "sid": sid,
            "src": src_clip.contiguous(),
            "gt_aligned": gt_aligned.contiguous(),
            "T": int(src_clip.shape[0]),
            "sent_ids": sent_ids,
            "sent_mask": sent_mask,
            "gt_speed": torch.tensor(gt_speed, dtype=torch.float32),
            "gt_jerk": torch.tensor(gt_jerk, dtype=torch.float32),
        }


def collate_aligned(batch: list[dict]) -> dict:
    B = len(batch)
    T_max = max(b["T"] for b in batch)
    src = torch.zeros(B, T_max, 178, 3)
    gt_aligned = torch.zeros(B, T_max, 178, 3)
    src_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        Ti = b["T"]
        src[i, :Ti] = b["src"]
        gt_aligned[i, :Ti] = b["gt_aligned"]
        src_mask[i, :Ti] = True
    return {
        "src": src,
        "gt_aligned": gt_aligned,
        "src_mask": src_mask,
        "sent_ids": torch.stack([b["sent_ids"] for b in batch], dim=0),
        "sent_mask": torch.stack([b["sent_mask"] for b in batch], dim=0),
        "gt_speed": torch.stack([b["gt_speed"] for b in batch], dim=0),
        "gt_jerk": torch.stack([b["gt_jerk"] for b in batch], dim=0),
        "sids": [b["sid"] for b in batch],
        "T_each": torch.tensor([b["T"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def vel_direction_loss(
    pred: torch.Tensor,
    gt_aligned: torch.Tensor,
    mask: torch.Tensor,
    region_idx: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-frame per-joint cosine direction loss on velocity, weighted by
    GT velocity magnitude so low-motion frames do not dominate.

    pred       : [B, T, 178, 3]
    gt_aligned : [B, T, 178, 3]
    mask       : [B, T] bool
    region_idx : LongTensor of joint indices to compare
    """
    p = pred.index_select(2, region_idx)                             # [B, T, J, 3]
    g = gt_aligned.index_select(2, region_idx)
    vp = p[:, 1:] - p[:, :-1]                                        # [B, T-1, J, 3]
    vg = g[:, 1:] - g[:, :-1]
    n_p = vp.norm(dim=-1).clamp_min(eps)                              # [B, T-1, J]
    n_g = vg.norm(dim=-1).clamp_min(eps)
    cos = (vp * vg).sum(dim=-1) / (n_p * n_g)                         # [B, T-1, J]
    weight_t = (mask[:, 1:] & mask[:, :-1]).float().unsqueeze(-1)     # [B, T-1, 1]
    weight = weight_t * n_g                                           # weight by gt-vel magnitude
    num = ((1.0 - cos) * weight).sum()
    den = weight.sum().clamp_min(eps)
    return num / den


def compute_losses_v22(
    pred: torch.Tensor,
    src: torch.Tensor,
    gt_aligned: torch.Tensor,
    mask: torch.Tensor,
    gt_speed: torch.Tensor,
    gt_jerk: torch.Tensor,
    deltas: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    B = pred.shape[0]
    eps = 1e-6
    mt = mask.float().unsqueeze(-1).unsqueeze(-1)
    n_frames = mt.sum().clamp_min(1.0)

    # --- v21 motion losses ---------------------------------------------------
    pred_speed = masked_hand_speed(pred, mask)
    src_speed = masked_hand_speed(src, mask)
    ratio = pred_speed / gt_speed.clamp_min(eps)
    log_ratio = torch.log((pred_speed + eps) / (gt_speed + eps))
    band_lo = float(args.speed_band_lo)
    band_hi = float(args.speed_band_hi)
    band_lo_t = pred.new_tensor(band_lo)
    band_hi_t = pred.new_tensor(band_hi)
    log_target = pred.new_tensor(math.log(0.5 * (band_lo + band_hi)))
    L_band = F.relu(band_lo_t - ratio).mean() + F.relu(ratio - band_hi_t).mean()
    L_logspeed = F.smooth_l1_loss(log_ratio, log_target.expand_as(log_ratio))
    L_speed = L_band + 0.1 * L_logspeed

    pred_jerk = masked_hand_jerk(pred, mask)
    jerk_ratio = pred_jerk / gt_jerk.clamp_min(eps)
    L_jerk = F.relu(jerk_ratio - float(args.jerk_cap)).mean()

    pred_arc = masked_arc_length(pred, mask)
    src_arc = masked_arc_length(src, mask)
    arc_gain = pred_arc / src_arc.clamp_min(eps)
    L_arc = F.relu(pred.new_tensor(float(args.arc_min_gain)) - arc_gain).mean()

    diff_hand = (pred[:, :, HAND] - src[:, :, HAND]).abs()
    L_dev = (diff_hand * mt).sum() / (n_frames * 42 * 3)

    non_wrist_idx = torch.tensor([0, 1, 3, 4, 6, 7], dtype=torch.long, device=pred.device)
    body_other_pred = pred[:, :, BODY].index_select(2, non_wrist_idx)
    body_other_src = src[:, :, BODY].index_select(2, non_wrist_idx)
    L_body_other = ((body_other_pred - body_other_src).pow(2) * mt).sum() / (
        n_frames * 6 * 3
    )
    wrist_idx = torch.tensor([RWRIST, LWRIST], dtype=torch.long, device=pred.device)
    wrist_pred = pred.index_select(2, wrist_idx)
    wrist_src = src.index_select(2, wrist_idx)
    L_anchor = ((wrist_pred - wrist_src).pow(2) * mt).sum() / (n_frames * 2 * 3)

    diff_face = (pred[:, :, FACE] - src[:, :, FACE]).pow(2)
    L_face = (diff_face * mt).sum() / (n_frames * 128 * 3)

    bl_pred = hand_bones_lengths(pred)
    bl_src = hand_bones_lengths(src)
    bl_mask = mask.float().unsqueeze(-1)
    bone_diff = (bl_pred - bl_src).pow(2)
    L_bone = (bone_diff * bl_mask).sum() / (
        mask.float().sum() * bl_pred.shape[-1] + eps
    )

    dv_body = deltas["dv_body"]
    dv_hand = deltas["dv_hand"]
    if dv_body.shape[1] >= 3:
        a_body = dv_body[:, 2:] - 2 * dv_body[:, 1:-1] + dv_body[:, :-2]
        a_hand = dv_hand[:, 2:] - 2 * dv_hand[:, 1:-1] + dv_hand[:, :-2]
        smask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1).unsqueeze(-1)
        L_vel_smooth = (
            (a_body.pow(2) * smask).sum() / (smask.sum().clamp_min(1.0) * 8 * 3 + eps)
            + (a_hand.pow(2) * smask).sum() / (smask.sum().clamp_min(1.0) * 42 * 3 + eps)
        )
    else:
        L_vel_smooth = pred.new_tensor(0.0)

    # --- v22 direction supervision ------------------------------------------
    hand_idx = torch.arange(8, 50, dtype=torch.long, device=pred.device)
    wrist_idx_full = torch.tensor([RWRIST, LWRIST], dtype=torch.long, device=pred.device)
    L_vel_dir_hand = vel_direction_loss(pred, gt_aligned, mask, hand_idx)
    L_vel_dir_wrist = vel_direction_loss(pred, gt_aligned, mask, wrist_idx_full)
    L_vel_dir = L_vel_dir_hand + 0.5 * L_vel_dir_wrist

    # Diagnostic: also compute the source-to-GT direction cosine baseline, to
    # see how much improvement over identity the refiner achieves.
    with torch.no_grad():
        L_vel_dir_hand_src = vel_direction_loss(src, gt_aligned, mask, hand_idx)
        L_vel_dir_wrist_src = vel_direction_loss(src, gt_aligned, mask, wrist_idx_full)

    loss = (
        args.lambda_speed * L_speed
        + args.lambda_jerk * L_jerk
        + args.lambda_arc * L_arc
        + args.lambda_dev * L_dev
        + args.lambda_body * L_body_other
        + args.lambda_face * L_face
        + args.lambda_anchor * L_anchor
        + args.lambda_bone * L_bone
        + args.lambda_vel_smooth * L_vel_smooth
        + args.lambda_vel_dir * L_vel_dir
    )

    res_hand_mag = (pred[:, :, HAND] - src[:, :, HAND]).norm(dim=-1).mean(dim=-1)
    res_hand_mag = (res_hand_mag * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)

    logs = {
        "loss": float(loss.detach().cpu()),
        "L_speed": float(L_speed.detach().cpu()),
        "L_band": float(L_band.detach().cpu()),
        "L_jerk": float(L_jerk.detach().cpu()),
        "L_arc": float(L_arc.detach().cpu()),
        "L_dev": float(L_dev.detach().cpu()),
        "L_body_other": float(L_body_other.detach().cpu()),
        "L_face": float(L_face.detach().cpu()),
        "L_anchor": float(L_anchor.detach().cpu()),
        "L_bone": float(L_bone.detach().cpu()),
        "L_vel_smooth": float(L_vel_smooth.detach().cpu()),
        "L_vel_dir": float(L_vel_dir.detach().cpu()),
        "L_vel_dir_hand": float(L_vel_dir_hand.detach().cpu()),
        "L_vel_dir_wrist": float(L_vel_dir_wrist.detach().cpu()),
        "L_vel_dir_hand_src": float(L_vel_dir_hand_src.detach().cpu()),
        "L_vel_dir_wrist_src": float(L_vel_dir_wrist_src.detach().cpu()),
        "speed_ratio_mean": float(ratio.mean().detach().cpu()),
        "speed_ratio_p10": float(torch.quantile(ratio, 0.1).detach().cpu()),
        "speed_ratio_p90": float(torch.quantile(ratio, 0.9).detach().cpu()),
        "jerk_ratio_mean": float(jerk_ratio.mean().detach().cpu()),
        "arc_gain_mean": float(arc_gain.mean().detach().cpu()),
        "src_speed_mean": float(
            (src_speed / gt_speed.clamp_min(eps)).mean().detach().cpu()
        ),
        "res_hand_mag_mean": float(res_hand_mag.mean().detach().cpu()),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, is_train, ids):
    return AlignedResidualFlowDataset(
        ids=ids,
        src_pose_map=src_pose_map,
        gt_pose_map=gt_pose_map,
        manifest_rows=manifest_rows,
        gloss_to_id=gloss_to_id,
        max_t=args.max_t,
        max_sent_len=args.sent_max_len,
        is_train=is_train,
        crop_p=args.crop_p,
        source_amp_aug=args.source_amp_aug,
        source_amp_min=args.source_amp_min,
        source_amp_max=args.source_amp_max,
        seed=args.seed + (0 if is_train else 1),
    )


def load_v11_artifacts(args):
    ck = torch.load(ROOT / args.v11_ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ck["gloss_to_id"]
    return gloss_to_id


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v22] loading train source poses from {args.train_src_pt}", flush=True)
    src_pose_map = load_pose_map(ROOT / args.train_src_pt)
    print(f"[v22] loaded {len(src_pose_map)} source clips", flush=True)
    print(f"[v22] loading GT train poses from {args.train_gt_pt}", flush=True)
    gt_pose_map = load_pose_map(ROOT / args.train_gt_pt)
    print(f"[v22] loaded {len(gt_pose_map)} GT clips", flush=True)
    manifest_rows = {str(r["id"]): r for r in json.load(open(ROOT / args.train_manifest_json))}
    gloss_to_id = load_v11_artifacts(args)
    print(f"[v22] gloss vocab = {len(gloss_to_id)}", flush=True)

    common_ids = [sid for sid in src_pose_map.keys() if sid in gt_pose_map and sid in manifest_rows]
    rnd = random.Random(args.seed)
    rnd.shuffle(common_ids)
    n_val = max(64, int(len(common_ids) * args.val_frac))
    val_ids = common_ids[:n_val]
    train_ids = common_ids[n_val:]
    if args.smoke:
        train_ids = train_ids[: args.smoke_train]
        val_ids = val_ids[: args.smoke_val]
    print(f"[v22] train={len(train_ids)} val={len(val_ids)}", flush=True)

    train_ds = build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, True, train_ids)
    val_ds = build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, False, val_ids)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_aligned, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_aligned,
    )

    model = VelocityFieldRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_t=args.max_t,
        sent_max_len=args.sent_max_len,
        max_dv_body=args.max_dv_body,
        max_dv_hand_local=args.max_dv_hand_local,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v22] device={device} params={n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_accum: dict[str, float] = {}
        n_train = 0
        for step, batch in enumerate(train_loader, start=1):
            src = batch["src"].to(device)
            gt_aligned = batch["gt_aligned"].to(device)
            src_mask = batch["src_mask"].to(device)
            sent_ids = batch["sent_ids"].to(device)
            sent_mask = batch["sent_mask"].to(device)
            gt_speed = batch["gt_speed"].to(device)
            gt_jerk = batch["gt_jerk"].to(device)
            pred, deltas = model(src, src_mask, sent_ids, sent_mask)
            loss, logs = compute_losses_v22(
                pred, src, gt_aligned, src_mask, gt_speed, gt_jerk, deltas, args
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in logs.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            n_train += 1
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "grad_norm": float(gnorm), **logs}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        train_log = {f"train_{k}": v / max(n_train, 1) for k, v in train_accum.items()}

        model.eval()
        val_accum: dict[str, float] = {}
        n_val_b = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.val_batches and bi >= args.val_batches:
                    break
                src = batch["src"].to(device)
                gt_aligned = batch["gt_aligned"].to(device)
                src_mask = batch["src_mask"].to(device)
                sent_ids = batch["sent_ids"].to(device)
                sent_mask = batch["sent_mask"].to(device)
                gt_speed = batch["gt_speed"].to(device)
                gt_jerk = batch["gt_jerk"].to(device)
                pred, deltas = model(src, src_mask, sent_ids, sent_mask)
                _, logs = compute_losses_v22(
                    pred, src, gt_aligned, src_mask, gt_speed, gt_jerk, deltas, args
                )
                for k, v in logs.items():
                    val_accum[k] = val_accum.get(k, 0.0) + v
                n_val_b += 1
        val_log = {f"val_{k}": v / max(n_val_b, 1) for k, v in val_accum.items()}
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **train_log, **val_log}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)

        # Selection score: same v21 score plus a small weight on direction loss
        # so we prefer epochs that actually use the supervision.
        score = (
            abs(rec["val_speed_ratio_mean"] - args.speed_band_target)
            + max(0.0, rec["val_jerk_ratio_mean"] - args.jerk_cap) * 2.0
            + max(0.0, args.arc_min_gain - rec["val_arc_gain_mean"]) * 1.0
            + 0.05 * rec["val_L_body_other"] / max(args.lambda_body, 1e-6)
            + 1.0 * rec["val_L_vel_dir"]
        )
        if best is None or score < best["score"]:
            best = {**rec, "score": float(score)}
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best": best,
                    "gloss_to_id": gloss_to_id,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[v22] saved best -> {out_dir / 'best.pt'} score={score:.4f}", flush=True)

    print(f"[v22] done best={best}", flush=True)


# ---------------------------------------------------------------------------
# Apply (inference) driver — IDENTICAL to v21 (GT never used at inference)
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    refiner_args = argparse.Namespace(**ck["args"])
    gloss_to_id = ck["gloss_to_id"]
    model = VelocityFieldRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=refiner_args.hidden,
        layers=refiner_args.layers,
        heads=refiner_args.heads,
        dropout=0.0,
        max_t=refiner_args.max_t,
        sent_max_len=refiner_args.sent_max_len,
        max_dv_body=refiner_args.max_dv_body,
        max_dv_hand_local=refiner_args.max_dv_hand_local,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    src_pose_map = load_pose_map(ROOT / args.in_pt)
    manifest_rows = {str(r["id"]): r for r in json.load(open(ROOT / args.manifest_json))}

    out: dict[str, torch.Tensor] = {}
    trace: dict[str, dict] = {}
    n = len(src_pose_map)
    sent_max_len = int(refiner_args.sent_max_len)
    for i, (sid, src_pose) in enumerate(src_pose_map.items(), start=1):
        row = manifest_rows.get(sid, {"text": "", "gloss": ""})
        sent_glosses = [g for g in str(row.get("gloss", "")).split() if g]
        sent_ids = torch.zeros(1, sent_max_len, dtype=torch.long, device=device)
        sent_mask = torch.zeros(1, sent_max_len, dtype=torch.bool, device=device)
        for j, g in enumerate(sent_glosses[:sent_max_len]):
            sent_ids[0, j] = int(gloss_to_id.get(g, 1))
            sent_mask[0, j] = True
        if sent_mask.sum() == 0:
            sent_mask[0, 0] = True

        T = src_pose.shape[0]
        src_b = src_pose.float().unsqueeze(0).to(device)
        m_b = torch.ones(1, T, dtype=torch.bool, device=device)
        pred, _ = model(src_b, m_b, sent_ids, sent_mask)
        out_pose = pred[0].detach().cpu()
        out[sid] = out_pose.contiguous()
        if args.trace_json:
            trace[sid] = {
                "T": T,
                "src_hand_speed": hand_speed_scalar(src_pose),
                "out_hand_speed": hand_speed_scalar(out_pose),
                "src_hand_jerk": hand_jerk_scalar(src_pose),
                "out_hand_jerk": hand_jerk_scalar(out_pose),
            }
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[v22-apply] {i}/{n}", flush=True)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}", flush=True)
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(trace, indent=2))
        print(f"trace -> {tp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--v11_ckpt", default="outputs/sota_chase/sign_jepa_v11_sentence_budget/best.pt")
    tr.add_argument("--train_src_pt", required=True)
    tr.add_argument("--train_gt_pt",
                    default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_v22_velocity_field_dtw")

    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--max_t", type=int, default=192)
    tr.add_argument("--sent_max_len", type=int, default=32)
    tr.add_argument("--crop_p", type=float, default=0.5)
    tr.add_argument("--source_amp_aug", action="store_true")
    tr.add_argument("--source_amp_min", type=float, default=0.5)
    tr.add_argument("--source_amp_max", type=float, default=1.0)

    tr.add_argument("--max_dv_body", type=float, default=0.004)
    tr.add_argument("--max_dv_hand_local", type=float, default=0.004)

    tr.add_argument("--speed_band_lo", type=float, default=0.85)
    tr.add_argument("--speed_band_hi", type=float, default=1.05)
    tr.add_argument("--speed_band_target", type=float, default=0.95)
    tr.add_argument("--jerk_cap", type=float, default=1.30)
    tr.add_argument("--arc_min_gain", type=float, default=1.20)

    tr.add_argument("--lambda_speed", type=float, default=2.0)
    tr.add_argument("--lambda_jerk", type=float, default=1.0)
    tr.add_argument("--lambda_arc", type=float, default=0.5)
    tr.add_argument("--lambda_dev", type=float, default=0.05)
    tr.add_argument("--lambda_body", type=float, default=20.0)
    tr.add_argument("--lambda_face", type=float, default=20.0)
    tr.add_argument("--lambda_anchor", type=float, default=5.0)
    tr.add_argument("--lambda_bone", type=float, default=50.0)
    tr.add_argument("--lambda_vel_smooth", type=float, default=200.0)
    tr.add_argument("--lambda_vel_dir", type=float, default=0.5,
                    help="weight on per-frame velocity-direction cosine loss vs linearly-resampled GT")

    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--eval_batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)

    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=64)
    tr.add_argument("--smoke_val", type=int, default=16)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--ckpt", required=True)
    ap_apply.add_argument("--in_pt", required=True)
    ap_apply.add_argument("--manifest_json", required=True)
    ap_apply.add_argument("--out_pt", required=True)
    ap_apply.add_argument("--trace_json", default="")
    ap_apply.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_apply.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
