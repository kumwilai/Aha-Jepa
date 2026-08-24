"""AHA-JEPA on CSL-Daily (HRNet-133), gloss-WER track.

Full faithful port of the PHOENIX AHA-JEPA pipeline to CSL-Daily's HRNet-133
keypoint layout. Same architecture and losses as
`train_sign_jepa_generator_slrtp178.py`; only the keypoint layout, data source,
duration policy, and export format differ.

Pipeline (subcommands):
  jepa     : Stage-1 Sign-JEPA pretraining (masked-latent prediction) on CSL
             gloss spans.  -> outputs/sota_chase/sign_jepa_csl/best.pt
  gen      : gloss+duration -> latent -> HRNet-133 pose generator.  Supports the
             stochastic decoder + adversarial hand critic + WAV-KL + energy /
             sentence-budget conditioning (the AHA-JEPA recipe).
  generate : assemble per-gloss segments to the NON-GT teacher (native_hclen)
             length, de-normalise, export an MSKA-format pickle for scoring.
  gov_fit / gov_apply : train-learned utterance-budget governor (leak-free).

Layout-agnostic model classes and math helpers are imported from the PHOENIX
scripts; the ~5 layout-specific functions are reimplemented here for J=133.

Evaluation: MSKA-SLR gloss-WER via scripts/score_csl_mska.py (swap the exported
pickle into MSKA's data dir).  No GT length at inference (teacher native_hclen
length is gloss-predicted, identical across all methods per the CSL protocol).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
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
    HAND_EDGES,
    SignJEPAModel,
    SLRTPSpanDataset,
    build_gloss_vocab,
    compute_stats,
    flat_pose,
    length_bucket_id,
    resize_seq,
    variance_loss,
    sample_future_mask,
)
from scripts.train_sign_jepa_generator_slrtp178 import (  # noqa: E402
    SignJEPAGenerator,
    MotionDiscriminator,
    velocity_loss,
    std_match_loss,
    speed_envelope_loss,
    _wav_smooth_pose_time,
    _soft_histogram,
    load_frozen_jepa,
    allocate_lengths,
    mean_lengths_by_gloss,
    latent_anchor_loss,
)

# ---------------------------------------------------------------------------
# HRNet-133 (COCO-WholeBody) layout.
# ---------------------------------------------------------------------------
J = 133
# region (lo, hi) blocks, ordered for the motion descriptor.
CSL_BLOCKS = {
    "body": (0, 23),     # 17 body + 6 feet
    "rh": (91, 112),     # 21 first-hand
    "lh": (112, 133),    # 21 second-hand
    "face": (23, 91),    # 68 face
}
HAND_OFFSETS = (91, 112)          # 21-joint hand block starts (hand roots / wrists)
LW, RW = 9, 10                    # body left/right wrist
DISC_IDX = [LW, RW] + list(range(91, 133))   # 44 joints for the hand critic
HAND_SLICE = slice(91, 133)       # both hands, for WAV-KL / governor
FACE_FLAT = slice(CSL_BLOCKS["face"][0] * 3, CSL_BLOCKS["face"][1] * 3)


class CSLMotionDiscriminator(MotionDiscriminator):
    """PatchGAN hand critic with HRNet-133 hand+wrist indices."""

    HAND_WRIST_IDX = DISC_IDX     # [9, 10] + range(91, 133) -> 44 joints


# ---------------------------------------------------------------------------
# Layout-specific feature / descriptor / loss reimplementations (J=133).
# ---------------------------------------------------------------------------
def carrier_pose_view(pose_norm: torch.Tensor, pose_raw: torch.Tensor,
                      mean: torch.Tensor | None, face_weight: float = 1.0):
    """Pose view used only by the carrier objective/contract.

    ``face_weight=0`` freezes the HRNet face block to the training-set mean,
    removing face-coordinate variance from the carrier while leaving the full
    133-joint pose generator and MSKA export untouched.
    """
    face_weight = float(face_weight)
    if abs(face_weight - 1.0) < 1e-8:
        return pose_norm, pose_raw
    if mean is None:
        raise ValueError("mean is required when carrier_face_weight != 1")
    p = pose_norm.clone()
    pr = pose_raw.clone()
    m = mean.to(device=pose_raw.device, dtype=pose_raw.dtype)
    p[..., FACE_FLAT] = p[..., FACE_FLAT] * face_weight
    pr[..., FACE_FLAT] = m[FACE_FLAT] + face_weight * (pr[..., FACE_FLAT] - m[FACE_FLAT])
    return p, pr


def latent_rank_spread_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Encourage a high-rank, non-collapsed latent without using evaluator labels."""
    if z.numel() == 0:
        return z.new_zeros(())
    x = z.reshape(-1, z.shape[-1])
    if x.shape[0] < 2:
        return z.new_zeros(())
    x = (x - x.mean(0, keepdim=True)) / torch.sqrt(x.var(0, keepdim=True) + eps)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return off.pow(2).mean()


def sequence_features(pose_norm: torch.Tensor, pose_raw: torch.Tensor):
    """HRNet-133 version of the SLRTP-178 sequence_features."""
    B, T, D = pose_norm.shape
    vel = torch.zeros_like(pose_norm)
    vel[:, 1:] = pose_norm[:, 1:] - pose_norm[:, :-1]
    acc = torch.zeros_like(pose_norm)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

    raw = pose_raw.reshape(B, T, J, 3)
    vel_raw = torch.zeros_like(raw)
    vel_raw[:, 1:] = raw[:, 1:] - raw[:, :-1]
    acc_raw = torch.zeros_like(raw)
    acc_raw[:, 1:] = vel_raw[:, 1:] - vel_raw[:, :-1]

    speeds, accels = [], []
    for lo, hi in CSL_BLOCKS.values():
        speeds.append(vel_raw[:, :, lo:hi].norm(dim=-1).mean(dim=-1, keepdim=True))
        accels.append(acc_raw[:, :, lo:hi].norm(dim=-1).mean(dim=-1, keepdim=True))
    speed_desc = torch.cat(speeds, dim=-1)
    accel_desc = torch.cat(accels, dim=-1)

    edges = torch.as_tensor(HAND_EDGES, device=pose_raw.device, dtype=torch.long)
    hand_dirs = []
    for lo in HAND_OFFSETS:
        hand = raw[:, :, lo:lo + 21]
        bones = hand[:, :, edges[:, 1]] - hand[:, :, edges[:, 0]]
        dirs = F.normalize(bones, dim=-1, eps=1e-6).reshape(B, T, -1)
        hand_dirs.append(dirs)
    hand_dirs = torch.cat(hand_dirs, dim=-1)

    wr = list(HAND_OFFSETS)
    wrists = raw[:, :, wr].reshape(B, T, 6)
    wrist_vel = vel_raw[:, :, wr].reshape(B, T, 6)

    motion_desc = torch.cat([speed_desc, accel_desc, hand_dirs, wrists, wrist_vel], dim=-1)
    enc_feat = torch.cat([pose_norm, vel, acc, motion_desc], dim=-1)
    return enc_feat, motion_desc


@torch.no_grad()
def encode_target_latents(jepa, pose_norm, pose_raw, gloss_id, len_id,
                          mean: torch.Tensor | None = None, carrier_face_weight: float = 1.0):
    pose_c, pose_raw_c = carrier_pose_view(pose_norm, pose_raw, mean, carrier_face_weight)
    feat, desc = sequence_features(pose_c, pose_raw_c)
    zero_mask = torch.zeros(pose_norm.shape[:2], dtype=torch.bool, device=pose_norm.device)
    z = jepa.target_encoder(feat, zero_mask, gloss_id, len_id)
    return z.detach(), desc.detach()


def energy_budget_descriptor(pose_raw: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
    """16-dim per-hand speed/accel motion-budget descriptor (HRNet-133 hands)."""
    pose = pose_raw.reshape(pose_raw.shape[0], pose_raw.shape[1], J, 3)
    parts = []
    for lo in HAND_OFFSETS:
        hand = pose[:, :, lo:lo + 21]
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


def _wav_hand_vel_magnitudes(pose_raw: torch.Tensor, kernel: int) -> torch.Tensor:
    smooth = _wav_smooth_pose_time(pose_raw, kernel)
    smooth = smooth.reshape(smooth.shape[0], smooth.shape[1], J, 3)
    hand = smooth[:, :, HAND_SLICE]                  # (B, T, 42, 3)
    vel = hand[:, 1:] - hand[:, :-1]
    return vel.norm(dim=-1).reshape(-1)


def wavkl_hand_loss(pred_pose_raw, gt_hists, bin_centers, sigma, scales, eps=1e-8):
    parts = []
    for k in scales:
        mag = _wav_hand_vel_magnitudes(pred_pose_raw, k)
        pred_h = _soft_histogram(mag, bin_centers, sigma)
        gt_h = gt_hists[k]
        kl = (pred_h * ((pred_h + eps).log() - (gt_h + eps).log())).sum()
        parts.append(kl)
    return torch.stack(parts).mean()


@torch.no_grad()
def precompute_gt_wavkl(loader, device, scales, n_bins, vmax, sigma, max_batches=0):
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_bank():
    bank = torch.load(ROOT / "outputs/sota_chase/csl_jepa_bank.pt",
                      map_location="cpu", weights_only=False)
    spans = []
    for gloss, rows in bank["gloss_to_exemplars"].items():
        for sid, s, e in rows:
            if sid not in bank["exemplar_poses"]:
                continue
            T = int(bank["exemplar_poses"][sid].shape[0])
            s = max(0, min(int(s), T))
            e = max(0, min(int(e), T))
            if e - s >= 3:
                spans.append((str(gloss), sid, s, e))
    return bank, spans


def load_msla(tag: str) -> dict:
    return pickle.load(open(ROOT / f"outputs/csl_msla_inputs/CSL-Daily.{tag}", "rb"))


# ===========================================================================
# Stage 1: Sign-JEPA pretraining
# ===========================================================================
def csl_carrier_loss(model, feat, desc, gloss_id, len_id, recon_head, lex_head, args, device):
    """Carrier objective for the CSL (J=133) layout, mirroring the PHOENIX
    ``carrier_losses``. objective='jepa': masked latent prediction + EMA target
    (default). 'mae': masked raw-feature reconstruction. 'ae': full-frame
    reconstruction. For mae/ae the predictor latent is decoded by ``recon_head``
    and the EMA momentum is forced to 0 by the caller, so the frozen target
    encoder used downstream is exactly the reconstruction-trained context encoder.
    """
    obj = getattr(args, "objective", "jepa")
    if obj == "ae":
        future_mask = torch.zeros(feat.shape[0], feat.shape[1], dtype=torch.bool, device=device)
    else:
        future_mask = sample_future_mask(feat.shape[0], feat.shape[1], args.min_context,
                                         args.target_len_min, args.target_len_max, device)
    pred_z, target_z, pred_desc = model(feat, gloss_id, len_id, future_mask)
    if obj == "jepa":
        m = future_mask.unsqueeze(-1)
        pred_n = F.normalize(pred_z, dim=-1); target_n = F.normalize(target_z, dim=-1)
        latent = F.smooth_l1_loss(pred_n[m.expand_as(pred_n)], target_n[m.expand_as(target_n)])
        desc_loss = F.smooth_l1_loss(pred_desc[m.expand_as(pred_desc)], desc[m.expand_as(desc)])
        var = variance_loss(pred_z[future_mask]) + variance_loss(target_z[future_mask])
        rank = latent_rank_spread_loss(pred_z[future_mask])
        lex = pred_z.new_zeros(())
        if lex_head is not None:
            lex = F.cross_entropy(lex_head(pred_z.mean(dim=1)), gloss_id)
        loss = (latent + args.lambda_desc * desc_loss + args.lambda_var * var
                + args.lambda_latent_rank * rank + args.lambda_lex_ce * lex)
        return loss, pred_z, target_z
    # reconstruction objectives (mae / ae)
    recon = recon_head(pred_z)
    if obj == "mae":
        m = future_mask.unsqueeze(-1)
        recon_loss = F.smooth_l1_loss(recon[m.expand_as(recon)], feat[m.expand_as(feat)])
        var = variance_loss(pred_z[future_mask])
    else:  # ae
        recon_loss = F.smooth_l1_loss(recon, feat)
        var = variance_loss(pred_z)
    loss = recon_loss + args.lambda_var * var
    return loss, pred_z, target_z


def train_jepa(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    objective = getattr(args, "objective", "jepa")
    ema = args.ema if objective == "jepa" else 0.0

    bank, spans = load_bank()
    gloss_to_id = build_gloss_vocab(bank)
    mean, std = compute_stats(bank["exemplar_poses"])
    print(f"[jepa] spans={len(spans)} glosses={len(gloss_to_id)} pose_dim={mean.numel()}", flush=True)

    random.Random(args.seed).shuffle(spans)
    n_val = max(args.min_val, int(args.val_frac * len(spans)))
    val_spans, train_spans = spans[:n_val], spans[n_val:]
    if args.smoke:
        train_spans = train_spans[:args.smoke]; val_spans = val_spans[:max(8, args.smoke // 4)]

    train_ds = SLRTPSpanDataset(bank, train_spans, gloss_to_id, mean, std, args.seg_len, seed=args.seed)
    val_ds = SLRTPSpanDataset(bank, val_spans, gloss_to_id, mean, std, args.seg_len, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    sample = train_ds[0]
    sample_pose, sample_pose_raw = carrier_pose_view(
        sample["pose"][None], sample["pose_raw"][None], mean, args.carrier_face_weight)
    feat, desc = sequence_features(sample_pose, sample_pose_raw)
    model = SignJEPAModel(feat_dim=feat.shape[-1], desc_dim=desc.shape[-1], num_gloss=len(gloss_to_id),
                          hidden=args.hidden, layers=args.layers, heads=args.heads,
                          pred_layers=args.pred_layers, dropout=args.dropout, max_len=args.seg_len).to(device)
    recon_head = None
    lex_head = None
    params = list(model.parameters())
    if objective in ("mae", "ae"):
        recon_head = nn.Linear(args.hidden, feat.shape[-1]).to(device)
        params = params + list(recon_head.parameters())
    elif args.lambda_lex_ce > 0:
        lex_head = nn.Sequential(nn.LayerNorm(args.hidden), nn.Linear(args.hidden, len(gloss_to_id))).to(device)
        params = params + list(lex_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[jepa] objective={objective} ema={ema} feat_dim={feat.shape[-1]} desc_dim={desc.shape[-1]} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    saved_args = dict(hidden=args.hidden, layers=args.layers, heads=args.heads,
                      pred_layers=args.pred_layers, seg_len=args.seg_len,
                      carrier_face_weight=float(args.carrier_face_weight),
                      lambda_latent_rank=float(args.lambda_latent_rank),
                      lambda_lex_ce=float(args.lambda_lex_ce))
    log, best = [], None
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); vals = []
        for step, batch in enumerate(train_loader, 1):
            pose = batch["pose"].to(device); pose_raw = batch["pose_raw"].to(device)
            gloss_id = batch["gloss_id"].to(device); len_id = batch["len_id"].to(device)
            pose_c, pose_raw_c = carrier_pose_view(pose, pose_raw, mean, args.carrier_face_weight)
            feat, desc = sequence_features(pose_c, pose_raw_c)
            loss, _pz, _tz = csl_carrier_loss(model, feat, desc, gloss_id, len_id,
                                              recon_head, lex_head, args, device)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step(); model.update_target(ema)
            vals.append(float(loss.detach().cpu()))
            if args.max_steps and step >= args.max_steps:
                break
        # validation
        model.eval(); vvals = []
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= args.val_batches:
                    break
                pose = batch["pose"].to(device); pose_raw = batch["pose_raw"].to(device)
                gloss_id = batch["gloss_id"].to(device); len_id = batch["len_id"].to(device)
                pose_c, pose_raw_c = carrier_pose_view(pose, pose_raw, mean, args.carrier_face_weight)
                feat, desc = sequence_features(pose_c, pose_raw_c)
                vloss, _pz, _tz = csl_carrier_loss(model, feat, desc, gloss_id, len_id,
                                                   recon_head, lex_head, args, device)
                vvals.append(float(vloss.cpu()))
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1),
               "train_loss": float(np.mean(vals)), "val_loss": float(np.mean(vvals))}
        log.append(rec); (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save({"model": model.state_dict(), "args": saved_args, "mean": mean, "std": std,
                        "gloss_to_id": gloss_to_id, "feat_dim": feat.shape[-1],
                        "desc_dim": desc.shape[-1], "best": best}, out_dir / "best.pt")
            print(f"[jepa] saved best -> {out_dir/'best.pt'} val={best['val_loss']:.4f}", flush=True)
    print(f"[jepa] done best={best}", flush=True)


# ===========================================================================
# Stage 2: generator (stochastic decoder + adversarial hand critic + WAV-KL)
# ===========================================================================
def csl_jepa_pc_loss(jepa, pose_pred, pose_pred_raw, gloss_id, len_id, z_tgt, args,
                     mean: torch.Tensor | None = None, carrier_face_weight: float = 1.0):
    """Predictive-contract loss for the CSL (J=133) layout.

    The ordinary latent anchor only asks the generated latent to be close to the
    frozen target encoder; a reconstruction carrier can satisfy that because the
    generator also sees pose/velocity/descriptor losses. This adds the missing
    JEPA contract: when future GENERATED frames are masked, the frozen carrier
    predictor must recover the GT target latent for those frames from the visible
    generated context. JEPA was pretrained for exactly this; AE/MAE were not.
    """
    pose_c, pose_raw_c = carrier_pose_view(pose_pred, pose_pred_raw, mean, carrier_face_weight)
    feat_pred, _desc_pred = sequence_features(pose_c, pose_raw_c)
    future_mask = sample_future_mask(pose_pred.shape[0], pose_pred.shape[1],
                                     args.jepa_pc_min_context, args.jepa_pc_target_len_min,
                                     args.jepa_pc_target_len_max, pose_pred.device)
    if not future_mask.any():
        return pose_pred.new_zeros(())
    pc_z, _self_z, _pc_desc = jepa(feat_pred, gloss_id, len_id, future_mask)
    return latent_anchor_loss(pc_z[future_mask], z_tgt[future_mask],
                              args.jepa_pc_loss_mode, args.jepa_pc_raw_weight, args.jepa_pc_cov_weight)


def train_gen(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jepa, jepa_ckpt = load_frozen_jepa(ROOT / args.jepa_ckpt, device)
    bank, spans = load_bank()
    gloss_to_id = jepa_ckpt["gloss_to_id"]
    mean = jepa_ckpt["mean"].float(); std = jepa_ckpt["std"].float()
    carrier_face_weight = float(jepa_ckpt.get("args", {}).get("carrier_face_weight", 1.0))
    gloss_mean_lengths = mean_lengths_by_gloss(spans)
    row_by_sid = {}
    if args.use_sentence_budget:
        rows = json.load(open(ROOT / args.train_manifest_json))
        row_by_sid = {str(r["id"]): r for r in rows}

    random.Random(args.seed).shuffle(spans)
    n_val = max(args.min_val, int(args.val_frac * len(spans)))
    val_spans, train_spans = spans[:n_val], spans[n_val:]
    if args.smoke:
        train_spans = train_spans[:args.smoke]; val_spans = val_spans[:max(8, args.smoke // 4)]

    seg_len = int(jepa_ckpt["args"]["seg_len"])
    hidden = int(jepa_ckpt["args"]["hidden"])
    pose_dim = J * 3
    train_ds = SLRTPSpanDataset(bank, train_spans, gloss_to_id, mean, std, seg_len, seed=args.seed,
                                row_by_sid=row_by_sid, sent_max_len=args.sentence_max_len)
    val_ds = SLRTPSpanDataset(bank, val_spans, gloss_to_id, mean, std, seg_len, seed=args.seed + 7,
                              row_by_sid=row_by_sid, sent_max_len=args.sentence_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = SignJEPAGenerator(num_gloss=len(gloss_to_id), hidden=hidden, gen_layers=args.gen_layers,
                              dec_layers=args.dec_layers, heads=args.heads, dropout=args.dropout,
                              max_len=seg_len, pose_dim=pose_dim, use_energy_cond=args.use_energy_cond,
                              energy_dim=args.energy_dim, use_sentence_budget=args.use_sentence_budget,
                              sentence_max_len=args.sentence_max_len, stochastic_decoder=args.stochastic_decoder,
                              init_log_var=args.init_log_var).to(device)
    if args.init_from:
        ck = torch.load(ROOT / args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
        print(f"[gen] init_from {args.init_from} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    print(f"[gen] seg_len={seg_len} hidden={hidden} pose_dim={pose_dim} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    discriminator = d_opt = None
    if args.lambda_adv > 0:
        discriminator = CSLMotionDiscriminator(hidden=args.d_hidden, dropout=args.d_dropout).to(device)
        d_opt = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, weight_decay=0.0, betas=(0.0, 0.99))
        print(f"[gen] v29 hand critic params={sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M "
              f"lambda_adv={args.lambda_adv}", flush=True)

    wavkl_state = None
    if args.lambda_wavkl > 0:
        scales = tuple(int(k) for k in str(args.wavkl_scales).split(",") if k.strip())
        sigma = float(args.wavkl_sigma) if args.wavkl_sigma > 0 else (args.wavkl_vmax / args.wavkl_bins) * 0.6
        hist_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        gt_hists, bin_centers = precompute_gt_wavkl(hist_loader, device, scales, args.wavkl_bins,
                                                    args.wavkl_vmax, sigma, max_batches=int(args.wavkl_max_batches))
        wavkl_state = {"gt_hists": gt_hists, "bin_centers": bin_centers, "sigma": sigma, "scales": scales}
        for k, h in gt_hists.items():
            print(f"[gen] WAV-KL GT k={k} zero-bin={h[0].item():.4f} peak-bin={h.argmax().item()}", flush=True)

    def run_batch(batch, train_mode):
        pose = batch["pose"].to(device); pose_raw = batch["pose_raw"].to(device)
        gloss_id = batch["gloss_id"].to(device); len_id = batch["len_id"].to(device)
        z_tgt, desc_tgt = encode_target_latents(
            jepa, pose, pose_raw, gloss_id, len_id, mean.to(device), carrier_face_weight)
        energy_cond = None; energy_pred_loss = pose.new_zeros(())
        if args.use_energy_cond:
            energy_gt = energy_budget_descriptor(pose_raw, scale=args.energy_desc_scale)
            if args.use_sentence_budget:
                energy_pred = model.predict_condition_energy(
                    gloss_id, len_id, batch["sent_gloss_ids"].to(device),
                    batch["sent_mask"].to(device), batch["sent_pos"].to(device))
            else:
                energy_pred = model.predict_energy(gloss_id, len_id)
            energy_pred_loss = F.smooth_l1_loss(energy_pred, energy_gt)
            energy_cond = energy_pred if args.use_sentence_budget else energy_gt
        z_pred, pose_pred, pose_log_var = model(gloss_id, len_id, pose.shape[1], energy=energy_cond)

        latent = F.smooth_l1_loss(F.normalize(z_pred, dim=-1), F.normalize(z_tgt, dim=-1))
        if pose_log_var is not None and args.lambda_nll > 0:
            lv = pose_log_var.clamp(min=-8.0, max=2.0)
            pose_loss = (0.5 * ((pose_pred - pose).pow(2) * (-lv).exp() + lv)).mean()
        else:
            pose_loss = F.smooth_l1_loss(pose_pred, pose)
        vel = velocity_loss(pose_pred, pose)
        pose_pred_raw = pose_pred * std.to(device) + mean.to(device)
        pose_pred_c, pose_pred_raw_c = carrier_pose_view(
            pose_pred, pose_pred_raw, mean.to(device), carrier_face_weight)
        _f, desc_pred = sequence_features(pose_pred_c, pose_pred_raw_c)
        desc = F.smooth_l1_loss(desc_pred, desc_tgt)
        wavkl = pose.new_zeros(())
        if wavkl_state is not None:
            wavkl = wavkl_hand_loss(pose_pred_raw, wavkl_state["gt_hists"], wavkl_state["bin_centers"],
                                    wavkl_state["sigma"], wavkl_state["scales"])
        adv_g = pose.new_zeros(()); adv_d = 0.0
        if discriminator is not None and train_mode:
            B, Tseg = pose_pred.shape[:2]
            sd = std.to(device); mn = mean.to(device)
            fake_d = (pose_pred.detach() * sd + mn).reshape(B, Tseg, J, 3)
            real_d = pose_raw.reshape(B, Tseg, J, 3)
            d_loss = F.relu(1.0 - discriminator(real_d)).mean() + F.relu(1.0 + discriminator(fake_d)).mean()
            d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
            adv_d = float(d_loss.detach().cpu())
            adv_g = -discriminator((pose_pred * sd + mn).reshape(B, Tseg, J, 3)).mean()
        jepa_pc = pose.new_zeros(())
        if args.lambda_jepa_pc > 0:
            jepa_pc = csl_jepa_pc_loss(
                jepa, pose_pred, pose_pred_raw, gloss_id, len_id, z_tgt, args,
                mean.to(device), carrier_face_weight)
        loss = (args.lambda_latent * latent + args.lambda_pose * pose_loss + args.lambda_vel * vel
                + args.lambda_desc * desc + args.lambda_wavkl * wavkl
                + args.lambda_energy_pred * energy_pred_loss + args.lambda_adv * adv_g
                + args.lambda_jepa_pc * jepa_pc)
        return loss, dict(latent=float(latent), pose=float(pose_loss), wavkl=float(wavkl),
                          jepa_pc=float(jepa_pc) if torch.is_tensor(jepa_pc) else jepa_pc,
                          adv_g=float(adv_g) if torch.is_tensor(adv_g) else adv_g, adv_d=adv_d)

    log, best = [], None
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); vals = []; aux = {}
        for step, batch in enumerate(train_loader, 1):
            loss, parts = run_batch(batch, True)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
            vals.append(float(loss.detach().cpu()))
            for k, v in parts.items():
                aux.setdefault(k, []).append(v)
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"ep": ep, "step": step, "loss": vals[-1], **{k: parts[k] for k in parts}}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        model.eval(); vvals = []
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= args.val_batches:
                    break
                loss, _ = run_batch(batch, False)
                vvals.append(float(loss.cpu()))
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1),
               "train_loss": float(np.mean(vals)), "val_loss": float(np.mean(vvals)),
               **{f"train_{k}": float(np.mean(v)) for k, v in aux.items()}}
        log.append(rec); (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best,
                        "gloss_to_id": gloss_to_id, "mean": mean, "std": std, "seg_len": seg_len,
                        "hidden": hidden, "pose_dim": pose_dim, "gloss_mean_lengths": gloss_mean_lengths,
                        "discriminator": discriminator.state_dict() if discriminator is not None else None},
                       out_dir / "best.pt")
            print(f"[gen] saved best -> {out_dir/'best.pt'} val={best['val_loss']:.4f}", flush=True)
    print(f"[gen] done best={best}", flush=True)


# ===========================================================================
# Inference: assemble per-gloss segments to teacher (non-GT) length, export.
# ===========================================================================
def build_gen_from_ckpt(ckpt, device):
    a = ckpt["args"]
    model = SignJEPAGenerator(num_gloss=len(ckpt["gloss_to_id"]), hidden=ckpt["hidden"],
                              gen_layers=a["gen_layers"], dec_layers=a["dec_layers"], heads=a["heads"],
                              dropout=0.0, max_len=ckpt["seg_len"], pose_dim=ckpt["pose_dim"],
                              use_energy_cond=a.get("use_energy_cond", False), energy_dim=a.get("energy_dim", 16),
                              use_sentence_budget=a.get("use_sentence_budget", False),
                              sentence_max_len=a.get("sentence_max_len", 32),
                              stochastic_decoder=a.get("stochastic_decoder", False),
                              init_log_var=a.get("init_log_var", -4.0)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ckpt["gloss_to_id"]
    model = build_gen_from_ckpt(ckpt, device)
    mean = ckpt["mean"].to(device); std = ckpt["std"].to(device)
    mean_lengths = ckpt.get("gloss_mean_lengths", {})
    seg_len = ckpt["seg_len"]; sent_max = ckpt["args"].get("sentence_max_len", 32)
    teacher = load_msla(f"{args.split}_{args.teacher}")
    out = {}
    for i, (sid, rec) in enumerate(teacher.items(), 1):
        glosses = [g for g in str(rec.get("gloss", "")).split() if g]
        total_len = int(rec["keypoint"].shape[0])     # NON-GT teacher (native_hclen) length
        if not glosses or total_len <= 0:
            out[sid] = {"name": rec["name"], "gloss": rec["gloss"], "text": rec["text"],
                        "keypoint": torch.zeros(max(total_len, 4), J, 3), "num_frames": max(total_len, 4)}
            continue
        lengths = allocate_lengths(glosses, total_len, mean_lengths)
        sent_ids = torch.zeros(1, sent_max, dtype=torch.long, device=device)
        sent_mask = torch.zeros(1, sent_max, dtype=torch.bool, device=device)
        for j, g in enumerate(glosses[:sent_max]):
            sent_ids[0, j] = int(gloss_to_id.get(g, 1)); sent_mask[0, j] = True
        if not glosses:
            sent_mask[0, 0] = True
        chunks = []
        for pos_i, (gloss, L) in enumerate(zip(glosses, lengths)):
            gid = torch.tensor([gloss_to_id.get(gloss, 1)], dtype=torch.long, device=device)
            lid = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
            energy = None
            if ckpt["args"].get("use_energy_cond", False):
                if ckpt["args"].get("use_sentence_budget", False):
                    spos = torch.tensor([min(pos_i, sent_max - 1)], dtype=torch.long, device=device)
                    energy = model.predict_condition_energy(gid, lid, sent_ids, sent_mask, spos)
                else:
                    energy = model.predict_energy(gid, lid)
                energy = energy.detach().clamp_min(0.0)
            _z, pose, log_var = model(gid, lid, seg_len, energy=energy)
            if log_var is not None and args.sample_temperature > 0:
                pose = pose + (0.5 * log_var.clamp(-8, 2)).exp() * torch.randn_like(pose) * args.sample_temperature
            pose = pose[0] * std + mean
            pose = resize_seq(pose.cpu(), L).reshape(L, J, 3).float()
            chunks.append(pose)
        X = torch.cat(chunks, dim=0).contiguous()
        out[sid] = {"name": rec["name"], "gloss": rec["gloss"], "text": rec["text"],
                    "keypoint": X, "num_frames": int(X.shape[0])}
        if i % args.log_every == 0 or i == 1:
            print(f"[generate] {i}/{len(teacher)}", flush=True)
    op = ROOT / f"outputs/csl_msla_inputs/CSL-Daily.{args.split}_{args.tag}"
    pickle.dump(out, open(op, "wb"))
    lens = np.asarray([v["keypoint"].shape[0] for v in out.values()])
    print(f"[generate] saved -> {op} clips={len(out)} T mean={lens.mean():.1f}", flush=True)


# ===========================================================================
# Train-learned utterance-budget governor (leak-free).
# ===========================================================================
def gov_hand_speed(pose: torch.Tensor) -> float:
    x = pose.reshape(pose.shape[0], J, 3).float()
    if x.shape[0] < 2:
        return 0.0
    v = x[1:, HAND_SLICE] - x[:-1, HAND_SLICE]
    return float(v.norm(dim=-1).mean())


def gov_hand_jerk(pose: torch.Tensor) -> float:
    x = pose.reshape(pose.shape[0], J, 3).float()
    if x.shape[0] < 3:
        return 0.0
    v = x[1:, HAND_SLICE] - x[:-1, HAND_SLICE]
    a = v[1:] - v[:-1]
    return float(a.norm(dim=-1).mean())


def gov_amplify(pose: torch.Tensor, g_hand: float, g_body: float, lp_kernel: int) -> torch.Tensor:
    """Per-region motion amplification about a temporal low-pass baseline."""
    x = pose.reshape(pose.shape[0], J, 3).float().clone()
    base = x
    if lp_kernel > 1 and x.shape[0] >= 3:
        y = x.reshape(x.shape[0], -1).T[None]
        pad = lp_kernel // 2
        y = F.pad(y, (pad, pad), mode="replicate")
        y = F.avg_pool1d(y, kernel_size=lp_kernel, stride=1)
        base = y[0].T.reshape(x.shape)
    out = x.clone()
    body = slice(0, 23)
    out[:, body] = base[:, body] + g_body * (x[:, body] - base[:, body])
    out[:, HAND_SLICE] = base[:, HAND_SLICE] + g_hand * (x[:, HAND_SLICE] - base[:, HAND_SLICE])
    return out.reshape(pose.shape)


def _gov_stable_hash(token: str, n: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n


def _gov_features(row: dict, src: torch.Tensor, n_hash: int) -> np.ndarray:
    glosses = [g for g in str(row.get("gloss", "")).split() if g]
    words = [w for w in str(row.get("text", "")).split() if w]
    sp = max(gov_hand_speed(src), 1e-8); jk = max(gov_hand_jerk(src), 1e-8)
    T = max(int(src.shape[0]), 1)
    nums = np.asarray([1.0, math.log1p(len(glosses)), math.log1p(len(words)),
                       math.log1p(len(str(row.get("text", "")))), math.log1p(T),
                       math.log(sp), math.log(jk), sp, jk], dtype=np.float64)
    x = np.zeros(nums.shape[0] + n_hash, dtype=np.float64); x[:nums.shape[0]] = nums
    toks = [f"g:{g}" for g in glosses] + [f"w:{w}" for w in words]
    if toks:
        scale = 1.0 / math.sqrt(len(toks))
        for tok in toks:
            x[nums.shape[0] + _gov_stable_hash(tok, n_hash)] += scale
    return x


def gov_fit(args):
    # train-only: source = generated TRAIN poses, reference = GT TRAIN poses.
    rows = {str(r["id"]): r for r in json.load(open(ROOT / args.train_manifest))}
    source = load_msla(f"train_{args.train_source}")
    ref = load_msla(f"train_{args.train_reference}")
    xs, ys, ratios, kept = [], [], [], 0
    for sid, srec in source.items():
        if sid not in rows or sid not in ref:
            continue
        src = srec["keypoint"].float()
        ref_speed = max(gov_hand_speed(ref[sid]["keypoint"].float()), 1e-8)
        src_speed = max(gov_hand_speed(src), 1e-8)
        ratio = float(np.clip(ref_speed / src_speed, args.target_ratio_min, args.target_ratio_max))
        xs.append(_gov_features(rows[sid], src, args.n_hash)); ys.append(math.log(ratio)); ratios.append(ratio)
        kept += 1
    if kept < 100:
        raise RuntimeError(f"too few train pairs: {kept}")
    X = np.stack(xs); y = np.asarray(ys)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-6] = 1.0
    Xn = (X - mu) / sd
    w = np.linalg.solve(Xn.T @ Xn + args.ridge * np.eye(Xn.shape[1]), Xn.T @ y)
    out = {"n_hash": args.n_hash, "mu": torch.from_numpy(mu).float(), "sd": torch.from_numpy(sd).float(),
           "w": torch.from_numpy(w).float(), "train_pairs": kept,
           "train_ratio_mean": float(np.mean(ratios))}
    op = ROOT / args.out_policy; op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, op)
    print(f"[gov] fit pairs={kept} ratio_mean={np.mean(ratios):.3f} -> {op}", flush=True)


def gov_apply(args):
    policy = torch.load(ROOT / args.policy, map_location="cpu", weights_only=False)
    rows = {str(r["id"]): r for r in json.load(open(ROOT / args.manifest_json))}
    source = load_msla(f"{args.split}_{args.in_tag}")
    mu = policy["mu"].numpy().astype(np.float64); sd = policy["sd"].numpy().astype(np.float64)
    w = policy["w"].numpy().astype(np.float64); n_hash = int(policy["n_hash"])
    out = {}; gains = []
    for sid, rec in source.items():
        src = rec["keypoint"].float()
        row = rows.get(sid, {"id": sid, "text": "", "gloss": ""})
        pred = float(((_gov_features(row, src, n_hash) - mu) / sd) @ w)
        g_hand = float(np.clip(math.exp(pred), args.min_gain, args.max_gain))
        g_body = float(np.clip(1.0 + args.body_coupling * (g_hand - 1.0), 1.0, args.max_body_gain))
        X = gov_amplify(src, g_hand, g_body, args.lp_kernel)
        out[sid] = {"name": rec["name"], "gloss": rec["gloss"], "text": rec["text"],
                    "keypoint": X, "num_frames": int(X.shape[0])}
        gains.append(g_hand)
    op = ROOT / f"outputs/csl_msla_inputs/CSL-Daily.{args.split}_{args.tag}"
    pickle.dump(out, open(op, "wb"))
    print(f"[gov] applied g_hand mean={np.mean(gains):.3f} -> {op} clips={len(out)}", flush=True)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    jp = sub.add_parser("jepa")
    jp.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_csl")
    jp.add_argument("--seg_len", type=int, default=64); jp.add_argument("--hidden", type=int, default=256)
    jp.add_argument("--layers", type=int, default=4); jp.add_argument("--heads", type=int, default=4)
    jp.add_argument("--pred_layers", type=int, default=2); jp.add_argument("--dropout", type=float, default=0.05)
    jp.add_argument("--epochs", type=int, default=12); jp.add_argument("--batch_size", type=int, default=64)
    jp.add_argument("--eval_batch_size", type=int, default=96); jp.add_argument("--lr", type=float, default=2e-4)
    jp.add_argument("--weight_decay", type=float, default=1e-4); jp.add_argument("--grad_clip", type=float, default=1.0)
    jp.add_argument("--ema", type=float, default=0.996); jp.add_argument("--lambda_desc", type=float, default=0.5)
    jp.add_argument("--lambda_var", type=float, default=0.02); jp.add_argument("--min_context", type=int, default=16)
    jp.add_argument("--target_len_min", type=int, default=8); jp.add_argument("--target_len_max", type=int, default=24)
    jp.add_argument("--val_frac", type=float, default=0.05); jp.add_argument("--min_val", type=int, default=512)
    jp.add_argument("--val_batches", type=int, default=20); jp.add_argument("--num_workers", type=int, default=2)
    jp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    jp.add_argument("--seed", type=int, default=0); jp.add_argument("--max_steps", type=int, default=0)
    jp.add_argument("--smoke", type=int, default=0)
    jp.add_argument("--objective", choices=["jepa", "mae", "ae"], default="jepa",
                    help="Carrier pretraining objective. jepa=masked latent prediction (EMA); "
                         "mae=masked raw-feature reconstruction; ae=full-frame reconstruction.")
    jp.add_argument("--carrier_face_weight", type=float, default=1.0,
                    help="Weight for HRNet face coordinates inside carrier features only. "
                         "1.0=full CSL pose; 0.0=freeze face to training-set mean.")
    jp.add_argument("--lambda_latent_rank", type=float, default=0.0,
                    help="JEPA-only latent decorrelation penalty for higher effective rank.")
    jp.add_argument("--lambda_lex_ce", type=float, default=0.0,
                    help="JEPA-only auxiliary gloss classifier on pooled predicted latents; "
                         "uses training gloss labels, not the MSKA evaluator.")

    gn = sub.add_parser("gen")
    gn.add_argument("--jepa_ckpt", default="outputs/sota_chase/sign_jepa_csl/best.pt")
    gn.add_argument("--out_dir", required=True); gn.add_argument("--init_from", default="")
    gn.add_argument("--gen_layers", type=int, default=4); gn.add_argument("--dec_layers", type=int, default=3)
    gn.add_argument("--heads", type=int, default=4); gn.add_argument("--dropout", type=float, default=0.05)
    gn.add_argument("--epochs", type=int, default=12); gn.add_argument("--batch_size", type=int, default=64)
    gn.add_argument("--eval_batch_size", type=int, default=96); gn.add_argument("--lr", type=float, default=2e-4)
    gn.add_argument("--weight_decay", type=float, default=1e-4); gn.add_argument("--grad_clip", type=float, default=1.0)
    gn.add_argument("--lambda_latent", type=float, default=1.0); gn.add_argument("--lambda_pose", type=float, default=0.2)
    gn.add_argument("--lambda_vel", type=float, default=0.2); gn.add_argument("--lambda_desc", type=float, default=0.3)
    gn.add_argument("--lambda_nll", type=float, default=0.0); gn.add_argument("--lambda_wavkl", type=float, default=0.0)
    gn.add_argument("--lambda_energy_pred", type=float, default=0.0); gn.add_argument("--lambda_adv", type=float, default=0.0)
    gn.add_argument("--stochastic_decoder", action="store_true"); gn.add_argument("--init_log_var", type=float, default=-4.0)
    gn.add_argument("--d_hidden", type=int, default=64); gn.add_argument("--d_dropout", type=float, default=0.1)
    gn.add_argument("--d_lr", type=float, default=2e-4)
    gn.add_argument("--wavkl_scales", default="1,3,7,15"); gn.add_argument("--wavkl_bins", type=int, default=64)
    gn.add_argument("--wavkl_vmax", type=float, default=0.15); gn.add_argument("--wavkl_sigma", type=float, default=0.0)
    gn.add_argument("--wavkl_max_batches", type=int, default=0)
    gn.add_argument("--use_energy_cond", action="store_true"); gn.add_argument("--energy_dim", type=int, default=16)
    gn.add_argument("--energy_desc_scale", type=float, default=0.01)
    gn.add_argument("--use_sentence_budget", action="store_true"); gn.add_argument("--sentence_max_len", type=int, default=32)
    gn.add_argument("--train_manifest_json", default="data/csl-daily/csl_daily_train.json")
    gn.add_argument("--val_frac", type=float, default=0.05); gn.add_argument("--min_val", type=int, default=512)
    gn.add_argument("--val_batches", type=int, default=20); gn.add_argument("--num_workers", type=int, default=2)
    gn.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    gn.add_argument("--seed", type=int, default=0); gn.add_argument("--log_every", type=int, default=200)
    gn.add_argument("--max_steps", type=int, default=0); gn.add_argument("--smoke", type=int, default=0)
    # predictive-contract (makes the carrier objective load-bearing)
    gn.add_argument("--lambda_jepa_pc", type=float, default=0.0,
                    help="Weight on the predictive-contract loss: masked future generated latents "
                         "must be recovered through the frozen carrier predictor.")
    gn.add_argument("--jepa_pc_loss_mode", choices=["cosine", "raw", "both"], default="cosine")
    gn.add_argument("--jepa_pc_raw_weight", type=float, default=1.0)
    gn.add_argument("--jepa_pc_cov_weight", type=float, default=0.0)
    gn.add_argument("--jepa_pc_min_context", type=int, default=16)
    gn.add_argument("--jepa_pc_target_len_min", type=int, default=8)
    gn.add_argument("--jepa_pc_target_len_max", type=int, default=24)

    ge = sub.add_parser("generate")
    ge.add_argument("--ckpt", required=True); ge.add_argument("--split", default="dev")
    ge.add_argument("--teacher", default="native_hclen_b0_budget000"); ge.add_argument("--tag", default="aha")
    ge.add_argument("--sample_temperature", type=float, default=0.0)
    ge.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ge.add_argument("--log_every", type=int, default=200)

    gf = sub.add_parser("gov_fit")
    gf.add_argument("--train_manifest", default="data/csl-daily/csl_daily_train.json")
    gf.add_argument("--train_source", required=True); gf.add_argument("--train_reference", default="gt_hrnet")
    gf.add_argument("--out_policy", required=True); gf.add_argument("--n_hash", type=int, default=512)
    gf.add_argument("--ridge", type=float, default=10.0)
    gf.add_argument("--target_ratio_min", type=float, default=0.75); gf.add_argument("--target_ratio_max", type=float, default=1.80)

    ga = sub.add_parser("gov_apply")
    ga.add_argument("--policy", required=True); ga.add_argument("--manifest_json", required=True)
    ga.add_argument("--split", default="dev"); ga.add_argument("--in_tag", required=True); ga.add_argument("--tag", required=True)
    ga.add_argument("--min_gain", type=float, default=1.0); ga.add_argument("--max_gain", type=float, default=1.55)
    ga.add_argument("--body_coupling", type=float, default=0.35); ga.add_argument("--max_body_gain", type=float, default=1.20)
    ga.add_argument("--lp_kernel", type=int, default=1)

    args = ap.parse_args()
    {"jepa": train_jepa, "gen": train_gen, "generate": generate,
     "gov_fit": gov_fit, "gov_apply": gov_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
