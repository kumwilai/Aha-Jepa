"""v32 Adversarial Conditional Flow Matching in JEPA Latent Space (ACFM-L).

Trains a velocity field v_theta(z_t, t, gloss_ctx) that pushes a frozen
v11 generator's deterministic latent anchor z_0 toward the JEPA target
encoder's latent z_1 of GT pose, supervised by both:
- flow-matching regression to the conditional OT velocity (z_1 - z_0)
- adversarial loss at the decoded one-step prediction x_1_hat

At inference: K-step Euler ODE integration from the v11 anchor z_0
through the velocity field to z_1, then decode with frozen v11 decoder.

What this is NOT:
- Pose-space FM (v55, v100): those operated on raw pose, ran into
  multimodal-trajectory averaging.
- Standard noise-to-data FM: x_0 is the v11 deterministic anchor (text-
  conditioned), not Gaussian noise.
- Pure adversarial (v29): single-shot G has limited capacity; FM
  multi-step refines incrementally.

What this IS:
- Latent-space FM with semantically meaningful endpoints at both ends.
- Combined FM regression + adversarial supervision at the data manifold.
- Multi-step ODE inference (text-only, no GT length, no retrieval).

Constraints honored:
- No GT length at inference (uses text-only duration policy via the v11
  sample_manifest path).
- No retrieval, no real-clip splicing, no SLRTP BT evaluator distillation.
- Single principled config per run (no dev/test sweep).
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
    SignJEPAModel,
    SLRTPSpanDataset,
    build_gloss_vocab,
    length_bucket_id,
    load_bank,
    resize_seq,
    sinusoidal_positions,
)
from scripts.train_sign_jepa_generator_slrtp178 import (  # noqa: E402
    SignJEPAGenerator,
    MotionDiscriminator,
    encode_target_latents,
    load_frozen_jepa,
    mean_lengths_by_gloss,
    energy_budget_descriptor,
    sentence_tensors_from_glosses,
    allocate_lengths,
    load_duration_policy,
    predict_duration_total_len,
)


# ---------------------------------------------------------------------------
# Velocity field architecture
# ---------------------------------------------------------------------------


class FourierTimeEmbed(nn.Module):
    def __init__(self, hidden: int, num_freqs: int = 32):
        super().__init__()
        # Random frozen Fourier features for time (standard FM/diffusion trick)
        self.register_buffer("freqs", torch.randn(num_freqs) * 16.0)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1]
        ang = t[:, None] * self.freqs[None] * 2 * math.pi
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.proj(emb)


class VelocityField(nn.Module):
    """v32 velocity network. Inputs (z_t, t, gloss_id, len_id, sent_ctx)."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden: int = 256,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        gloss_vocab: int = 1083,
        sent_max_len: int = 32,
    ):
        super().__init__()
        self.hidden = hidden
        self.z_in = nn.Linear(latent_dim, hidden)
        self.t_embed = FourierTimeEmbed(hidden)
        self.gloss_emb = nn.Embedding(gloss_vocab, hidden, padding_idx=0)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.sent_emb = nn.Embedding(gloss_vocab, hidden, padding_idx=0)
        self.sent_pos = nn.Embedding(sent_max_len, hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, latent_dim)
        # Zero-init head so untrained velocity field starts at identity (v=0).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
        sent_ids: torch.Tensor,
        sent_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = z_t.shape
        h = self.z_in(z_t)
        h = h + self.t_embed(t)[:, None]
        h = h + self.gloss_emb(gloss_id)[:, None]
        h = h + self.len_emb(len_id)[:, None]
        # Sentence-mean context
        S = sent_ids.shape[1]
        sent_pos = torch.arange(S, device=z_t.device)
        sent_h = self.sent_emb(sent_ids) + self.sent_pos(sent_pos)[None]
        denom = sent_mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        sent_ctx = (sent_h * sent_mask.float()[..., None]).sum(1) / denom
        h = h + sent_ctx[:, None]
        # Positional encoding over T
        h = h + sinusoidal_positions(T, self.hidden, z_t.device)[None]
        z = self.norm(self.blocks(h))
        return self.head(z)


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def sample_t(B: int, device: torch.device, mode: str = "uniform") -> torch.Tensor:
    """Sample t for FM training. 'logit_normal' emphasises mid-path which is
    standard practice in FM (Stable Diffusion 3, EDM). 'uniform' is simpler."""
    if mode == "logit_normal":
        u = torch.randn(B, device=device)
        return torch.sigmoid(u)
    return torch.rand(B, device=device)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frozen v11 generator + JEPA encoder.
    jepa, jepa_ckpt = load_frozen_jepa(ROOT / args.jepa_ckpt, device)
    bank, spans = load_bank(ROOT / args.bank)
    gloss_to_id = jepa_ckpt["gloss_to_id"]
    mean = jepa_ckpt["mean"].float()
    std = jepa_ckpt["std"].float()
    gloss_mean_lengths = mean_lengths_by_gloss(spans)
    seg_len = int(jepa_ckpt["args"]["seg_len"])

    # Load v11 generator (frozen, used for anchor z_0 and decoding z_1 -> pose)
    v11_ckpt = torch.load(ROOT / args.v11_ckpt, map_location=device, weights_only=False)
    v11 = SignJEPAGenerator(
        num_gloss=len(gloss_to_id),
        hidden=v11_ckpt["hidden"],
        gen_layers=v11_ckpt["args"]["gen_layers"],
        dec_layers=v11_ckpt["args"]["dec_layers"],
        heads=v11_ckpt["args"]["heads"],
        dropout=0.0,
        max_len=v11_ckpt["seg_len"],
        style_dim=v11_ckpt["args"].get("style_dim", 0),
        use_energy_cond=v11_ckpt["args"].get("use_energy_cond", False),
        energy_dim=v11_ckpt["args"].get("energy_dim", 16),
        use_sentence_budget=v11_ckpt["args"].get("use_sentence_budget", False),
        sentence_max_len=v11_ckpt["args"].get("sentence_max_len", 32),
        stochastic_decoder=v11_ckpt["args"].get("stochastic_decoder", False),
        init_log_var=v11_ckpt["args"].get("init_log_var", -4.0),
        latent_noise_std=v11_ckpt["args"].get("latent_noise_std", 0.0),
    ).to(device)
    v11.load_state_dict(v11_ckpt["model"])
    v11.eval()
    for p in v11.parameters():
        p.requires_grad_(False)
    print(f"[v32] frozen v11 from {args.v11_ckpt} "
          f"params={sum(p.numel() for p in v11.parameters())/1e6:.2f}M", flush=True)

    # Velocity network (the only big trainable module).
    latent_dim = jepa_ckpt["args"]["hidden"]
    velocity_net = VelocityField(
        latent_dim=latent_dim,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        gloss_vocab=len(gloss_to_id),
        sent_max_len=args.sent_max_len,
    ).to(device)
    print(f"[v32] velocity_net params="
          f"{sum(p.numel() for p in velocity_net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(velocity_net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Optional adversarial discriminator (v29 reuse).
    discriminator = None
    d_opt = None
    if args.lambda_adv > 0:
        discriminator = MotionDiscriminator(hidden=args.d_hidden, dropout=args.d_dropout).to(device)
        d_opt = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr,
                                  weight_decay=0.0, betas=(0.0, 0.99))
        print(f"[v32] discriminator params="
              f"{sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M "
              f"lambda_adv={args.lambda_adv}", flush=True)

    # Dataset: same as v11's SLRTPSpanDataset
    rnd = random.Random(args.seed)
    rnd.shuffle(spans)
    n_val = max(args.min_val, int(len(spans) * args.val_frac))
    val_spans = spans[:n_val]
    train_spans = spans[n_val:]
    if args.smoke:
        train_spans = train_spans[: args.smoke_train]
        val_spans = val_spans[: args.smoke_val]

    row_by_sid = None
    if args.use_sentence_budget and args.train_manifest_json:
        rows = json.load(open(ROOT / args.train_manifest_json))
        row_by_sid = {str(r["id"]): r for r in rows}

    train_ds = SLRTPSpanDataset(
        bank, train_spans, gloss_to_id, mean, std, seg_len,
        max_items=args.max_train_items, seed=args.seed,
        row_by_sid=row_by_sid, sent_max_len=args.sent_max_len,
    )
    val_ds = SLRTPSpanDataset(
        bank, val_spans, gloss_to_id, mean, std, seg_len,
        max_items=args.max_val_items, seed=args.seed + 1,
        row_by_sid=row_by_sid, sent_max_len=args.sent_max_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        velocity_net.train()
        if discriminator is not None:
            discriminator.train()
        train_accum: dict[str, float] = {}
        n_train = 0
        for step, batch in enumerate(train_loader, start=1):
            pose_norm = batch["pose"].to(device)              # [B, T, 534]
            pose_raw = batch["pose_raw"].to(device)
            gloss_id = batch["gloss_id"].to(device)
            len_id = batch["len_id"].to(device)
            sent_ids = batch["sent_gloss_ids"].to(device)
            sent_mask = batch["sent_mask"].to(device)
            B, T, _ = pose_norm.shape

            with torch.no_grad():
                # z_anchor = v11 generator's latent (deterministic, frozen)
                energy = None
                if v11.use_energy_cond:
                    if v11.use_sentence_budget:
                        sent_pos = batch.get(
                            "sent_pos", torch.zeros(B, dtype=torch.long)
                        ).to(device)
                        energy = v11.predict_condition_energy(
                            gloss_id, len_id, sent_ids, sent_mask, sent_pos
                        )
                    else:
                        energy = v11.predict_energy(gloss_id, len_id)
                    energy = energy.detach().clamp_min(0.0)
                z_anchor = v11.generator(gloss_id, len_id, T, style=None, energy=energy)
                # z_gt from JEPA target encoder of GT pose
                z_gt, _ = encode_target_latents(jepa, pose_norm, pose_raw, gloss_id, len_id)

            # FM: interpolate at random t
            t = sample_t(B, device, mode=args.t_sample_mode)
            t_view = t.view(B, 1, 1)
            z_t = (1.0 - t_view) * z_anchor + t_view * z_gt

            v_target = z_gt - z_anchor                          # [B, T, D]
            v_pred = velocity_net(z_t, t, gloss_id, len_id, sent_ids, sent_mask)
            L_fm = F.smooth_l1_loss(v_pred, v_target)

            # One-step extrapolation to x_1 prediction
            z_1_hat = z_t + (1.0 - t_view) * v_pred

            # Decode through frozen v11 pose decoder
            pose_hat_norm = v11.decoder(z_1_hat)               # [B, T, 534]
            pose_hat_raw = pose_hat_norm * std.to(device) + mean.to(device)
            L_pose = F.smooth_l1_loss(pose_hat_norm, pose_norm)

            # Adversarial step
            adv_g = pose_norm.new_zeros(())
            adv_d = pose_norm.new_zeros(())
            d_real_mean = 0.0
            d_fake_mean = 0.0
            if discriminator is not None and args.lambda_adv > 0:
                pose_hat_raw_d = pose_hat_raw.detach().reshape(B, T, 178, 3)
                pose_real_d = pose_raw.reshape(B, T, 178, 3)
                logits_real = discriminator(pose_real_d)
                logits_fake = discriminator(pose_hat_raw_d)
                d_loss = F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()
                adv_d = d_loss.detach()
                d_real_mean = float(logits_real.detach().mean().cpu())
                d_fake_mean = float(logits_fake.detach().mean().cpu())
                pose_hat_raw_g = pose_hat_raw.reshape(B, T, 178, 3)
                adv_g = -discriminator(pose_hat_raw_g).mean()

            loss = (
                args.lambda_fm * L_fm
                + args.lambda_pose * L_pose
                + args.lambda_adv * adv_g
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), args.grad_clip)
            opt.step()

            logs = {
                "loss": float(loss.detach().cpu()),
                "L_fm": float(L_fm.detach().cpu()),
                "L_pose": float(L_pose.detach().cpu()),
                "adv_g": float(adv_g.detach().cpu()) if torch.is_tensor(adv_g) else float(adv_g),
                "adv_d": float(adv_d.detach().cpu()) if torch.is_tensor(adv_d) else float(adv_d),
                "d_real": d_real_mean,
                "d_fake": d_fake_mean,
            }
            for k, v in logs.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            n_train += 1
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "grad_norm": float(gnorm), **logs}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        train_log = {f"train_{k}": v / max(n_train, 1) for k, v in train_accum.items()}

        # Validation
        velocity_net.eval()
        if discriminator is not None:
            discriminator.eval()
        val_accum: dict[str, float] = {}
        n_val_b = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.val_batches and bi >= args.val_batches:
                    break
                pose_norm = batch["pose"].to(device)
                pose_raw = batch["pose_raw"].to(device)
                gloss_id = batch["gloss_id"].to(device)
                len_id = batch["len_id"].to(device)
                sent_ids = batch["sent_gloss_ids"].to(device)
                sent_mask = batch["sent_mask"].to(device)
                B, T, _ = pose_norm.shape
                energy = None
                if v11.use_energy_cond:
                    if v11.use_sentence_budget:
                        sent_pos = batch.get(
                            "sent_pos", torch.zeros(B, dtype=torch.long)
                        ).to(device)
                        energy = v11.predict_condition_energy(
                            gloss_id, len_id, sent_ids, sent_mask, sent_pos
                        )
                    else:
                        energy = v11.predict_energy(gloss_id, len_id)
                    energy = energy.detach().clamp_min(0.0)
                z_anchor = v11.generator(gloss_id, len_id, T, style=None, energy=energy)
                z_gt, _ = encode_target_latents(jepa, pose_norm, pose_raw, gloss_id, len_id)
                # Multi-step Euler integration from z_anchor → z_1
                z_t = z_anchor.clone()
                dt = 1.0 / args.n_steps_val
                for k in range(args.n_steps_val):
                    t = torch.full((B,), k * dt, device=device)
                    v = velocity_net(z_t, t, gloss_id, len_id, sent_ids, sent_mask)
                    z_t = z_t + dt * v
                pose_hat = v11.decoder(z_t)
                pose_hat_raw = pose_hat * std.to(device) + mean.to(device)
                # Metrics
                val_pose = F.smooth_l1_loss(pose_hat, pose_norm)
                pred_v = (pose_hat_raw[:, 1:] - pose_hat_raw[:, :-1]).reshape(B, T-1, 178, 3)
                gt_v = (pose_raw[:, 1:] - pose_raw[:, :-1]).reshape(B, T-1, 178, 3)
                hs_ratio = (pred_v[:, :, 8:50].norm(-1).mean() / gt_v[:, :, 8:50].norm(-1).mean().clamp_min(1e-9)).item()
                jerk_p = (pred_v[:, 1:] - pred_v[:, :-1])[:, :, 8:50].norm(-1).mean()
                jerk_g = (gt_v[:, 1:] - gt_v[:, :-1])[:, :, 8:50].norm(-1).mean().clamp_min(1e-9)
                jerk_ratio = (jerk_p / jerk_g).item()
                vlogs = {
                    "val_pose": float(val_pose.cpu()),
                    "val_hand_speed_ratio": hs_ratio,
                    "val_hand_jerk_ratio": jerk_ratio,
                }
                for k, v in vlogs.items():
                    val_accum[k] = val_accum.get(k, 0.0) + v
                n_val_b += 1
        val_log = {k: v / max(n_val_b, 1) for k, v in val_accum.items()}
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **train_log, **val_log}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)

        score = rec["val_pose"]  # primary selection
        if best is None or score < best["score"]:
            best = {**rec, "score": float(score)}
            torch.save(
                {
                    "model": velocity_net.state_dict(),
                    "discriminator": discriminator.state_dict() if discriminator is not None else None,
                    "args": vars(args),
                    "best": best,
                    "v11_ckpt": str(args.v11_ckpt),
                    "jepa_ckpt": str(args.jepa_ckpt),
                    "gloss_to_id": gloss_to_id,
                    "mean": mean,
                    "std": std,
                    "seg_len": seg_len,
                    "latent_dim": latent_dim,
                    "gloss_mean_lengths": gloss_mean_lengths,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[v32] saved best -> {out_dir / 'best.pt'}", flush=True)

    print(f"[v32] done best={best}", flush=True)


# ---------------------------------------------------------------------------
# Inference / sample_manifest
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_manifest(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.ckpt, map_location=device, weights_only=False)
    gloss_to_id = ck["gloss_to_id"]
    seg_len = int(ck["seg_len"])
    mean = ck["mean"].to(device)
    std = ck["std"].to(device)
    gloss_mean_lengths = ck.get("gloss_mean_lengths", {})
    sent_max_len = int(ck["args"].get("sent_max_len", 32))

    # Load JEPA encoder (for nothing at inference, but needed if other paths use it)
    jepa_ckpt = torch.load(ROOT / ck["jepa_ckpt"], map_location=device, weights_only=False)
    latent_dim = ck["latent_dim"]

    # v11 generator (frozen)
    v11_ckpt = torch.load(ROOT / ck["v11_ckpt"], map_location=device, weights_only=False)
    v11 = SignJEPAGenerator(
        num_gloss=len(gloss_to_id),
        hidden=v11_ckpt["hidden"],
        gen_layers=v11_ckpt["args"]["gen_layers"],
        dec_layers=v11_ckpt["args"]["dec_layers"],
        heads=v11_ckpt["args"]["heads"],
        dropout=0.0,
        max_len=v11_ckpt["seg_len"],
        style_dim=v11_ckpt["args"].get("style_dim", 0),
        use_energy_cond=v11_ckpt["args"].get("use_energy_cond", False),
        energy_dim=v11_ckpt["args"].get("energy_dim", 16),
        use_sentence_budget=v11_ckpt["args"].get("use_sentence_budget", False),
        sentence_max_len=v11_ckpt["args"].get("sentence_max_len", 32),
        stochastic_decoder=v11_ckpt["args"].get("stochastic_decoder", False),
        init_log_var=v11_ckpt["args"].get("init_log_var", -4.0),
        latent_noise_std=v11_ckpt["args"].get("latent_noise_std", 0.0),
    ).to(device)
    v11.load_state_dict(v11_ckpt["model"])
    v11.eval()

    # Velocity field
    velocity_net = VelocityField(
        latent_dim=latent_dim,
        hidden=ck["args"]["hidden"],
        layers=ck["args"]["layers"],
        heads=ck["args"]["heads"],
        dropout=0.0,
        gloss_vocab=len(gloss_to_id),
        sent_max_len=sent_max_len,
    ).to(device)
    velocity_net.load_state_dict(ck["model"])
    velocity_net.eval()

    rows = json.load(open(ROOT / args.manifest_json))
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
        ref_keys = set(str(k) for k in ref.keys())
        rows = [r for r in rows if str(r["id"]) in ref_keys]

    duration_policy = load_duration_policy(args.duration_policy)
    if not duration_policy or duration_policy.get("length_model") is None:
        raise ValueError(
            "v32 sample_manifest requires a duration policy with a text-only "
            "length_model. The rebuilt train_bt_rf_policy.joblib provides this."
        )

    out: dict[str, torch.Tensor] = {}
    for i, row in enumerate(rows, start=1):
        sid = str(row["id"])
        glosses = [str(g) for g in str(row.get("gloss", "")).split() if g]
        if not glosses:
            continue
        # Predict total T from text-only features
        total_len = predict_duration_total_len(row, duration_policy)
        if total_len is None or total_len <= 0:
            continue
        lengths = allocate_lengths(glosses, total_len, gloss_mean_lengths)
        sent_ids = torch.zeros(1, sent_max_len, dtype=torch.long, device=device)
        sent_mask = torch.zeros(1, sent_max_len, dtype=torch.bool, device=device)
        for j, g in enumerate(glosses[:sent_max_len]):
            sent_ids[0, j] = int(gloss_to_id.get(g, 1))
            sent_mask[0, j] = True

        chunks = []
        for pos_i, (gloss, L) in enumerate(zip(glosses, lengths)):
            gid = torch.tensor([gloss_to_id.get(gloss, 1)], dtype=torch.long, device=device)
            lid = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
            energy = None
            if v11.use_energy_cond:
                if v11.use_sentence_budget:
                    spos = torch.tensor([min(pos_i, sent_max_len - 1)], dtype=torch.long, device=device)
                    energy = v11.predict_condition_energy(gid, lid, sent_ids, sent_mask, spos)
                else:
                    energy = v11.predict_energy(gid, lid)
                energy = energy.detach().clamp_min(0.0)
            # z_anchor at seg_len, then ODE integrate
            z_t = v11.generator(gid, lid, seg_len, style=None, energy=energy)
            n_steps = int(args.n_steps)
            dt = 1.0 / n_steps
            for k in range(n_steps):
                t = torch.full((1,), k * dt, device=device)
                v = velocity_net(z_t, t, gid, lid, sent_ids, sent_mask)
                z_t = z_t + dt * v
            pose = v11.decoder(z_t)
            pose = pose[0] * std + mean
            pose = resize_seq(pose.cpu(), L).reshape(L, 178, 3).float()
            chunks.append(pose)
        out[sid] = torch.cat(chunks, dim=0).contiguous()
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[v32-sample] {i}/{len(rows)}", flush=True)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    lens = np.asarray([v.shape[0] for v in out.values()])
    print(f"saved -> {out_path}", flush=True)
    print(f"clips={len(out)} T mean={lens.mean():.1f} "
          f"p10/50/90={np.percentile(lens, [10, 50, 90]).tolist()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--jepa_ckpt", default="outputs/sota_chase/sign_jepa_slrtp178/best.pt")
    tr.add_argument("--bank", default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    tr.add_argument("--v11_ckpt", default="outputs/sota_chase/sign_jepa_v25_stochastic/best.pt",
                    help="Frozen v11-class generator providing z_anchor and pose decoder.")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_v32_acfm")
    tr.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")

    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--layers", type=int, default=4)
    tr.add_argument("--heads", type=int, default=8)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--sent_max_len", type=int, default=32)
    tr.add_argument("--use_sentence_budget", action="store_true", default=True)

    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)

    tr.add_argument("--lambda_fm", type=float, default=1.0)
    tr.add_argument("--lambda_pose", type=float, default=0.2,
                    help="Auxiliary L1 on decoded x_1_hat vs GT pose. Stabilizes early.")
    tr.add_argument("--lambda_adv", type=float, default=0.05)
    tr.add_argument("--d_lr", type=float, default=2e-4)
    tr.add_argument("--d_hidden", type=int, default=64)
    tr.add_argument("--d_dropout", type=float, default=0.1)

    tr.add_argument("--t_sample_mode", choices=["uniform", "logit_normal"], default="logit_normal")
    tr.add_argument("--n_steps_val", type=int, default=4)

    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--min_val", type=int, default=512)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--max_train_items", type=int, default=0)
    tr.add_argument("--max_val_items", type=int, default=2048)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=256)
    tr.add_argument("--smoke_val", type=int, default=96)

    sm = sub.add_parser("sample_manifest")
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--manifest_json", required=True)
    sm.add_argument("--out_pt", required=True)
    sm.add_argument("--reference_pt", default="")
    sm.add_argument("--duration_policy", required=True)
    sm.add_argument("--n_steps", type=int, default=4)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
