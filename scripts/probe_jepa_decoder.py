"""Decoder-isolation probe (Step 1 of the Sign-JEPA collapse fix).

Trains ONLY the pose decoder on (frozen JEPA target latent z_tgt) -> pose, with
velocity + acceleration + speed-envelope + std-match losses ON. Question it
answers: can a decoder render full-speed, smooth motion from the JEPA target
latent?

  - val decoder(z_tgt) hand-speed ratio -> ~1.0x GT with hand-jerk ratio ~1.0-1.5x
        => the JEPA latent is rich enough; build a latent-FM generator + this decoder.
  - speed plateaus ~0.5x, or jerk cannot fall below ~2x
        => the Stage-1 JEPA latent is itself lossy; rework Stage-1 first.

Baseline for reference: the existing generator's decoder, evaluated on z_tgt,
gives a hand-speed ratio of 0.37x (Step-0b diagnostic). The first 30-epoch run
recovered speed (0.37x -> 1.04x) but jerk plateaued at 2.4x because the loss had
no smoothness term; this version adds an acceleration-matching loss.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_generator_slrtp178 import (  # noqa: E402
    SignJEPAGenerator,
    encode_target_latents,
    load_frozen_jepa,
    speed_envelope_loss,
    std_match_loss,
    velocity_loss,
)
from scripts.train_sign_jepa_slrtp178 import SLRTPSpanDataset, load_bank  # noqa: E402


def region_speed(raw: torch.Tensor) -> dict[str, float]:
    """raw: [B, T, 178, 3] -> mean per-joint speed for body / hands / face."""
    v = raw[:, 1:] - raw[:, :-1]
    return {
        "hand": float(v[:, :, 8:50].norm(dim=-1).mean()),
        "body": float(v[:, :, 0:8].norm(dim=-1).mean()),
        "face": float(v[:, :, 50:178].norm(dim=-1).mean()),
    }


def hand_jerk(raw: torch.Tensor) -> float:
    v = raw[:, 1:] - raw[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]
    return float(j[:, :, 8:50].norm(dim=-1).mean())


def accel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match discrete acceleration (2nd difference) -> directly controls jerk."""
    pa = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    ta = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return F.smooth_l1_loss(pa, ta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", default="outputs/sota_chase/sign_jepa_slrtp178/best.pt")
    ap.add_argument("--gen_ckpt", default="outputs/sota_chase/sign_jepa_generator_slrtp178/best.pt")
    ap.add_argument("--bank", default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    ap.add_argument("--out_dir", default="outputs/sota_chase/jepa_decoder_probe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_pose", type=float, default=1.0)
    ap.add_argument("--lambda_vel", type=float, default=2.0)
    ap.add_argument("--lambda_accel", type=float, default=2.0)
    ap.add_argument("--lambda_speed", type=float, default=0.5)
    ap.add_argument("--lambda_std", type=float, default=0.5)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--from_scratch", action="store_true",
                    help="Re-init the decoder instead of warm-starting from the generator.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jepa, _ = load_frozen_jepa(ROOT / args.jepa_ckpt, device)
    g = torch.load(ROOT / args.gen_ckpt, map_location="cpu", weights_only=False)
    gen = SignJEPAGenerator(
        num_gloss=len(g["gloss_to_id"]),
        hidden=g["hidden"],
        gen_layers=g["args"]["gen_layers"],
        dec_layers=g["args"]["dec_layers"],
        heads=g["args"]["heads"],
        dropout=0.05,
        max_len=g["seg_len"],
        style_dim=g["args"].get("style_dim", 0),
    )
    gen.load_state_dict(g["model"])
    decoder = gen.decoder.to(device)
    if args.from_scratch:
        for m in decoder.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    mean = g["mean"].float().to(device)
    std = g["std"].float().to(device)
    seg_len = int(g["seg_len"])
    gloss_to_id = g["gloss_to_id"]

    bank, spans = load_bank(ROOT / args.bank)
    random.Random(args.seed).shuffle(spans)
    val_spans = spans[: args.n_val]
    train_spans = spans[args.n_val:]

    def make_loader(sp, shuffle, drop_last):
        ds = SLRTPSpanDataset(bank, sp, gloss_to_id, g["mean"].float(), g["std"].float(), seg_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=drop_last)

    train_loader = make_loader(train_spans, True, True)
    val_loader = make_loader(val_spans, False, False)

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[probe] device={device} train={len(train_spans)} val={len(val_spans)} "
          f"seg_len={seg_len} decoder_params={sum(p.numel() for p in decoder.parameters())/1e6:.2f}M "
          f"warm_start={not args.from_scratch} "
          f"lambdas(pose/vel/accel/speed/std)="
          f"{args.lambda_pose}/{args.lambda_vel}/{args.lambda_accel}/{args.lambda_speed}/{args.lambda_std}",
          flush=True)

    def encode(batch):
        pose = batch["pose"].to(device)
        pose_raw = batch["pose_raw"].to(device)
        gid = batch["gloss_id"].to(device)
        lid = batch["len_id"].to(device)
        z, _ = encode_target_latents(jepa, pose, pose_raw, gid, lid)
        return z, pose

    @torch.no_grad()
    def evaluate(epoch: int) -> dict:
        decoder.eval()
        acc = {f"{k}_pred": [] for k in ("hand", "body", "face")}
        acc.update({f"{k}_real": [] for k in ("hand", "body", "face")})
        jk_pred, jk_real, ps_pred, ps_real = [], [], [], []
        for batch in val_loader:
            z, pose = encode(batch)
            pred = decoder(z)
            praw = (pred * std + mean).reshape(pred.shape[0], pred.shape[1], 178, 3)
            traw = (pose * std + mean).reshape(pose.shape[0], pose.shape[1], 178, 3)
            rsp, rst = region_speed(praw), region_speed(traw)
            for k in ("hand", "body", "face"):
                acc[f"{k}_pred"].append(rsp[k])
                acc[f"{k}_real"].append(rst[k])
            jk_pred.append(hand_jerk(praw))
            jk_real.append(hand_jerk(traw))
            ps_pred.append(float(praw[:, :, 8:50].std()))
            ps_real.append(float(traw[:, :, 8:50].std()))
        ratio = lambda a, b: round(float(np.mean(a) / max(np.mean(b), 1e-9)), 3)
        return {
            "epoch": epoch,
            "hand_speed_ratio": ratio(acc["hand_pred"], acc["hand_real"]),
            "body_speed_ratio": ratio(acc["body_pred"], acc["body_real"]),
            "face_speed_ratio": ratio(acc["face_pred"], acc["face_real"]),
            "hand_jerk_ratio": ratio(jk_pred, jk_real),
            "hand_posestd_ratio": ratio(ps_pred, ps_real),
        }

    log = [{**evaluate(0), "note": "warm-start baseline (should ~match Step-0b 0.37x)"}]
    print(json.dumps(log[-1]), flush=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        decoder.train()
        tl = []
        for batch in train_loader:
            z, pose = encode(batch)
            pred = decoder(z)
            loss = (
                args.lambda_pose * F.smooth_l1_loss(pred, pose)
                + args.lambda_vel * velocity_loss(pred, pose)
                + args.lambda_accel * accel_loss(pred, pose)
                + args.lambda_speed * speed_envelope_loss(pred, pose)
                + args.lambda_std * std_match_loss(pred, pose)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach().cpu()))
        rec = {**evaluate(ep), "wall_s": round(time.time() - t0, 1),
               "train_loss": round(float(np.mean(tl)), 5)}
        log.append(rec)
        (out_dir / "probe_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)

    torch.save({"decoder": decoder.state_dict(), "args": vars(args), "log": log},
               out_dir / "decoder_probe.pt")
    final = log[-1]["hand_speed_ratio"]
    jerk = log[-1]["hand_jerk_ratio"]
    if final >= 0.85 and jerk <= 1.6:
        verdict = "JEPA LATENT IS RICH -> build latent-FM generator on top of this decoder"
    elif final < 0.6:
        verdict = "JEPA LATENT LOOKS LOSSY -> rework Stage-1 JEPA before the generator"
    else:
        verdict = "PARTIAL recovery -> inspect (latent FM may still help; consider Stage-1 tweak)"
    print(f"[probe] baseline old-decoder(z_tgt) hand-speed ratio = 0.37x", flush=True)
    print(f"[probe] VERDICT: final decoder(z_tgt) hand-speed ratio = {final}x, "
          f"hand-jerk ratio = {jerk}x -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
