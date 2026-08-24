"""JEPA carrier plus retrieved real hand-motion texture.

This is a nonparametric breakthrough path: keep the old Sign-JEPA pose as the
semantic low-frequency carrier, but borrow high-frequency hand texture from
lexically similar real training clips.  The retrieved texture is high-pass only,
so body, face, and hand low-pass timing stay source locked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

HAND = slice(8, 50)


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def tokens(row: dict) -> list[str]:
    return [g for g in str(row.get("gloss", "")).split() if g]


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 178, 3)
    if x.ndim != 3 or x.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(x.shape)}")
    return x


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def hand_speed(pose: torch.Tensor) -> float:
    h = pose[:, HAND]
    if h.shape[0] < 2:
        return 0.0
    return float((h[1:] - h[:-1]).norm(dim=-1).mean())


def carrier_signature(pose: torch.Tensor, kernel: int = 9, bins: int = 16) -> torch.Tensor:
    lp = temporal_smooth(pose.float(), kernel)
    hand = lp[:, HAND]
    body = lp[:, :8]
    if lp.shape[0] < 2:
        return torch.zeros(4 + bins)
    hv = (hand[1:] - hand[:-1]).norm(dim=-1).mean(dim=-1)
    bv = (body[1:] - body[:-1]).norm(dim=-1).mean(dim=-1)
    hp = resize_seq(hv[:, None], bins).flatten()
    hp = hp / hp.mean().clamp_min(1e-6)
    return torch.cat([
        torch.tensor([
            float(lp.shape[0]) / 100.0,
            float(hv.mean()),
            float(bv.mean()),
            float(hand.std()),
        ]),
        hp,
    ]).float()


def build_index(
    train_manifest: list[dict],
    train_gt: dict[str, torch.Tensor],
    train_source: dict[str, torch.Tensor] | None = None,
    signature_kernel: int = 9,
) -> list[dict]:
    rows = []
    for row in train_manifest:
        sid = str(row["id"])
        if sid not in train_gt:
            continue
        toks = tokens(row)
        if not toks:
            continue
        carrier = train_source.get(sid) if train_source is not None and sid in train_source else train_gt[sid]
        rows.append({
            "id": sid,
            "tokens": toks,
            "tokset": set(toks),
            "length": int(row.get("length", train_gt[sid].shape[0]) or train_gt[sid].shape[0]),
            "carrier_sig": carrier_signature(carrier, signature_kernel),
        })
    return rows


def fit_energy_model(
    train_manifest: list[dict],
    train_gt: dict[str, torch.Tensor],
    train_source: dict[str, torch.Tensor],
    signature_kernel: int,
    ridge: float,
) -> dict[str, torch.Tensor]:
    feats = []
    labels = []
    for row in train_manifest:
        sid = str(row["id"])
        if sid not in train_gt or sid not in train_source:
            continue
        feats.append(carrier_signature(train_source[sid], signature_kernel))
        labels.append(hand_speed(train_gt[sid]))
    if len(feats) < 8:
        raise RuntimeError("not enough paired train-source clips for learned energy")
    X = torch.stack(feats).float()
    y = torch.tensor(labels, dtype=torch.float32)
    mean = X.mean(0)
    std = X.std(0).clamp_min(1e-6)
    Xn = (X - mean) / std
    Xa = torch.cat([Xn, torch.ones(Xn.shape[0], 1)], dim=1)
    eye = torch.eye(Xa.shape[1])
    eye[-1, -1] = 0.0
    coef = torch.linalg.solve(Xa.T @ Xa + ridge * eye, Xa.T @ y)
    return {
        "mean": mean,
        "std": std,
        "coef": coef,
        "min": torch.quantile(y, 0.05),
        "max": torch.quantile(y, 0.95),
    }


def predict_energy(model: dict[str, torch.Tensor], sig: torch.Tensor) -> float:
    x = (sig.float() - model["mean"]) / model["std"]
    xa = torch.cat([x, torch.ones(1)])
    y = torch.dot(xa, model["coef"]).clamp(model["min"], model["max"])
    return float(y)


def retrieve(row: dict, index: list[dict], q_sig: torch.Tensor | None = None, carrier_weight: float = 0.0) -> dict:
    q = set(tokens(row))
    q_len = int(row.get("length", 0) or 0)
    best = None
    best_score = -1e9
    for cand in index:
        inter = len(q & cand["tokset"])
        union = max(1, len(q | cand["tokset"]))
        jacc = inter / union
        len_penalty = abs(q_len - cand["length"]) / max(q_len, cand["length"], 1)
        # Prefer lexical overlap first, but let JEPA carrier dynamics choose
        # between semantically compatible real-motion textures.
        score = 4.0 * jacc + 0.2 * inter - 0.35 * len_penalty
        if q_sig is not None and carrier_weight > 0:
            dist = float(F.mse_loss(q_sig, cand["carrier_sig"]))
            score -= carrier_weight * dist
        if score > best_score:
            best_score = score
            best = cand
    if best is None:
        raise ValueError("empty retrieval index")
    return best


def compose(args: argparse.Namespace) -> None:
    train_manifest = load_manifest(ROOT / args.train_manifest)
    split_manifest = load_manifest(ROOT / args.manifest_json)
    source = load_pose_map(ROOT / args.source_pt)
    train_gt = load_pose_map(ROOT / args.train_gt_pt)
    train_source = load_pose_map(ROOT / args.train_source_pt) if args.train_source_pt else None
    energy_model = None
    if args.learned_energy:
        if train_source is None:
            raise ValueError("--learned_energy requires --train_source_pt")
        energy_model = fit_energy_model(train_manifest, train_gt, train_source, args.signature_kernel, args.energy_ridge)
    index = build_index(train_manifest, train_gt, train_source, args.signature_kernel)
    out = {}
    trace = []
    for row in split_manifest:
        sid = str(row["id"])
        if sid not in source:
            continue
        src = source[sid].float()
        L = int(src.shape[0])
        q_sig = carrier_signature(src, args.signature_kernel) if (args.carrier_weight > 0 or energy_model is not None) else None
        cand = retrieve(row, index, q_sig, args.carrier_weight)
        tex = resize_seq(train_gt[cand["id"]].reshape(train_gt[cand["id"]].shape[0], -1), L).reshape(L, 178, 3)
        src_lp = temporal_smooth(src, args.kernel)
        tex_hp = tex - temporal_smooth(tex, args.kernel)
        pose = src.clone()
        pose[:, HAND] = src_lp[:, HAND] + tex_hp[:, HAND]
        # Preserve a real-motion energy level from the retrieved clip while
        # allowing the JEPA low-pass carrier to define timing.
        cur = max(hand_speed(pose), 1e-8)
        tgt = predict_energy(energy_model, q_sig) if energy_model is not None and q_sig is not None else hand_speed(tex)
        tgt = max(tgt, 1e-8)
        hp = pose[:, HAND] - src_lp[:, HAND]
        scale = float(np.clip(tgt / cur, args.min_texture_scale, args.max_texture_scale))
        pose[:, HAND] = src_lp[:, HAND] + scale * hp
        if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
            pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
        out[sid] = pose.contiguous()
        trace.append({
            "id": sid,
            "retrieved_id": cand["id"],
            "retrieved_tokens": cand["tokens"],
            "source_len": L,
            "retrieved_len": cand["length"],
            "texture_scale": scale,
        })
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if args.trace_json:
        trace_path = ROOT / args.trace_json
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2))
        print(f"saved trace -> {trace_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_pt", required=True)
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    ap.add_argument("--train_gt_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--train_source_pt", default="")
    ap.add_argument("--carrier_weight", type=float, default=0.0)
    ap.add_argument("--signature_kernel", type=int, default=9)
    ap.add_argument("--learned_energy", action="store_true")
    ap.add_argument("--energy_ridge", type=float, default=1e-2)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--min_texture_scale", type=float, default=0.25)
    ap.add_argument("--max_texture_scale", type=float, default=4.0)
    ap.add_argument("--post_smooth_kernel", type=int, default=3)
    ap.add_argument("--post_smooth_blend", type=float, default=0.05)
    args = ap.parse_args()
    compose(args)


if __name__ == "__main__":
    main()
