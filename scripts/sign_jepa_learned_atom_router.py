"""Learned gloss-atom router for Sign-JEPA motion texture.

This keeps the algorithm source-locked, but replaces hand-picked atom selection
with a train-only learned compatibility model.

Training signal:
  for each train gloss occurrence, candidate atoms for that gloss are scored by
  how well their high-frequency hand texture reconstructs the real occurrence.

Inference:
  select the atom with highest predicted compatibility from JEPA carrier
  segment features + candidate atom features, then overlap-add onto JEPA.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_gloss_texture_memory import (  # noqa: E402
    HAND,
    HAND_DIM,
    as_pose,
    blend_weight,
    hand_speed_from_flat,
    hand_speed_pose,
    load_manifest,
    load_pose_map,
    source_guided_bounds,
    split_bounds,
    tokens,
)
from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402


def segment_features(src: torch.Tensor, a: int, b: int, gloss_idx: int, gloss_count: int) -> torch.Tensor:
    seg = src[a:b]
    h = seg[:, HAND].reshape(seg.shape[0], HAND_DIM)
    if h.shape[0] < 2:
        speed = accel = 0.0
    else:
        v = h[1:] - h[:-1]
        speed = float(v.reshape(v.shape[0], 42, 3).norm(dim=-1).mean())
        accel = float((v[1:] - v[:-1]).abs().mean()) if v.shape[0] > 1 else 0.0
    T = max(1, src.shape[0])
    return torch.tensor([
        (b - a) / T,
        a / T,
        b / T,
        gloss_idx / max(1, gloss_count - 1),
        speed,
        accel,
        float(h.std()),
    ], dtype=torch.float32)


def atom_features(atom: dict, target_len: int) -> torch.Tensor:
    src_len = max(1, int(atom["source_len"]))
    speed = float(atom["speed"])
    std = float(atom["atom"].std())
    return torch.tensor([
        src_len / max(1, target_len),
        target_len / src_len,
        abs(src_len - target_len) / max(src_len, target_len, 1),
        speed,
        std,
    ], dtype=torch.float32)


def pair_features(src: torch.Tensor, a: int, b: int, gloss_idx: int, gloss_count: int, atom: dict) -> torch.Tensor:
    return torch.cat([segment_features(src, a, b, gloss_idx, gloss_count), atom_features(atom, b - a)], dim=0)


def build_atoms(train_manifest: list[dict], train_gt: dict[str, torch.Tensor], kernel: int, atom_len: int, max_atoms: int) -> dict:
    by_gloss = defaultdict(list)
    for row in train_manifest:
        sid = str(row["id"])
        if sid not in train_gt:
            continue
        glosses = tokens(row)
        if not glosses:
            continue
        pose = train_gt[sid].float()
        hp = pose - temporal_smooth(pose, kernel)
        for gi, (gloss, (a, b)) in enumerate(zip(glosses, split_bounds(pose.shape[0], len(glosses)))):
            seg = hp[a:b, HAND].reshape(b - a, HAND_DIM)
            if seg.shape[0] < 2:
                continue
            atom = resize_seq(seg, atom_len).reshape(atom_len, HAND_DIM).contiguous()
            by_gloss[gloss].append({
                "source_id": sid,
                "source_len": int(b - a),
                "atom": atom,
                "speed": hand_speed_from_flat(atom),
                "gloss_index": gi,
            })
    memory = {}
    for gloss, atoms in by_gloss.items():
        atoms.sort(key=lambda x: x["speed"], reverse=True)
        memory[gloss] = atoms[:max_atoms]
    return memory


class Router(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_training_pairs(args, train_manifest, train_gt, train_source, memory) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(args.seed)
    rows = list(train_manifest)
    rng.shuffle(rows)
    if args.max_train_clips:
        rows = rows[: args.max_train_clips]
    feats = []
    labels = []
    for row in rows:
        sid = str(row["id"])
        if sid not in train_gt or sid not in train_source:
            continue
        glosses = tokens(row)
        if not glosses:
            continue
        src = as_pose(train_source[sid]).float()
        gt = resize_seq(train_gt[sid].reshape(train_gt[sid].shape[0], -1), src.shape[0]).reshape(src.shape[0], 178, 3)
        gt_hp = gt - temporal_smooth(gt, args.kernel)
        bounds = source_guided_bounds(src, len(glosses), args.boundary_window_frac)
        for gi, (gloss, (a, b)) in enumerate(zip(glosses, bounds)):
            atoms = [x for x in memory.get(gloss, []) if x["source_id"] != sid]
            if not atoms:
                continue
            # Candidate pool: energetic atoms plus a few random alternatives.
            cand = atoms[: args.candidates_per_gloss]
            if len(atoms) > args.candidates_per_gloss:
                cand = cand + rng.sample(atoms[args.candidates_per_gloss :], min(args.random_candidates, len(atoms) - args.candidates_per_gloss))
            target = gt_hp[a:b, HAND].reshape(b - a, HAND_DIM)
            for atom in cand:
                seg = resize_seq(atom["atom"], b - a).reshape(b - a, HAND_DIM)
                score = -float(F.smooth_l1_loss(seg, target))
                feats.append(pair_features(src, a, b, gi, len(glosses), atom))
                labels.append(score)
    if not feats:
        raise RuntimeError("no training pairs built")
    X = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.float32)
    y = (y - y.mean()) / y.std().clamp_min(1e-6)
    return X, y


def atom_segment(atom: dict, length: int) -> torch.Tensor:
    return resize_seq(atom["atom"], length).reshape(length, HAND_DIM)


def transition_cost(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    pos = (left[-1] - right[0]).pow(2).mean().sqrt()
    if left.shape[0] > 1 and right.shape[0] > 1:
        lv = left[-1] - left[-2]
        rv = right[1] - right[0]
        vel = (lv - rv).pow(2).mean().sqrt()
    else:
        vel = torch.tensor(0.0)
    scale = torch.cat([left.flatten(), right.flatten()]).std().clamp_min(1e-4)
    return float((pos + 0.5 * vel) / scale)


def sequence_route(stages: list[dict], transition_weight: float) -> list[int]:
    if not stages:
        return []
    scores = []
    backptrs: list[torch.Tensor | None] = []
    for si, stage in enumerate(stages):
        unary = stage["scores"]
        if unary.numel() > 1:
            unary = (unary - unary.mean()) / unary.std().clamp_min(1e-6)
        else:
            unary = torch.zeros_like(unary)
        if si == 0:
            scores.append(unary)
            backptrs.append(None)
            continue
        prev = scores[-1]
        trans = torch.empty(prev.numel(), unary.numel())
        for pi, pseg in enumerate(stages[si - 1]["segments"]):
            for ci, cseg in enumerate(stage["segments"]):
                trans[pi, ci] = transition_cost(pseg, cseg)
        vals = prev[:, None] + unary[None, :] - transition_weight * trans
        best_vals, best_idx = vals.max(dim=0)
        scores.append(best_vals)
        backptrs.append(best_idx)
    route = [int(torch.argmax(scores[-1]))]
    for si in range(len(stages) - 1, 0, -1):
        route.append(int(backptrs[si][route[-1]]))
    route.reverse()
    return route


def train_router(args) -> None:
    train_manifest = load_manifest(ROOT / args.train_manifest)
    train_gt = load_pose_map(ROOT / args.train_gt_pt)
    train_source = load_pose_map(ROOT / args.train_source_pt)
    memory = build_atoms(train_manifest, train_gt, args.kernel, args.atom_len, args.max_atoms_per_gloss)
    X, y = build_training_pairs(args, train_manifest, train_gt, train_source, memory)
    n = X.shape[0]
    idx = torch.randperm(n)
    n_val = max(512, int(0.05 * n)) if n > 1024 else max(1, n // 10)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    mean, std = X[train_idx].mean(0), X[train_idx].std(0).clamp_min(1e-6)
    Xn = (X - mean) / std
    device = torch.device(args.device)
    model = Router(X.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best = None
    log = []
    print(f"[atom-router] pairs={n} train={len(train_idx)} val={len(val_idx)} memory_glosses={len(memory)}", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        perm = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(perm), args.batch_size):
            bi = perm[start : start + args.batch_size]
            pred = model(Xn[bi].to(device))
            loss = F.mse_loss(pred, y[bi].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vp = model(Xn[val_idx].to(device)).cpu()
            vl = F.mse_loss(vp, y[val_idx]).item()
            corr = float(np.corrcoef(vp.numpy(), y[val_idx].numpy())[0, 1]) if len(val_idx) > 2 else 0.0
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vl, "val_corr": corr}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if best is None or vl < best["val_loss"]:
            best = rec
            torch.save({
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "args": vars(args),
                "memory": memory,
                "best": best,
            }, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))


@torch.no_grad()
def compose(args) -> None:
    ckpt = torch.load(ROOT / args.router_ckpt, map_location="cpu", weights_only=False)
    memory = ckpt["memory"]
    mean = ckpt["mean"]
    std = ckpt["std"].clamp_min(1e-6)
    model = Router(mean.numel(), ckpt["args"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    split_manifest = load_manifest(ROOT / args.manifest_json)
    source = load_pose_map(ROOT / args.source_pt)
    out = {}
    trace = []
    for row in split_manifest:
        sid = str(row["id"])
        if sid not in source:
            continue
        src = source[sid].float()
        L = int(src.shape[0])
        glosses = tokens(row)
        src_lp = temporal_smooth(src, args.kernel)
        tex_num = torch.zeros(L, HAND_DIM)
        tex_den = torch.zeros(L, 1)
        entries = []
        bounds = source_guided_bounds(src, len(glosses), args.boundary_window_frac)
        stages = []
        for gi, (gloss, (a, b)) in enumerate(zip(glosses, bounds)):
            atoms = memory.get(gloss, [])
            if not atoms:
                entries.append({"gloss": gloss, "hit": False, "frames": [a, b]})
                continue
            cand = atoms[: args.compose_candidates]
            feats = torch.stack([pair_features(src, a, b, gi, len(glosses), atom) for atom in cand])
            scores = model((feats - mean) / std).cpu()
            stages.append({
                "gloss": gloss,
                "frames": (a, b),
                "atoms": cand,
                "scores": scores,
                "segments": [atom_segment(atom, b - a) for atom in cand],
            })
        route = sequence_route(stages, args.dp_transition_weight) if args.sequence_dp else []
        selected = {}
        for si, stage in enumerate(stages):
            best_i = route[si] if args.sequence_dp else int(torch.argmax(stage["scores"]))
            selected[(stage["gloss"], stage["frames"])] = (stage, best_i)
        for gi, (gloss, (a, b)) in enumerate(zip(glosses, bounds)):
            key = (gloss, (a, b))
            if key not in selected:
                continue
            stage, best_i = selected[key]
            atom = stage["atoms"][best_i]
            aa = max(0, a - args.overlap)
            bb = min(L, b + args.overlap)
            seg = resize_seq(atom["atom"], bb - aa).reshape(bb - aa, HAND_DIM)
            w = blend_weight(bb - aa, args.overlap).reshape(bb - aa, 1)
            tex_num[aa:bb] += seg * w
            tex_den[aa:bb] += w
            entries.append({
                "gloss": gloss,
                "hit": True,
                "retrieved_id": atom["source_id"],
                "score": float(stage["scores"][best_i]),
                "frames": [a, b],
                "route": "sequence_dp" if args.sequence_dp else "local_argmax",
            })
        tex = tex_num / tex_den.clamp_min(1e-6)
        pose = src.clone()
        pose[:, HAND] = src_lp[:, HAND] + tex.reshape(L, 42, 3)
        if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
            pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
        out[sid] = pose.contiguous()
        trace.append({"id": sid, "entries": entries})
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
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--train_gt_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_source_pt", default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_train_scale080.pt")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_learned_atom_router")
    tr.add_argument("--kernel", type=int, default=9)
    tr.add_argument("--atom_len", type=int, default=16)
    tr.add_argument("--max_atoms_per_gloss", type=int, default=24)
    tr.add_argument("--candidates_per_gloss", type=int, default=8)
    tr.add_argument("--random_candidates", type=int, default=4)
    tr.add_argument("--boundary_window_frac", type=float, default=0.45)
    tr.add_argument("--max_train_clips", type=int, default=0)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch_size", type=int, default=1024)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)

    co = sub.add_parser("compose")
    co.add_argument("--router_ckpt", required=True)
    co.add_argument("--source_pt", required=True)
    co.add_argument("--manifest_json", required=True)
    co.add_argument("--out_pt", required=True)
    co.add_argument("--trace_json", default="")
    co.add_argument("--kernel", type=int, default=9)
    co.add_argument("--boundary_window_frac", type=float, default=0.45)
    co.add_argument("--overlap", type=int, default=4)
    co.add_argument("--compose_candidates", type=int, default=16)
    co.add_argument("--sequence_dp", action="store_true")
    co.add_argument("--dp_transition_weight", type=float, default=0.35)
    co.add_argument("--post_smooth_kernel", type=int, default=3)
    co.add_argument("--post_smooth_blend", type=float, default=0.05)
    args = ap.parse_args()
    if args.mode == "train":
        train_router(args)
    else:
        compose(args)


if __name__ == "__main__":
    main()
