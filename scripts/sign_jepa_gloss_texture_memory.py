"""Gloss-level motion texture memory for Sign-JEPA carriers.

Algorithm:

1. Build a train-only memory of high-frequency hand textures per gloss.
2. For a target gloss sequence, allocate the old JEPA clip timeline over glosses.
3. Retrieve one texture atom per gloss, time-warp it to the allocated segment.
4. Add only that hand high-frequency texture to the old JEPA low-pass carrier.

This is stricter than whole-clip retrieval: the semantic carrier remains JEPA,
and motion texture is aligned at the lexical token level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import LENGTH_BUCKETS, length_bucket_id, resize_seq  # noqa: E402

HAND = slice(8, 50)
HAND_DIM = 42 * 3


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


def split_bounds(T: int, n: int) -> list[tuple[int, int]]:
    n = max(1, n)
    cuts = [round(i * T / n) for i in range(n + 1)]
    out = []
    for i in range(n):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if b <= a:
            b = min(T, a + 1)
        out.append((a, b))
    return out


def source_guided_bounds(src: torch.Tensor, n: int, window_frac: float = 0.45) -> list[tuple[int, int]]:
    """Place gloss cuts near low-motion valleys in the JEPA carrier."""
    T = int(src.shape[0])
    n = max(1, n)
    if n == 1 or T <= 2:
        return [(0, T)]
    h = src[:, HAND]
    speed = torch.zeros(T)
    speed[1:] = (h[1:] - h[:-1]).norm(dim=-1).mean(dim=-1)
    base = [round(i * T / n) for i in range(n + 1)]
    cuts = [0]
    min_gap = max(1, T // (4 * n))
    half = max(2, int(round(T * window_frac / n)))
    for i in range(1, n):
        center = base[i]
        lo = max(cuts[-1] + min_gap, center - half)
        hi = min(T - (n - i) * min_gap, center + half)
        if hi <= lo:
            cut = max(cuts[-1] + 1, min(T - (n - i), center))
        else:
            local = speed[lo:hi]
            cut = lo + int(torch.argmin(local))
        cuts.append(cut)
    cuts.append(T)
    return [(int(cuts[i]), int(cuts[i + 1])) for i in range(n) if cuts[i + 1] > cuts[i]]


def hand_speed_from_flat(flat: torch.Tensor) -> float:
    if flat.shape[0] < 2:
        return 0.0
    h = flat.reshape(flat.shape[0], 42, 3)
    return float((h[1:] - h[:-1]).norm(dim=-1).mean())


def hand_speed_pose(pose: torch.Tensor) -> float:
    h = pose[:, HAND]
    if h.shape[0] < 2:
        return 0.0
    return float((h[1:] - h[:-1]).norm(dim=-1).mean())


def taper(seg: torch.Tensor) -> torch.Tensor:
    if seg.shape[0] < 4:
        return seg
    w = torch.hann_window(seg.shape[0], periodic=False, dtype=seg.dtype).clamp_min(0.15)
    return seg * w[:, None]


def blend_weight(length: int, overlap: int) -> torch.Tensor:
    if length <= 1:
        return torch.ones(length)
    w = torch.ones(length)
    o = min(overlap, max(0, length // 2))
    if o > 1:
        ramp = torch.linspace(0.0, 1.0, o)
        w[:o] = ramp
        w[-o:] = torch.flip(ramp, dims=[0])
    return w.clamp_min(0.05)


def energy_stats(train_gt: dict[str, torch.Tensor]) -> list[float]:
    vals = [[] for _ in range(len(LENGTH_BUCKETS))]
    all_vals = []
    for pose in train_gt.values():
        v = hand_speed_pose(pose)
        vals[length_bucket_id(int(pose.shape[0]))].append(v)
        all_vals.append(v)
    global_v = float(np.median(all_vals)) if all_vals else 1.0
    return [float(np.median(v)) if v else global_v for v in vals]


def build_memory(train_manifest: list[dict], train_gt: dict[str, torch.Tensor], kernel: int, atom_len: int, max_atoms: int) -> dict:
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
        for gloss, (a, b) in zip(glosses, split_bounds(pose.shape[0], len(glosses))):
            seg = hp[a:b, HAND].reshape(b - a, HAND_DIM)
            if seg.shape[0] < 2:
                continue
            atom = resize_seq(seg, atom_len).reshape(atom_len, HAND_DIM).contiguous()
            by_gloss[gloss].append({
                "source_id": sid,
                "source_len": int(b - a),
                "atom": atom,
                "speed": hand_speed_from_flat(atom),
            })
    memory = {}
    for gloss, atoms in by_gloss.items():
        atoms.sort(key=lambda x: x["speed"], reverse=True)
        # Keep energetic exemplars; averaging high-pass texture destroys motion.
        memory[gloss] = atoms[:max_atoms]
    return memory


def retrieve_atom(gloss: str, target_len: int, memory: dict) -> dict | None:
    atoms = memory.get(gloss)
    if not atoms:
        return None
    return min(atoms, key=lambda x: abs(x["source_len"] - target_len))


def compose(args: argparse.Namespace) -> None:
    train_manifest = load_manifest(ROOT / args.train_manifest)
    split_manifest = load_manifest(ROOT / args.manifest_json)
    source = load_pose_map(ROOT / args.source_pt)
    train_gt = load_pose_map(ROOT / args.train_gt_pt)
    memory = build_memory(train_manifest, train_gt, args.kernel, args.atom_len, args.max_atoms_per_gloss)
    bucket_speed = energy_stats(train_gt)
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
        bounds = source_guided_bounds(src, len(glosses), args.boundary_window_frac) if args.source_guided_bounds else split_bounds(L, len(glosses))
        for gloss, (a, b) in zip(glosses, bounds):
            atom = retrieve_atom(gloss, b - a, memory)
            if atom is None:
                entries.append({"gloss": gloss, "hit": False, "frames": [a, b]})
                continue
            seg = resize_seq(atom["atom"], b - a).reshape(b - a, HAND_DIM)
            if args.boundary_taper:
                seg = taper(seg)
            aa = max(0, a - args.overlap)
            bb = min(L, b + args.overlap)
            seg_ext = resize_seq(atom["atom"], bb - aa).reshape(bb - aa, HAND_DIM)
            if args.boundary_taper:
                seg_ext = taper(seg_ext)
            w = blend_weight(bb - aa, args.overlap).reshape(bb - aa, 1)
            tex_num[aa:bb] += seg_ext * w
            tex_den[aa:bb] += w
            entries.append({
                "gloss": gloss,
                "hit": True,
                "retrieved_id": atom["source_id"],
                "source_len": atom["source_len"],
                "frames": [a, b],
            })
        tex = tex_num / tex_den.clamp_min(1e-6)
        pose = src.clone()
        pose[:, HAND] = src_lp[:, HAND] + args.texture_gain * tex.reshape(L, 42, 3)
        if args.energy_normalize:
            cur = max(hand_speed_pose(pose), 1e-8)
            tgt = max(float(bucket_speed[length_bucket_id(L)]), 1e-8)
            hp = pose[:, HAND] - src_lp[:, HAND]
            scale = float(np.clip(tgt / cur, args.min_energy_scale, args.max_energy_scale))
            pose[:, HAND] = src_lp[:, HAND] + scale * hp
        if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
            pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
        out[sid] = pose.contiguous()
        trace.append({"id": sid, "glosses": glosses, "entries": entries})
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)} memory_glosses={len(memory)}")
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
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--atom_len", type=int, default=16)
    ap.add_argument("--max_atoms_per_gloss", type=int, default=8)
    ap.add_argument("--texture_gain", type=float, default=1.0)
    ap.add_argument("--boundary_taper", action="store_true", default=True)
    ap.add_argument("--no_boundary_taper", action="store_false", dest="boundary_taper")
    ap.add_argument("--source_guided_bounds", action="store_true", default=True)
    ap.add_argument("--uniform_bounds", action="store_false", dest="source_guided_bounds")
    ap.add_argument("--boundary_window_frac", type=float, default=0.45)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--energy_normalize", action="store_true", default=True)
    ap.add_argument("--no_energy_normalize", action="store_false", dest="energy_normalize")
    ap.add_argument("--min_energy_scale", type=float, default=0.25)
    ap.add_argument("--max_energy_scale", type=float, default=4.0)
    ap.add_argument("--post_smooth_kernel", type=int, default=3)
    ap.add_argument("--post_smooth_blend", type=float, default=0.05)
    args = ap.parse_args()
    compose(args)


if __name__ == "__main__":
    main()
