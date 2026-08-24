"""Trainable candidate selector for Sign-JEPA memory systems.

This imitates an oracle candidate mixer using only deployable features from the
candidate outputs and the manifest.  It is an exploratory selector prototype:
train on oracle labels from one split, then compose another split without using
ground-truth pose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_motion_texture_retrieval import as_pose, tokens  # noqa: E402

HAND = slice(8, 50)
BODY = slice(0, 8)
FACE = slice(50, 178)


def load_manifest(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in json.loads(path.read_text())}


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def parse_candidate(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def speed(pose: torch.Tensor, joints: slice) -> float:
    x = pose[:, joints]
    if x.shape[0] < 2:
        return 0.0
    return float((x[1:] - x[:-1]).norm(dim=-1).mean())


def jerk(pose: torch.Tensor) -> float:
    x = pose[:, HAND]
    if x.shape[0] < 3:
        return 0.0
    v = x[1:] - x[:-1]
    a = v[1:] - v[:-1]
    return float(a.norm(dim=-1).mean())


def pose_features(pose: torch.Tensor, target_len: int, method_i: int, method_count: int, row: dict) -> torch.Tensor:
    onehot = torch.zeros(method_count)
    onehot[method_i] = 1.0
    glosses = tokens(row)
    T = max(1, int(pose.shape[0]))
    target_len = max(1, target_len)
    vals = torch.tensor([
        T / target_len,
        target_len / T,
        abs(T - target_len) / max(T, target_len),
        len(glosses) / 12.0,
        speed(pose, HAND),
        speed(pose, BODY),
        speed(pose, FACE),
        jerk(pose),
        float(pose[:, HAND].std()),
        float(pose[:, BODY].std()),
    ], dtype=torch.float32)
    return torch.cat([onehot, vals])


class Selector(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_oracle_labels(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    return {str(row["id"]): row["choice"] for row in items}


def build_examples(manifest, candidates, labels=None):
    names = [name for name, _ in candidates]
    pose_maps = [(name, load_pose_map(ROOT / path)) for name, path in candidates]
    rows = []
    for sid, row in manifest.items():
        target_len = int(row.get("length", 0) or 0)
        feats = []
        valid_names = []
        for mi, (name, poses) in enumerate(pose_maps):
            if sid not in poses:
                continue
            feats.append(pose_features(poses[sid], target_len, mi, len(names), row))
            valid_names.append(name)
        if not feats:
            continue
        label = labels.get(sid) if labels else None
        rows.append({"id": sid, "features": torch.stack(feats), "names": valid_names, "label": label})
    return rows


def train(args) -> None:
    candidates = [parse_candidate(x) for x in args.candidate]
    manifest = load_manifest(ROOT / args.manifest_json)
    labels = load_oracle_labels(ROOT / args.oracle_trace_json)
    rows = build_examples(manifest, candidates, labels)
    X = []
    y = []
    for row in rows:
        for feat, name in zip(row["features"], row["names"]):
            X.append(feat)
            y.append(1.0 if name == row["label"] else 0.0)
    X = torch.stack(X)
    y = torch.tensor(y)
    mean = X.mean(0)
    std = X.std(0).clamp_min(1e-6)
    Xn = (X - mean) / std
    model = Selector(X.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([(len(y) - y.sum()).item() / y.sum().clamp_min(1).item()])
    for ep in range(1, args.epochs + 1):
        pred = model(Xn)
        loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            with torch.no_grad():
                correct = 0
                for row in rows:
                    scores = model((row["features"] - mean) / std)
                    correct += int(row["names"][int(torch.argmax(scores))] == row["label"])
            print(json.dumps({"epoch": ep, "loss": float(loss), "choice_acc": correct / max(1, len(rows))}), flush=True)
    out = ROOT / args.out_ckpt
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "hidden": args.hidden,
        "candidates": [(name, str(path)) for name, path in candidates],
    }, out)
    print(f"saved -> {out}")


@torch.no_grad()
def compose(args) -> None:
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    if args.candidate:
        candidates = [parse_candidate(x) for x in args.candidate]
    else:
        candidates = [(name, Path(path)) for name, path in ckpt["candidates"]]
    manifest = load_manifest(ROOT / args.manifest_json)
    rows = build_examples(manifest, candidates)
    mean = ckpt["mean"]
    std = ckpt["std"].clamp_min(1e-6)
    model = Selector(mean.numel(), ckpt["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    pose_maps = {name: load_pose_map(ROOT / path) for name, path in candidates}
    filter_keys = None
    if args.filter_keys_pt:
        filter_keys = [str(k) for k in torch.load(ROOT / args.filter_keys_pt, map_location="cpu", weights_only=False).keys()]
    out = {}
    trace = []
    for row in rows:
        scores = model((row["features"] - mean) / std)
        choice = row["names"][int(torch.argmax(scores))]
        out[row["id"]] = pose_maps[choice][row["id"]].contiguous()
        trace.append({"id": row["id"], "choice": choice, "scores": {n: float(s) for n, s in zip(row["names"], scores)}})
    if filter_keys is not None:
        out = {sid: out[sid] for sid in filter_keys if sid in out}
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
    tr.add_argument("--manifest_json", required=True)
    tr.add_argument("--oracle_trace_json", required=True)
    tr.add_argument("--out_ckpt", required=True)
    tr.add_argument("--candidate", action="append", required=True)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)

    co = sub.add_parser("compose")
    co.add_argument("--ckpt", required=True)
    co.add_argument("--manifest_json", required=True)
    co.add_argument("--out_pt", required=True)
    co.add_argument("--trace_json", default="")
    co.add_argument("--filter_keys_pt", default="")
    co.add_argument("--candidate", action="append")
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        compose(args)


if __name__ == "__main__":
    main()
