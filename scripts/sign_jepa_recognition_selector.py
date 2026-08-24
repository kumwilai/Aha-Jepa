"""Recognition-aware selector for Sign-JEPA candidate families.

This selector is trained from per-sample back-translation text rewards instead
of pose-oracle labels.  It uses only deployable candidate-output features at
inference time.
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

from scripts.sign_jepa_candidate_selector import (  # noqa: E402
    Selector,
    load_manifest,
    load_pose_map,
    parse_candidate,
    pose_features,
)
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

HAND = slice(8, 50)


def load_gt_text(path: Path) -> dict[str, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(k): str(v["text"]) for k, v in data.items()}


def load_text_preds(path: Path, gt_keys: list[str]) -> dict[str, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        return {sid: str(txt) for sid, txt in zip(gt_keys, data)}
    raise TypeError(f"unsupported text prediction format: {type(data)}")


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[-1] + 1,
                prev[j - 1] + (0 if x == y else 1),
            ))
        prev = cur
    return prev[-1]


def text_reward(hyp: str, ref: str) -> float:
    h = hyp.lower().strip().split()
    r = ref.lower().strip().split()
    if not r:
        return 0.0
    ed_sim = 1.0 - edit_distance(h, r) / max(len(h), len(r), 1)
    hs, rs = set(h), set(r)
    inter = len(hs & rs)
    prec = inter / max(1, len(hs))
    rec = inter / max(1, len(rs))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    len_penalty = abs(len(h) - len(r)) / max(len(h), len(r), 1)
    return 0.65 * ed_sim + 0.45 * f1 - 0.10 * len_penalty


def parse_text_pred(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def resize_to(pose: torch.Tensor, length: int) -> torch.Tensor:
    if pose.shape[0] == length:
        return pose
    return resize_seq(pose.reshape(pose.shape[0], -1), length).reshape(length, 178, 3)


def speed_slice(pose: torch.Tensor, sl: slice) -> torch.Tensor:
    x = pose[:, sl]
    if x.shape[0] < 2:
        return torch.tensor(0.0)
    return (x[1:] - x[:-1]).norm(dim=-1).mean()


def carrier_relative_features(pose: torch.Tensor, carrier: torch.Tensor | None) -> torch.Tensor:
    if carrier is None:
        return torch.zeros(8)
    p = resize_to(pose.float(), carrier.shape[0])
    c = carrier.float()
    full = (p - c).norm(dim=-1).mean()
    hand = (p[:, HAND] - c[:, HAND]).norm(dim=-1).mean()
    body = (p[:, :8] - c[:, :8]).norm(dim=-1).mean()
    face = (p[:, 50:] - c[:, 50:]).norm(dim=-1).mean()
    return torch.tensor([
        pose.shape[0] / max(1, carrier.shape[0]),
        carrier.shape[0] / max(1, pose.shape[0]),
        float(full),
        float(hand),
        float(body),
        float(face),
        float(speed_slice(p, HAND) / speed_slice(c, HAND).clamp_min(1e-6)),
        float(p[:, HAND].std() / c[:, HAND].std().clamp_min(1e-6)),
    ], dtype=torch.float32)


def build_examples_rec(manifest, candidates, carrier_map=None):
    names = [name for name, _ in candidates]
    pose_maps = [(name, load_pose_map(ROOT / path)) for name, path in candidates]
    rows = []
    for sid, row in manifest.items():
        target_len = int(row.get("length", 0) or 0)
        feats = []
        valid_names = []
        carrier = carrier_map.get(sid) if carrier_map is not None else None
        for mi, (name, poses) in enumerate(pose_maps):
            if sid not in poses:
                continue
            base = pose_features(poses[sid], target_len, mi, len(names), row)
            rel = carrier_relative_features(poses[sid], carrier)
            feats.append(torch.cat([base, rel]))
            valid_names.append(name)
        if feats:
            rows.append({"id": sid, "features": torch.stack(feats), "names": valid_names})
    return rows


def build_label_rows(args):
    gt_text = load_gt_text(ROOT / args.gt_pt)
    gt_keys = list(gt_text.keys())
    rewards_by_name = {}
    for spec in args.text_pred:
        name, path = parse_text_pred(spec)
        preds = load_text_preds(ROOT / path, gt_keys)
        rewards_by_name[name] = {sid: text_reward(preds.get(sid, ""), gt_text[sid]) for sid in gt_keys}
    carrier_map = load_pose_map(ROOT / args.carrier_pt) if args.carrier_pt else None
    rows = build_examples_rec(load_manifest(ROOT / args.manifest_json), [parse_candidate(x) for x in args.candidate], carrier_map)
    labeled = []
    for row in rows:
        scores = torch.tensor([rewards_by_name.get(name, {}).get(row["id"], -1.0) for name in row["names"]], dtype=torch.float32)
        if scores.numel() == 0:
            continue
        labeled.append({**row, "reward": scores, "label_idx": int(torch.argmax(scores))})
    return labeled


def train(args) -> None:
    rows = build_label_rows(args)
    X = torch.cat([r["features"] for r in rows], dim=0)
    mean = X.mean(0)
    std = X.std(0).clamp_min(1e-6)
    model = Selector(X.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for ep in range(1, args.epochs + 1):
        losses = []
        correct = 0
        for row in rows:
            logits = model((row["features"] - mean) / std)
            target = torch.tensor([row["label_idx"]], dtype=torch.long)
            ce = F.cross_entropy(logits[None, :], target)
            # Also regress the reward shape so near-ties are less brittle.
            reward = row["reward"]
            reward = (reward - reward.mean()) / reward.std().clamp_min(1e-6) if reward.numel() > 1 else reward * 0
            mse = F.mse_loss(logits, reward)
            loss = ce + args.reward_mse_weight * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            correct += int(torch.argmax(logits).item() == row["label_idx"])
        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            print(json.dumps({"epoch": ep, "loss": sum(losses) / len(losses), "choice_acc": correct / len(rows)}), flush=True)
    out = ROOT / args.out_ckpt
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "hidden": args.hidden,
        "candidates": [parse_candidate(x) for x in args.candidate],
    }, out)
    print(f"saved -> {out}")


@torch.no_grad()
def compose(args) -> None:
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    candidates = [parse_candidate(x) for x in args.candidate] if args.candidate else [(n, Path(p)) for n, p in ckpt["candidates"]]
    manifest = load_manifest(ROOT / args.manifest_json)
    carrier_map = load_pose_map(ROOT / args.carrier_pt) if args.carrier_pt else None
    rows = build_examples_rec(manifest, candidates, carrier_map)
    pose_maps = {name: load_pose_map(ROOT / path) for name, path in candidates}
    model = Selector(ckpt["mean"].numel(), ckpt["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt["mean"]
    std = ckpt["std"].clamp_min(1e-6)
    out = {}
    trace = []
    replacements = {}
    for spec in args.replace_choice or []:
        src, dst = spec.split("=", 1)
        replacements[src] = dst
    for row in rows:
        logits = model((row["features"] - mean) / std)
        choice = row["names"][int(torch.argmax(logits))]
        raw_choice = choice
        choice = replacements.get(choice, choice)
        if choice not in pose_maps or row["id"] not in pose_maps[choice]:
            choice = raw_choice
        out[row["id"]] = pose_maps[choice][row["id"]].contiguous()
        trace.append({"id": row["id"], "choice": choice, "raw_choice": raw_choice, "scores": {n: float(v) for n, v in zip(row["names"], logits)}})
    if args.filter_keys_pt:
        keys = [str(k) for k in torch.load(ROOT / args.filter_keys_pt, map_location="cpu", weights_only=False).keys()]
        out = {sid: out[sid] for sid in keys if sid in out}
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
    tr.add_argument("--gt_pt", required=True)
    tr.add_argument("--out_ckpt", required=True)
    tr.add_argument("--carrier_pt", default="")
    tr.add_argument("--candidate", action="append", required=True)
    tr.add_argument("--text_pred", action="append", required=True)
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--reward_mse_weight", type=float, default=0.2)

    co = sub.add_parser("compose")
    co.add_argument("--ckpt", required=True)
    co.add_argument("--manifest_json", required=True)
    co.add_argument("--out_pt", required=True)
    co.add_argument("--trace_json", default="")
    co.add_argument("--filter_keys_pt", default="")
    co.add_argument("--carrier_pt", default="")
    co.add_argument("--candidate", action="append")
    co.add_argument("--replace_choice", action="append")
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        compose(args)


if __name__ == "__main__":
    main()
