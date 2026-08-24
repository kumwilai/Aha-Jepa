"""Memory-only retrieval baseline for Sign-JEPA ablations.

This intentionally removes the JEPA carrier.  It retrieves a real train clip
using only target gloss overlap and length compatibility, then resizes the full
retrieved pose to the target duration.  The baseline answers whether the gains
come from JEPA as a semantic trajectory or from retrieval alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_motion_texture_retrieval import as_pose, tokens  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def build_index(train_manifest: list[dict], train_gt: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for row in train_manifest:
        sid = str(row["id"])
        if sid not in train_gt:
            continue
        glosses = tokens(row)
        if not glosses:
            continue
        rows.append({
            "id": sid,
            "tokens": glosses,
            "tokset": set(glosses),
            "length": int(row.get("length", train_gt[sid].shape[0]) or train_gt[sid].shape[0]),
        })
    return rows


def retrieve(row: dict, index: list[dict]) -> dict:
    query = set(tokens(row))
    q_len = int(row.get("length", 0) or 0)
    best = None
    best_score = -1e9
    for cand in index:
        inter = len(query & cand["tokset"])
        union = max(1, len(query | cand["tokset"]))
        len_penalty = abs(q_len - cand["length"]) / max(q_len, cand["length"], 1)
        score = 4.0 * (inter / union) + 0.2 * inter - 0.35 * len_penalty
        if score > best_score:
            best_score = score
            best = cand
    if best is None:
        raise RuntimeError("empty memory index")
    return best


def target_length(row: dict, fallback: int) -> int:
    value = int(row.get("length", 0) or 0)
    return value if value > 1 else fallback


def compose(args: argparse.Namespace) -> None:
    train_manifest = load_manifest(ROOT / args.train_manifest)
    split_manifest = load_manifest(ROOT / args.manifest_json)
    train_gt = load_pose_map(ROOT / args.train_gt_pt)
    filter_keys = None
    if args.filter_keys_pt:
        filter_keys = [str(k) for k in torch.load(ROOT / args.filter_keys_pt, map_location="cpu", weights_only=False).keys()]
    index = build_index(train_manifest, train_gt)
    out = {}
    trace = []
    for row in split_manifest:
        sid = str(row["id"])
        cand = retrieve(row, index)
        src = train_gt[cand["id"]].float()
        L = target_length(row, src.shape[0])
        pose = resize_seq(src.reshape(src.shape[0], -1), L).reshape(L, 178, 3)
        out[sid] = pose.contiguous()
        trace.append({
            "id": sid,
            "retrieved_id": cand["id"],
            "retrieved_tokens": cand["tokens"],
            "target_tokens": tokens(row),
            "target_len": L,
            "retrieved_len": cand["length"],
        })
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
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--filter_keys_pt", default="")
    ap.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    ap.add_argument("--train_gt_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    args = ap.parse_args()
    compose(args)


if __name__ == "__main__":
    main()
