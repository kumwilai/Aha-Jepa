"""Hierarchical texture memory for Sign-JEPA carriers.

This combines the two useful memory levels:

1. Whole-clip retrieved texture for global temporal coherence.
2. Source-guided gloss texture memory when whole-clip retrieval has weak
   lexical coverage.

The switch is data-dependent, based on retrieved-token coverage from the trace;
it does not look at dev/test ground truth.
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

from scripts.sign_jepa_gloss_texture_memory import as_pose, load_manifest, tokens  # noqa: E402


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        out[str(sid)] = as_pose(pose).contiguous()
    return out


def trace_by_id(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in json.loads(path.read_text())}


def coverage(target_tokens: list[str], retrieved_tokens: list[str]) -> float:
    if not target_tokens:
        return 0.0
    rt = set(retrieved_tokens)
    return sum(1 for tok in target_tokens if tok in rt) / max(1, len(target_tokens))


def compose(args: argparse.Namespace) -> None:
    manifest = load_manifest(ROOT / args.manifest_json)
    clip = load_pose_map(ROOT / args.clip_texture_pt)
    gloss = load_pose_map(ROOT / args.gloss_texture_pt)
    trace = trace_by_id(ROOT / args.clip_trace_json)
    gt_keys = None
    if args.filter_keys_pt:
        gt_keys = [str(k) for k in torch.load(ROOT / args.filter_keys_pt, map_location="cpu", weights_only=False).keys()]
    out = {}
    rows = []
    for row in manifest:
        sid = str(row["id"])
        if sid not in clip and sid not in gloss:
            continue
        toks = tokens(row)
        tr = trace.get(sid, {})
        cov = coverage(toks, tr.get("retrieved_tokens", []))
        use_gloss = cov < args.min_clip_coverage and sid in gloss
        out[sid] = (gloss if use_gloss else clip)[sid].contiguous()
        rows.append({
            "id": sid,
            "coverage": cov,
            "choice": "gloss_memory" if use_gloss else "clip_texture",
            "retrieved_id": tr.get("retrieved_id", ""),
        })
    if gt_keys is not None:
        out = {sid: out[sid] for sid in gt_keys if sid in out}
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if args.trace_json:
        trace_path = ROOT / args.trace_json
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(rows, indent=2))
        print(f"saved trace -> {trace_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--clip_texture_pt", required=True)
    ap.add_argument("--gloss_texture_pt", required=True)
    ap.add_argument("--clip_trace_json", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--filter_keys_pt", default="")
    ap.add_argument("--min_clip_coverage", type=float, default=0.50)
    args = ap.parse_args()
    compose(args)


if __name__ == "__main__":
    main()
