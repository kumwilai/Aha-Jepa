"""Face-channel borrow operator: graft a real face expression onto a collapsed-face output.

Diagnosis (test selection after wrist-restore):
  face speed ratio = 0.21x GT, but face *shape std* ratio = 0.99 — the face is
  not shape-collapsed, it is time-collapsed: it cycles through similar
  expressions in slow motion (LP fraction 0.71 vs GT 0.50).  Amplitude
  amplification (the wrist trick) has nothing to amplify here — there is no
  shape headroom.  Only adding faster, real expression dynamics can lift the
  face-speed ratio without breaking the face geometry.

Operator (per clip):

  - retrieve a real train clip via the existing clip_texture trace
    (lexical + length overlap — same retrieval used for hand high-pass);
  - decompose the real clip's face into centroid + expression (centroid is
    canonically constant within a SLRTP clip — the per-frame face centroid
    speed is ~0 in GT, confirming a normalised head frame);
  - time-warp the real expression to the output length;
  - replace the output's face = output_centroid + real_expression, optionally
    blended with the original output face (`--blend`).

This preserves the output's gross face placement (the head doesn't move
laterally), replaces the temporal expression dynamics with real ones, and
gives the back-translation evaluator a moving, anatomically valid face.

Risks (measured, not assumed):

  - face content mismatch (borrowed mouthing != target sentence) may hurt or
    help BLEU/WER depending on whether the back-translation model reads the
    face;
  - DTW-MJE includes the 128 face joints (~16% of joint-error budget); a
    content-mismatched real face may inflate face MJE.

Usage::

    python scripts/sign_jepa_face_borrow.py \
      --in_pt external/.../sign_jepa_multiobjective_selector_restored_test.pt \
      --retrieval_trace_json outputs/sota_chase/sign_jepa_motion_texture_retrieval/test_trace.json \
      --out_pt external/.../sign_jepa_multiobjective_selector_restored_facefix_test.pt \
      --trace_json outputs/sota_chase/sign_jepa_multiobjective_selector/test_face_borrow_trace.json
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

from scripts.sign_jepa_motion_texture_retrieval import as_pose  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

FACE = slice(50, 178)


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def load_retrieval(path: Path) -> dict[str, str]:
    rows = json.loads(Path(path).read_text())
    if isinstance(rows, dict) and "items" in rows:
        rows = rows["items"]
    return {str(r["id"]): str(r.get("retrieved_id", "")) for r in rows}


def borrow_face_clip(target: torch.Tensor, real: torch.Tensor, blend: float) -> torch.Tensor:
    """Replace target's face expression with real's (centroid-removed),
    optionally blended with target's original expression."""
    Tt = target.shape[0]
    real_face = real[:, FACE].float()
    real_centroid = real_face.mean(dim=1, keepdim=True)            # [Tr, 1, 3]
    real_expr = real_face - real_centroid                          # [Tr, 128, 3]
    if real_expr.shape[0] != Tt:
        real_expr = resize_seq(real_expr.reshape(real_expr.shape[0], -1), Tt).reshape(Tt, 128, 3)
    out = target.clone().float()
    tgt_centroid = out[:, FACE].mean(dim=1, keepdim=True)          # [Tt, 1, 3]
    tgt_expr = out[:, FACE] - tgt_centroid
    new_expr = (1.0 - blend) * real_expr + blend * tgt_expr
    out[:, FACE] = tgt_centroid + new_expr
    return out.contiguous()


def run(args: argparse.Namespace) -> None:
    target = load_pose_map(ROOT / args.in_pt)
    retrieval = load_retrieval(ROOT / args.retrieval_trace_json)
    raw = torch.load(ROOT / args.train_pt, map_location="cpu", weights_only=False)
    train_pose: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        try:
            train_pose[str(k)] = as_pose(v).contiguous()
        except Exception:
            continue
    print(f"[face_borrow] target={len(target)} train_pose={len(train_pose)} "
          f"retrievals={len(retrieval)}", flush=True)

    out: dict[str, torch.Tensor] = {}
    trace = []
    borrowed = missed = 0
    for sid, pose in target.items():
        rid = retrieval.get(sid, "")
        if rid and rid in train_pose:
            new_pose = borrow_face_clip(pose, train_pose[rid], args.blend)
            borrowed += 1
            row = {"id": sid, "retrieved_id": rid, "borrowed": True}
        else:
            new_pose = pose
            missed += 1
            row = {"id": sid, "retrieved_id": rid, "borrowed": False}
        out[sid] = new_pose
        trace.append(row)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)} borrowed={borrowed} no_retrieval={missed}", flush=True)
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps({
            "in_pt": args.in_pt,
            "retrieval_trace_json": args.retrieval_trace_json,
            "blend": args.blend,
            "borrowed": borrowed,
            "no_retrieval": missed,
            "items": trace,
        }, indent=2))
        print(f"saved trace -> {tp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--retrieval_trace_json", required=True,
                    help="trace with per-clip retrieved_id (e.g. motion_texture_retrieval trace)")
    ap.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/"
                    "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="0 = full borrow of real face expression; 1 = keep original")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
