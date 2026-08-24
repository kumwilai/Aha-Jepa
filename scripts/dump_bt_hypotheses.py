"""Dump SLRTP back-translation hypotheses for a generated pose file.

Mirrors external/SLRTP-Sign-Production-Evaluation/main.py exactly:

  * load {clip_id: [T, 178, 3]} (or {clip_id: {'poses_3d': ...}}),
  * downsample 25fps -> 12fps with [::2] when --fps 25 (main.py line ~96),
  * re-order predictions to the key order of the reference .pt,
  * run back_translation.back_translate with the same model builder,
  * torch.save a plain python list[str] of hypotheses in reference key order.

The only deviation from main.py is optional sequential chunking (--chunk),
which exists purely to bound peak host RAM on the 7060-clip train split.
back_translate pads every clip to the max length of whatever it is handed and
masks the padding, so chunking is numerically inert; scripts/dump_bt_hypotheses
is verified against main.py's own *_text_preds.pt before being trusted.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT", Path(__file__).resolve().parents[1]))
SLRTP = (ROOT / "external/SLRTP-Sign-Production-Evaluation").resolve()


def load_predictions(input_path: Path):
    # same branching as main.py:load_predictions
    if input_path.suffix == ".json":
        from helpers import load_json
        pred = load_json(input_path)
    elif input_path.suffix == ".pt":
        pred = torch.load(input_path, weights_only=True)
    elif input_path.suffix in (".pkl", ".pickle"):
        from helpers import load_pickle
        pred = load_pickle(input_path)
    else:
        raise ValueError(f"File type {input_path.suffix} not supported.")
    k = list(pred.keys())[0]
    if isinstance(pred[k], dict):
        pred = {k: v["poses_3d"] for k, v in pred.items()}
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_pt", required=True, help="generated poses {id: [T,178,3]}")
    ap.add_argument("--reference_pt", required=True, help="SLRTP {train,dev,test}.pt (key order + presence check)")
    ap.add_argument("--model_dir", default=str(SLRTP / "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model"))
    ap.add_argument("--out_pt", required=True, help="destination for the list[str] of hypotheses")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=0, help="0 = one call like main.py; >0 = sequential chunks of N clips")
    args = ap.parse_args()

    # back_translation.back_translate does relative imports and the harness
    # expects to be importable from its own root.
    sys.path.insert(0, str(SLRTP))
    from back_translation.back_translate import back_translate, make_back_translation_model

    pose_path = Path(args.pose_pt)
    ref_path = Path(args.reference_pt)
    assert pose_path.exists(), f"Path {pose_path} does not exist."
    assert ref_path.exists(), f"Path {ref_path} does not exist."

    pose_predictions = load_predictions(pose_path)
    ref = torch.load(str(ref_path), weights_only=True)
    ids = list(ref.keys())

    missing = [i for i in ids if i not in pose_predictions]
    if missing:
        raise KeyError(f"{len(missing)}/{len(ids)} reference ids missing from {pose_path}, first={missing[:3]}")
    if len(pose_predictions) != len(ids):
        raise ValueError(
            f"Number of predictions ({len(pose_predictions)}) and ground truth ({len(ids)}) does not match."
        )

    if int(args.fps) == 25:
        pose_predictions = {k: v[::2, ...] for k, v in pose_predictions.items()}
    poses = [pose_predictions[i] for i in ids]

    model = make_back_translation_model(model_dir=Path(args.model_dir))

    t0 = time.time()
    if args.chunk and args.chunk > 0:
        text_pred = []
        for s in range(0, len(poses), args.chunk):
            text_pred.extend(back_translate(model=model, poses=poses[s:s + args.chunk]))
            print(f"[dump-bt] {min(s + args.chunk, len(poses))}/{len(poses)} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    else:
        text_pred = back_translate(model=model, poses=poses)
    dt = time.time() - t0

    assert len(text_pred) == len(ids), f"{len(text_pred)} hypotheses for {len(ids)} clips"
    out_path = Path(args.out_pt)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(text_pred, out_path)
    print(f"clips={len(ids)} bt_seconds={dt:.1f} rate={len(ids) / max(dt, 1e-9):.2f} clips/s")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
