"""Build a native SLRTP-178 exemplar bank for PG-RAST++.

This preserves the Phase-26 PG-RAST++ symbolic index
(`gloss_to_exemplars`, `exemplar_upc`) but replaces PT-201 exemplar poses
with SLRTP train `poses_3d` in flattened `(T, 178*3)` format. Native-178
assembly avoids the lossy PT-201 -> 178 converter for SLRTP evaluation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_bank",
                    default="outputs/sota_chase/phase24_pgrast/exemplars_upc.pt")
    ap.add_argument("--slrtp_train",
                    default="external/SLRTP-Sign-Production-Evaluation/pretrained/"
                            "SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--out",
                    default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    args = ap.parse_args()

    print(f"Loading source bank: {args.source_bank}")
    bank = torch.load(args.source_bank, map_location="cpu", weights_only=False)
    print(f"  source clips={len(bank['exemplar_poses'])} "
          f"glosses={bank['n_glosses']} exemplars={bank['n_exemplars']}")

    print(f"Loading SLRTP train: {args.slrtp_train}")
    slrtp = torch.load(args.slrtp_train, map_location="cpu", weights_only=False)
    pose_pool = {}
    missing = []
    for sid in bank["exemplar_poses"].keys():
        item = slrtp.get(sid)
        if item is None:
            missing.append(sid)
            continue
        pose = item["poses_3d"]
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        pose_pool[sid] = np.asarray(pose, dtype=np.float32).reshape(pose.shape[0], -1)

    # Drop spans whose source clip is not present in SLRTP train.
    g2e = {}
    for gloss, rows in bank["gloss_to_exemplars"].items():
        keep = [(sid, int(s), int(e)) for sid, s, e in rows if sid in pose_pool]
        if keep:
            g2e[gloss] = keep

    exemplar_upc = {}
    for key, val in bank.get("exemplar_upc", {}).items():
        sid, s, e = key
        if sid in pose_pool:
            exemplar_upc[(sid, int(s), int(e))] = val

    out = dict(bank)
    out["exemplar_poses"] = pose_pool
    out["gloss_to_exemplars"] = g2e
    out["exemplar_upc"] = exemplar_upc
    out["n_glosses"] = len(g2e)
    out["n_exemplars"] = sum(len(v) for v in g2e.values())
    out["pose_format"] = "slrtp178_flat"
    out["missing_source_clips"] = missing

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved: {out_path}")
    print(f"  native clips={len(pose_pool)} missing={len(missing)}")
    print(f"  native glosses={out['n_glosses']} exemplars={out['n_exemplars']}")


if __name__ == "__main__":
    main()
