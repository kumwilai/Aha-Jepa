"""3-seed ensemble for the carrier swap: average the raw generated pose dicts
across seeds {0,1,2} per carrier, per split. Length is duration-policy
deterministic (seed-independent), so samples are frame-aligned; we still guard
against any T mismatch by truncating to the min length.

  usage: python scripts/carrier_swap_ensemble.py <OBJ> <split>
writes outputs/sota_chase/carrier_swap/<OBJ>_raw_ens_<split>.pt
"""
import sys
from pathlib import Path
import torch

OBJ, SP = sys.argv[1], sys.argv[2]
CSW = Path("outputs/sota_chase/carrier_swap")
# seed 0 has no suffix; seeds 1,2 do
paths = [CSW / f"{OBJ}_raw_{SP}.pt",
         CSW / f"{OBJ}_raw_s1_{SP}.pt",
         CSW / f"{OBJ}_raw_s2_{SP}.pt"]
present = [p for p in paths if p.exists()]
if len(present) < 2:
    sys.exit(f"[ensemble] need >=2 seeds, found {len(present)}: {present}")
dicts = [torch.load(p, map_location="cpu", weights_only=False) for p in present]
keys = set(dicts[0])
for d in dicts[1:]:
    keys &= set(d)
out = {}
mism = 0
for k in dicts[0]:
    if k not in keys:
        out[k] = dicts[0][k]  # fall back to seed-0 for any non-shared key
        continue
    tens = [d[k] for d in dicts]
    Ts = [t.shape[0] for t in tens]
    if len(set(Ts)) > 1:
        mism += 1
        T = min(Ts)
        tens = [t[:T] for t in tens]
    out[k] = torch.stack(tens, 0).mean(0)
dst = CSW / f"{OBJ}_raw_ens_{SP}.pt"
torch.save(out, dst)
print(f"[ensemble] {OBJ} {SP}: {len(present)} seeds, {len(out)} samples, "
      f"{mism} length-mismatch (truncated) -> {dst}")
