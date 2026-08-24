"""Visualise the frozen CSL-Daily Sign-JEPA latent space: extract per-gloss-span
target-encoder latents for the most frequent CSL glosses, mean-pool over time,
project to 2D with PCA, and scatter colored by gloss. CSL analogue of
``scripts/jepa_latent_figure.py`` (PHOENIX / SLRTP-178).

The carrier and data use the HRNet-133 (COCO-WholeBody) layout, so the model is
rebuilt exactly as ``scripts/aha_jepa_csl.py`` does for CSL (layout-agnostic
``SignJEPAModel`` + the CSL-specific ``sequence_features`` / span dataset /
normalisation), with the stored checkpoint args.

Outputs: paper/jepa_latent_csl.pdf  (+ EV(2) and between/within separability printed).
"""
from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_sign_jepa_slrtp178 import SignJEPAModel, SLRTPSpanDataset
# CSL HRNet-133-specific feature pipeline + span loader (the train-time path).
from aha_jepa_csl import sequence_features, load_bank

CKPT = ROOT / "outputs/sota_chase/sign_jepa_csl/best.pt"
TOPK = 8           # number of glosses to color
PER_GLOSS = 80     # spans per gloss
OUT = ROOT / "paper/jepa_latent_csl.pdf"

# CSL glosses are single Chinese characters; DejaVu Sans (the only font available
# here) cannot render CJK glyphs, so the legend uses a romanised pinyin + English
# label.  The map covers the top-8-by-frequency glosses (deterministic); any gloss
# not in the map falls back to its raw character (will render as tofu, but the
# scatter colour still carries the identity).
GLOSS_LABEL = {
    "我": "wo (I)",      "他": "ta (he)",   "你": "ni (you)",
    "这": "zhe (this)",  "人": "ren (person)", "多": "duo (many)",
    "不": "bu (not)",    "要": "yao (want)",
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    g2id = ck["gloss_to_id"]
    mean, std = ck["mean"].float(), ck["std"].float()
    model = SignJEPAModel(
        feat_dim=ck["feat_dim"], desc_dim=ck["desc_dim"], num_gloss=len(g2id),
        hidden=a["hidden"], layers=a["layers"], heads=a["heads"],
        pred_layers=a["pred_layers"], dropout=0.0, max_len=a["seg_len"],
    )
    model.load_state_dict(ck["model"]); model.eval().to(dev)

    # CSL bank + validated spans (sid present, e-s >= 3) -- identical to training.
    bank, all_spans = load_bank()
    # spans are (gloss, sid, s, e); group by gloss to pick the most frequent.
    from collections import defaultdict
    g2spans = defaultdict(list)
    for gloss, sid, s, e in all_spans:
        if gloss and gloss in g2id:
            g2spans[gloss].append((gloss, sid, s, e))
    freq = sorted(((g, len(v)) for g, v in g2spans.items()), key=lambda x: -x[1])
    glosses = [g for g, _ in freq[:TOPK]]
    spans = []
    for g in glosses:
        ex = list(g2spans[g])
        random.Random(0).shuffle(ex)
        spans.extend(ex[:PER_GLOSS])
    print(f"[latent-csl] {len(glosses)} glosses, {len(spans)} spans")
    print("[latent-csl] glosses:", [f"{g}({n})" for g, n in freq[:TOPK]])

    ds = SLRTPSpanDataset(bank, spans, g2id, mean, std, a["seg_len"])
    lat, lab = [], []
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            batch = [ds[j] for j in range(i, min(i + 64, len(ds)))]
            pose = torch.stack([b["pose"] for b in batch]).to(dev)
            pose_raw = torch.stack([b["pose_raw"] for b in batch]).to(dev)
            gid = torch.stack([b["gloss_id"] for b in batch]).to(dev)
            lid = torch.stack([b["len_id"] for b in batch]).to(dev)
            feat, _ = sequence_features(pose, pose_raw)       # CSL HRNet-133 features
            zmask = torch.zeros(feat.shape[:2], dtype=torch.bool, device=dev)
            z = model.target_encoder(feat, zmask, gid, lid)   # [B,T,H]
            lat.append(z.mean(1).cpu().numpy())               # mean-pool time
            lab += [b["gloss_id"].item() for b in batch]
    X = np.concatenate(lat, 0)
    y = np.array(lab)

    # PCA to 2D (numpy SVD; no sklearn dependency)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    ev = (S[:2] ** 2) / (S ** 2).sum()

    # separability: ratio of between- to within-gloss scatter on the 2D proj
    gm = P.mean(0)
    sb = sw = 0.0
    for g in glosses:
        idx = y == g2id[g]
        if idx.sum() < 2:
            continue
        c = P[idx].mean(0)
        sb += idx.sum() * ((c - gm) ** 2).sum()
        sw += ((P[idx] - c) ** 2).sum()
    sep = sb / (sw + 1e-9)
    print(f"[latent-csl] PCA EV(2)={ev.sum():.3f} between/within={sep:.3f}")

    plt.figure(figsize=(4.2, 3.4))
    cmap = plt.get_cmap("tab10")
    for k, g in enumerate(glosses):
        idx = y == g2id[g]
        plt.scatter(P[idx, 0], P[idx, 1], s=8, alpha=0.6, color=cmap(k),
                    label=GLOSS_LABEL.get(g, g[:12]), edgecolors="none")
    plt.xlabel(f"PC1 ({ev[0]*100:.0f}%)"); plt.ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    plt.legend(fontsize=5, markerscale=1.5, ncol=2, loc="best", framealpha=0.6,
               prop={"size": 5})
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"[latent-csl] saved -> {OUT}")


if __name__ == "__main__":
    main()
