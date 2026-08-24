"""Multi-objective JEPA-relative candidate ranker for Sign-JEPA.

Contribution framing:

    JEPA semantic carrier  +  candidate motion-memory generators
        +  a JEPA-relative multi-objective ranker

The ranker picks one candidate output per clip.  It is trained on a
*multi-objective* reward that combines three families of objective:

  recognition   - token edit similarity + token F1 + BLEU-1/2/3/4 n-gram
                  dominance terms
                  against the back-translation text predictions
  motion        - hand-speed band reward + hand-jerk stability penalty
  carrier       - body/face drift from the JEPA carrier + a total-distance
                  geometric proxy

The reward (the *training label*) is allowed to look at dev ground truth
(text + pose).  The ranker *features* are strictly JEPA-relative and
candidate-only, so the ranker is deployable on test where no ground truth
exists:

  - method one-hot
  - target length / gloss count
  - candidate hand/body/face speed, jerk, hand/body std
  - candidate-vs-JEPA full / hand / body / face distance
  - candidate-vs-JEPA hand/body/face speed ratio, jerk ratio, hand-std ratio
  - duration ratio vs the JEPA carrier and vs the manifest target

The model is a small MLP, trained listwise (soft cross-entropy to a
reward-softmax target + a reward-regression term).  We K-fold the dev clips
to report honest held-out choice accuracy / chosen reward, and deploy the
bagged K-fold ensemble.

`compose` applies the ranker, then enforces motion-feasibility fallbacks:
  - raw choice == jepa            -> jepa fails the motion gate; use a
                                     motion-feasible carrier-locked candidate
  - raw choice == memory_only and JEPA drift too high -> hierarchical
  - chosen candidate predicted hand-speed too low      -> hierarchical/locked

Usage
-----
train::

    python scripts/sign_jepa_multiobjective_selector.py train \
      --manifest_json data/phoenix/phoenix_dev.json \
      --gt_pt external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
      --carrier_pt external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_generator_manifest_dev_trainpolicy.pt \
      --candidate jepa=... --candidate refiner=... ... \
      --text_pred jepa=... --text_pred refiner=... ... \
      --out_ckpt outputs/sota_chase/sign_jepa_multiobjective_selector/dev_ranker.pt

compose::

    python scripts/sign_jepa_multiobjective_selector.py compose \
      --ckpt outputs/sota_chase/sign_jepa_multiobjective_selector/dev_ranker.pt \
      --manifest_json data/phoenix/phoenix_test.json \
      --carrier_pt external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_generator_manifest_test_trainpolicy.pt \
      --candidate jepa=... ... \
      --out_pt external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_multiobjective_selector_test.pt \
      --trace_json outputs/sota_chase/sign_jepa_multiobjective_selector/test_trace.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_candidate_selector import Selector  # noqa: E402
from scripts.sign_jepa_motion_texture_retrieval import as_pose, tokens  # noqa: E402
from scripts.train_sign_jepa_ae_fullclip import hand_jerk, region_speed  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

HAND = slice(8, 50)
BODY = slice(0, 8)
FACE = slice(50, 178)

# Candidates that pass the motion gate and stay carrier-locked.  Used as
# motion-feasibility fallbacks (the raw `jepa` candidate fails the motion gate).
DEFAULT_JEPA_FALLBACK = "source_locked"
DEFAULT_LOWSPEED_FALLBACK = "hierarchical"
DEFAULT_DRIFT_FALLBACK = "hierarchical"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def torch_load_cpu(path: Path):
    """Load CPU tensors across Torch versions.

    Newer Torch accepts `weights_only`; the repo's NSLT environment is pinned to
    Torch 1.10, which rejects that keyword.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def parse_kv(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def load_manifest(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in json.loads(path.read_text())}


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch_load_cpu(path)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def load_split_gt(path: Path) -> tuple[dict[str, str], dict[str, torch.Tensor], list[str]]:
    """Returns (text_map, pose_map, ordered_ids) from an organizer split .pt."""
    data = torch_load_cpu(path)
    text_map: dict[str, str] = {}
    pose_map: dict[str, torch.Tensor] = {}
    order: list[str] = []
    for sid, v in data.items():
        sid = str(sid)
        order.append(sid)
        text_map[sid] = str(v["text"]) if isinstance(v, dict) else ""
        try:
            pose_map[sid] = as_pose(v).contiguous()
        except Exception:
            pass
    return text_map, pose_map, order


def load_text_preds(path: Path, ordered_ids: list[str]) -> dict[str, str]:
    data = torch_load_cpu(path)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        return {sid: str(txt) for sid, txt in zip(ordered_ids, data)}
    raise TypeError(f"unsupported text prediction format: {type(data)}")


def load_coverage(path: Path | None) -> dict[str, float]:
    """Per-clip retrieval coverage from a hierarchical-memory trace JSON."""
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text())
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    return {str(r["id"]): float(r.get("coverage", 0.0)) for r in rows}


# --------------------------------------------------------------------------
# Text reward (recognition objective)
# --------------------------------------------------------------------------
def _toks(s: str) -> list[str]:
    return s.lower().strip().split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if x == y else 1)))
        prev = cur
    return prev[-1]


def semantic_reward(hyp: str, ref: str) -> float:
    """Token edit similarity + token F1 (WER-side + recall-side recognition)."""
    h, r = _toks(hyp), _toks(ref)
    if not r:
        return 0.0
    ed_sim = 1.0 - edit_distance(h, r) / max(len(h), len(r), 1)
    hs, rs = set(h), set(r)
    inter = len(hs & rs)
    prec = inter / max(1, len(hs))
    rec = inter / max(1, len(rs))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return 0.5 * ed_sim + 0.5 * f1


def bleu_order_rewards(hyp: str, ref: str, max_order: int = 4) -> dict[str, float]:
    """BLEU-like clipped n-gram precision per order with one brevity penalty.

    We keep the orders separate because the SLRTP leaderboard reports BLEU-1
    through BLEU-4, and optimizing a collapsed BLEU proxy can improve BLEU-4
    while silently losing BLEU-1/2/3 against a MoE baseline.
    """
    h, r = _toks(hyp), _toks(ref)
    if not h or not r:
        return {f"bleu{i}": 0.0 for i in range(1, max_order + 1)}

    def ngram_counts(toks: list[str], n: int) -> Counter:
        return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

    def clipped_precision(n: int) -> float:
        hn, rn = ngram_counts(h, n), ngram_counts(r, n)
        if not hn:
            return 0.0
        match = sum(min(c, rn.get(g, 0)) for g, c in hn.items())
        return match / max(1, sum(hn.values()))

    bp = 1.0 if len(h) >= len(r) else math.exp(1.0 - len(r) / max(1, len(h)))
    return {f"bleu{i}": bp * clipped_precision(i) for i in range(1, max_order + 1)}


def bleu_like_reward(hyp: str, ref: str) -> float:
    """Backward-compatible collapsed BLEU proxy, now using all four orders."""
    vals = bleu_order_rewards(hyp, ref)
    return 0.35 * vals["bleu1"] + 0.25 * vals["bleu2"] + 0.20 * vals["bleu3"] + 0.20 * vals["bleu4"]


# --------------------------------------------------------------------------
# Pose / motion helpers
# --------------------------------------------------------------------------
def resize_to(pose: torch.Tensor, length: int) -> torch.Tensor:
    if pose.shape[0] == length:
        return pose
    length = max(2, int(length))
    return resize_seq(pose.reshape(pose.shape[0], -1), length).reshape(length, 178, 3)


def region_speeds(pose: torch.Tensor) -> dict[str, float]:
    if pose.shape[0] < 2:
        return {"hand": 0.0, "body": 0.0, "face": 0.0}
    return region_speed(pose[None].float())


def jerk_of(pose: torch.Tensor) -> float:
    if pose.shape[0] < 3:
        return 0.0
    return float(hand_jerk(pose[None].float()))


def ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-9)


# --------------------------------------------------------------------------
# JEPA-relative, candidate-only features
# --------------------------------------------------------------------------
def candidate_features(
    pose: torch.Tensor,
    carrier: torch.Tensor | None,
    target_len: int,
    gloss_count: int,
    method_idx: int,
    n_methods: int,
    coverage: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (feature_vector, signal_dict).

    The feature vector never sees ground truth.  `signal_dict` carries a few
    raw JEPA-relative quantities used by the deterministic fallback rules.
    """
    pose = pose.float()
    T = max(1, int(pose.shape[0]))
    target_len = max(1, int(target_len))

    onehot = torch.zeros(n_methods)
    onehot[method_idx] = 1.0

    sp = region_speeds(pose)
    jk = jerk_of(pose)
    manifest_block = torch.tensor([
        T / target_len,
        target_len / T,
        abs(T - target_len) / max(T, target_len),
        gloss_count / 12.0,
        sp["hand"],
        sp["body"],
        sp["face"],
        jk,
        float(pose[:, HAND].std()),
        float(pose[:, BODY].std()),
    ], dtype=torch.float32)

    if carrier is None:
        carrier_block = torch.zeros(13, dtype=torch.float32)
        signals = {"hand_speed_ratio_jepa": 1.0, "full_dist_jepa": 0.0}
    else:
        carrier = carrier.float()
        Lc = max(2, int(carrier.shape[0]))
        p = resize_to(pose, Lc)
        csp = region_speeds(carrier)
        cjk = jerk_of(carrier)
        full_d = float((p - carrier).norm(dim=-1).mean())
        hand_d = float((p[:, HAND] - carrier[:, HAND]).norm(dim=-1).mean())
        body_d = float((p[:, BODY] - carrier[:, BODY]).norm(dim=-1).mean())
        face_d = float((p[:, FACE] - carrier[:, FACE]).norm(dim=-1).mean())
        hs_ratio = ratio(sp["hand"], csp["hand"])
        carrier_block = torch.tensor([
            T / Lc,
            Lc / T,
            full_d,
            hand_d,
            body_d,
            face_d,
            hs_ratio,
            ratio(sp["body"], csp["body"]),
            ratio(sp["face"], csp["face"]),
            ratio(jk, cjk),
            ratio(float(pose[:, HAND].std()), float(carrier[:, HAND].std())),
            ratio(float(pose[:, BODY].std()), float(carrier[:, BODY].std())),
            ratio(float(pose[:, FACE].std()), float(carrier[:, FACE].std())),
        ], dtype=torch.float32)
        signals = {"hand_speed_ratio_jepa": hs_ratio, "full_dist_jepa": full_d}

    feat = torch.cat([onehot, manifest_block, carrier_block, torch.tensor([coverage], dtype=torch.float32)])
    return feat, signals


# --------------------------------------------------------------------------
# Multi-objective reward (the training label; may use dev ground truth)
# --------------------------------------------------------------------------
def speed_band_reward(hs: float) -> float:
    """Healthy in [0.85, 1.20]; strong penalty for collapse, mild for excess."""
    if hs < 0.85:
        return (hs - 0.85) * 3.0
    if hs <= 1.20:
        return 1.0
    return max(-1.5, 1.0 - (hs - 1.20) * 1.5)


def zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


class RewardWeights:
    """Weights for the multi-objective reward (CLI-settable, one principled set)."""

    def __init__(
        self,
        sem=1.0,
        bleu=1.0,
        motion=0.6,
        drift=0.3,
        jerk=0.3,
        dist=0.4,
        bleu1=0.35,
        bleu2=0.25,
        bleu3=0.20,
        bleu4=0.20,
        dominance=0.0,
    ):
        self.sem, self.bleu, self.motion = sem, bleu, motion
        self.drift, self.jerk, self.dist = drift, jerk, dist
        self.bleu1, self.bleu2, self.bleu3, self.bleu4 = bleu1, bleu2, bleu3, bleu4
        self.dominance = dominance

    def to_dict(self) -> dict[str, float]:
        return {"sem": self.sem, "bleu": self.bleu, "motion": self.motion,
                "drift": self.drift, "jerk": self.jerk, "dist": self.dist,
                "bleu1": self.bleu1, "bleu2": self.bleu2, "bleu3": self.bleu3,
                "bleu4": self.bleu4, "dominance": self.dominance}


def build_reward_terms(rows, gt_text, gt_pose, text_preds, carrier_map):
    """Per (clip, candidate) raw reward terms.  Mutates `rows` in place.

    raw terms (all dev-only, label-side):
      sem   - semantic_reward(candidate text pred, GT text)
      bleu  - collapsed BLEU proxy across orders 1..4
      bleu1..4 - clipped n-gram BLEU-order rewards kept separate so the
                 ranker can be tuned to beat every USTC-MoE BLEU column
      dominance - worst normalized BLEU-order slack against configured floors
      motion- speed_band_reward(candidate-vs-GT hand-speed ratio)
      drift - body+face drift from the JEPA carrier             (penalty)
      jerk  - relu(candidate-vs-GT hand-jerk ratio - 1.6)        (penalty)
      dist  - candidate-vs-GT mean-joint distance (DTW-MJE proxy)(penalty)
    """
    flat = {k: [] for k in ("sem", "bleu", "bleu1", "bleu2", "bleu3", "bleu4",
                            "dominance", "motion", "drift", "jerk", "dist")}
    kept_rows = []
    for row in rows:
        sid = row["id"]
        if sid not in gt_text or sid not in gt_pose:
            continue
        gt_p = gt_pose[sid].float()
        carrier = carrier_map.get(sid)
        ref_text = gt_text[sid]
        gt_hand_sp = region_speeds(gt_p)["hand"]
        gt_jk = jerk_of(gt_p)
        for cand in row["cands"]:
            pose = cand["pose"].float()
            aligned = resize_to(pose, gt_p.shape[0])
            hs_ratio = ratio(region_speeds(aligned)["hand"], gt_hand_sp)
            jk_ratio = ratio(jerk_of(aligned), gt_jk)
            hyp = text_preds.get(cand["name"], {}).get(sid, "")
            sem = semantic_reward(hyp, ref_text)
            bleu_orders = bleu_order_rewards(hyp, ref_text)
            bleu = (0.35 * bleu_orders["bleu1"]
                    + 0.25 * bleu_orders["bleu2"]
                    + 0.20 * bleu_orders["bleu3"]
                    + 0.20 * bleu_orders["bleu4"])
            dominance = min(bleu_orders.values())
            motion = speed_band_reward(hs_ratio)
            jerk_pen = max(0.0, jk_ratio - 1.6)
            dist_pen = float((aligned - gt_p).norm(dim=-1).mean())
            if carrier is not None:
                cj = resize_to(pose, carrier.shape[0])
                drift = 0.5 * (
                    float((cj[:, BODY] - carrier[:, BODY].float()).norm(dim=-1).mean())
                    + float((cj[:, FACE] - carrier[:, FACE].float()).norm(dim=-1).mean())
                )
            else:
                drift = 0.0
            cand["terms_raw"] = {
                "sem": sem,
                "bleu": bleu,
                "bleu1": bleu_orders["bleu1"],
                "bleu2": bleu_orders["bleu2"],
                "bleu3": bleu_orders["bleu3"],
                "bleu4": bleu_orders["bleu4"],
                "dominance": dominance,
                "motion": motion,
                "drift": drift,
                "jerk": jerk_pen,
                "dist": dist_pen,
            }
            cand["hs_ratio_gt"] = hs_ratio
            cand["jk_ratio_gt"] = jk_ratio
            for k in flat:
                flat[k].append(cand["terms_raw"][k])
        kept_rows.append(row)
    z = {k: zscore(v) for k, v in flat.items()}
    return kept_rows, z


def finalize_reward(rows, z, weights: RewardWeights):
    """Combine globally z-scored terms into a scalar reward per candidate."""
    i = 0
    for row in rows:
        for cand in row["cands"]:
            r = (weights.sem * z["sem"][i]
                 + weights.bleu * z["bleu"][i]
                 + weights.bleu1 * z["bleu1"][i]
                 + weights.bleu2 * z["bleu2"][i]
                 + weights.bleu3 * z["bleu3"][i]
                 + weights.bleu4 * z["bleu4"][i]
                 + weights.dominance * z["dominance"][i]
                 + weights.motion * z["motion"][i]
                 - weights.drift * z["drift"][i]
                 - weights.jerk * z["jerk"][i]
                 - weights.dist * z["dist"][i])
            cand["reward"] = float(r)
            i += 1
        rewards = torch.tensor([c["reward"] for c in row["cands"]], dtype=torch.float32)
        if rewards.numel() > 1 and rewards.std() > 1e-9:
            row["reward_z"] = (rewards - rewards.mean()) / rewards.std()
        else:
            row["reward_z"] = torch.zeros_like(rewards)
        row["reward"] = rewards
        row["best_idx"] = int(torch.argmax(rewards))


# --------------------------------------------------------------------------
# Row construction
# --------------------------------------------------------------------------
def build_rows(manifest, candidates, carrier_map, coverage_map, method_index):
    """One row per clip; each row holds the per-candidate feature + pose."""
    pose_maps = [(name, load_pose_map(ROOT / path)) for name, path in candidates]
    rows = []
    for sid, mrow in manifest.items():
        target_len = int(mrow.get("length", 0) or 0)
        gloss_count = len(tokens(mrow))
        carrier = carrier_map.get(sid)
        cands = []
        for name, poses in pose_maps:
            if sid not in poses:
                continue
            pose = poses[sid]
            feat, sig = candidate_features(
                pose, carrier, target_len or pose.shape[0], gloss_count,
                method_index[name], len(method_index), coverage_map.get(sid, 0.0),
            )
            cands.append({"name": name, "pose": pose, "feature": feat, "signals": sig})
        if cands:
            rows.append({
                "id": sid,
                "cands": cands,
                "features": torch.stack([c["feature"] for c in cands]),
            })
    return rows


# --------------------------------------------------------------------------
# Training (listwise MLP ranker, K-fold)
# --------------------------------------------------------------------------
def train_one_model(train_rows, mean, std, dim, hidden, epochs, lr, wd, reward_tau, mse_w):
    torch.manual_seed(0)
    model = Selector(dim, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        order = torch.randperm(len(train_rows))
        for idx in order:
            row = train_rows[int(idx)]
            feats = (row["features"] - mean) / std
            logits = model(feats)
            rz = row["reward_z"]
            if rz.numel() < 2:
                continue
            target = F.softmax(rz / reward_tau, dim=0)
            listnet = -(target * F.log_softmax(logits, dim=0)).sum()
            mse = F.mse_loss(logits, rz)
            loss = listnet + mse_w * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(model, rows, mean, std):
    correct = 0
    chosen, oracle, mean_r = [], [], []
    for row in rows:
        logits = model((row["features"] - mean) / std)
        pick = int(torch.argmax(logits))
        correct += int(pick == row["best_idx"])
        chosen.append(float(row["reward"][pick]))
        oracle.append(float(row["reward"].max()))
        mean_r.append(float(row["reward"].mean()))
    n = max(1, len(rows))
    return {
        "choice_acc": correct / n,
        "avg_chosen_reward": sum(chosen) / n,
        "avg_oracle_reward": sum(oracle) / n,
        "avg_mean_reward": sum(mean_r) / n,
    }


def calibrate_fallbacks(rows):
    """Dev-calibrated thresholds for the deterministic motion fallbacks."""
    fail_j, pass_j, mem_drift = [], [], []
    for row in rows:
        for cand in row["cands"]:
            j = cand["signals"]["hand_speed_ratio_jepa"]
            if cand["hs_ratio_gt"] < 0.85:
                fail_j.append(j)
            else:
                pass_j.append(j)
            if cand["name"] == "memory_only":
                mem_drift.append(cand["signals"]["full_dist_jepa"])
    fail_j = fail_j or [1.0]
    pass_j = pass_j or [3.0]
    # threshold below which a candidate is treated as motion-infeasible
    lowspeed_tau = float(np.clip(np.quantile(fail_j, 0.90), 1.2,
                                 max(1.3, np.quantile(pass_j, 0.10))))
    drift_tau = float(np.quantile(mem_drift, 0.20)) if mem_drift else 0.0
    return {
        "lowspeed_tau": lowspeed_tau,
        "drift_tau": drift_tau,
        "fail_j_q": [float(np.quantile(fail_j, q)) for q in (0.5, 0.9)],
        "pass_j_q": [float(np.quantile(pass_j, q)) for q in (0.1, 0.5)],
    }


def train(args) -> None:
    weights = RewardWeights(args.w_sem, args.w_bleu, args.w_motion,
                            args.w_drift, args.w_jerk, args.w_dist,
                            args.w_bleu1, args.w_bleu2, args.w_bleu3,
                            args.w_bleu4, args.w_bleu_dominance)
    candidates = [parse_kv(x) for x in args.candidate]
    method_index = {name: i for i, (name, _) in enumerate(candidates)}
    manifest = load_manifest(ROOT / args.manifest_json)
    carrier_map = load_pose_map(ROOT / args.carrier_pt) if args.carrier_pt else {}
    coverage_map = load_coverage(ROOT / args.coverage_trace if args.coverage_trace else None)
    gt_text, gt_pose, order = load_split_gt(ROOT / args.gt_pt)
    text_preds = {name: load_text_preds(ROOT / path, order) for name, path in
                  (parse_kv(x) for x in args.text_pred)}

    rows = build_rows(manifest, candidates, carrier_map, coverage_map, method_index)
    rows, z = build_reward_terms(rows, gt_text, gt_pose, text_preds, carrier_map)
    finalize_reward(rows, z, weights)
    print(f"[train] usable dev clips with reward: {len(rows)}", flush=True)

    # reward diagnostics: which candidate the reward argmax prefers
    pref = Counter(row["cands"][row["best_idx"]]["name"] for row in rows)
    print(f"[train] reward-argmax candidate distribution: {dict(pref)}", flush=True)
    per_method = {name: [] for name in method_index}
    for row in rows:
        for c in row["cands"]:
            per_method[c["name"]].append(c["reward"])
    print("[train] mean reward per candidate: " + json.dumps(
        {k: round(float(np.mean(v)), 3) for k, v in per_method.items() if v}), flush=True)

    X = torch.cat([row["features"] for row in rows], dim=0)
    mean = X.mean(0)
    std = X.std(0).clamp_min(1e-6)
    dim = X.shape[1]

    # ---- K-fold honest held-out evaluation -------------------------------
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(rows), generator=g).tolist()
    folds = [perm[k::args.kfolds] for k in range(args.kfolds)]
    # clip id -> the fold that held it out, so compose can do honest
    # out-of-fold prediction on dev (and the full ensemble on unseen test).
    fold_of = {rows[i]["id"]: k for k in range(args.kfolds) for i in folds[k]}
    fold_metrics = []
    fold_models = []
    for k in range(args.kfolds):
        val_ids = set(folds[k])
        tr = [rows[i] for i in range(len(rows)) if i not in val_ids]
        va = [rows[i] for i in folds[k]]
        model = train_one_model(tr, mean, std, dim, args.hidden, args.epochs,
                                args.lr, args.weight_decay, args.reward_tau, args.reward_mse_weight)
        m = evaluate_model(model, va, mean, std)
        fold_metrics.append(m)
        fold_models.append(model)
        print(f"[fold {k}] held-out " + json.dumps({kk: round(vv, 4) for kk, vv in m.items()}), flush=True)

    agg = {kk: round(float(np.mean([m[kk] for m in fold_metrics])), 4) for kk in fold_metrics[0]}
    print(f"[cv] mean held-out: {json.dumps(agg)}", flush=True)

    # ---- final full-data model + bagged ensemble -------------------------
    full_model = train_one_model(rows, mean, std, dim, args.hidden, args.epochs,
                                 args.lr, args.weight_decay, args.reward_tau, args.reward_mse_weight)
    full_train = evaluate_model(full_model, rows, mean, std)
    print(f"[full] in-sample: {json.dumps({k: round(v, 4) for k, v in full_train.items()})}", flush=True)

    fb = calibrate_fallbacks(rows)
    print(f"[calib] fallback thresholds: {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in fb.items()})}", flush=True)

    out = ROOT / args.out_ckpt
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fold_models": [m.state_dict() for m in fold_models],
        "fold_of": fold_of,
        "full_model": full_model.state_dict(),
        "mean": mean,
        "std": std,
        "dim": dim,
        "hidden": args.hidden,
        "method_index": method_index,
        "reward_weights": weights.to_dict(),
        "fallbacks": fb,
        "cv_metrics": agg,
        "fold_metrics": fold_metrics,
        "full_train_metrics": full_train,
        "reward_argmax_dist": dict(pref),
    }, out)
    print(f"saved -> {out}", flush=True)


# --------------------------------------------------------------------------
# Compose (deployable selection + motion-feasibility fallbacks)
# --------------------------------------------------------------------------
@torch.no_grad()
def compose(args) -> None:
    ckpt = torch_load_cpu(ROOT / args.ckpt)
    method_index = ckpt["method_index"]
    candidates = [parse_kv(x) for x in args.candidate]
    for name, _ in candidates:
        if name not in method_index:
            raise ValueError(f"candidate '{name}' not in trained method set {list(method_index)}")
    manifest = load_manifest(ROOT / args.manifest_json)
    carrier_map = load_pose_map(ROOT / args.carrier_pt) if args.carrier_pt else {}
    coverage_map = load_coverage(ROOT / args.coverage_trace if args.coverage_trace else None)
    rows = build_rows(manifest, candidates, carrier_map, coverage_map, method_index)

    mean, std = ckpt["mean"], ckpt["std"].clamp_min(1e-6)

    def _load(state):
        m = Selector(ckpt["dim"], ckpt["hidden"])
        m.load_state_dict(state)
        m.eval()
        return m

    fold_models = [_load(s) for s in ckpt.get("fold_models", [])]
    full_model = _load(ckpt["full_model"])
    fold_of = ckpt.get("fold_of", {})
    oof = {"count": 0}

    def models_for(sid: str):
        # honest out-of-fold prediction for dev clips the ranker trained on;
        # bagged fold ensemble for genuinely unseen (test) clips.
        if args.use_full_model or not fold_models:
            return [full_model]
        if sid in fold_of:
            oof["count"] += 1
            return [fold_models[fold_of[sid]]]
        return fold_models

    fb = ckpt["fallbacks"]
    lowspeed_tau = args.lowspeed_tau if args.lowspeed_tau > 0 else fb["lowspeed_tau"]
    drift_tau = args.drift_tau if args.drift_tau > 0 else fb["drift_tau"]

    pose_by_name: dict[str, dict[str, torch.Tensor]] = {}
    for row in rows:
        for cand in row["cands"]:
            pose_by_name.setdefault(cand["name"], {})[row["id"]] = cand["pose"]

    def pick_fallback(sid: str, preferred: list[str]) -> str | None:
        for name in preferred:
            if name in pose_by_name and sid in pose_by_name[name]:
                return name
        return None

    out: dict[str, torch.Tensor] = {}
    trace = []
    raw_counts, final_counts, rule_counts = Counter(), Counter(), Counter()
    for row in rows:
        sid = row["id"]
        feats = (row["features"] - mean) / std
        logits = torch.stack([m(feats) for m in models_for(sid)]).mean(0)
        names = [c["name"] for c in row["cands"]]
        sig = {c["name"]: c["signals"] for c in row["cands"]}
        raw_choice = names[int(torch.argmax(logits))]
        choice = raw_choice
        rule = "none"

        # rule 1: raw JEPA fails the hand-speed motion gate -> carrier-locked candidate
        if choice == "jepa":
            fb_name = pick_fallback(sid, [args.jepa_fallback, DEFAULT_LOWSPEED_FALLBACK, "source_locked"])
            if fb_name:
                choice, rule = fb_name, "jepa_motion_fallback"

        # rule 2: memory_only ignores the JEPA carrier; replace when drift too high
        if choice == "memory_only" and sig["memory_only"]["full_dist_jepa"] > drift_tau:
            fb_name = pick_fallback(sid, [args.drift_fallback, "source_locked"])
            if fb_name:
                choice, rule = fb_name, "memory_drift_fallback"

        # rule 3: chosen candidate predicted hand-speed too low -> motion-feasible candidate
        if choice in sig and sig[choice]["hand_speed_ratio_jepa"] < lowspeed_tau:
            fb_name = pick_fallback(sid, [args.lowspeed_fallback, "source_locked", "hierarchical"])
            if fb_name and fb_name != choice:
                choice, rule = fb_name, (rule + "+lowspeed" if rule != "none" else "lowspeed_fallback")

        if sid not in pose_by_name.get(choice, {}):
            choice, rule = raw_choice, "fallback_unavailable"

        out[sid] = pose_by_name[choice][sid].contiguous()
        raw_counts[raw_choice] += 1
        final_counts[choice] += 1
        rule_counts[rule] += 1
        trace.append({
            "id": sid,
            "raw_choice": raw_choice,
            "choice": choice,
            "fallback_rule": rule,
            "scores": {n: round(float(v), 4) for n, v in zip(names, logits)},
            "hand_speed_ratio_jepa": round(float(sig.get(choice, {}).get("hand_speed_ratio_jepa", 0.0)), 4),
            "full_dist_jepa": round(float(sig.get(choice, {}).get("full_dist_jepa", 0.0)), 4),
        })

    if args.filter_keys_pt:
        keys = [str(k) for k in torch_load_cpu(ROOT / args.filter_keys_pt).keys()]
        out = {sid: out[sid] for sid in keys if sid in out}
        trace = [t for t in trace if t["id"] in out]

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}", flush=True)
    print(f"[compose] raw choices:   {json.dumps(dict(raw_counts))}", flush=True)
    print(f"[compose] final choices: {json.dumps(dict(final_counts))}", flush=True)
    print(f"[compose] fallback rules:{json.dumps(dict(rule_counts))}", flush=True)
    print(f"[compose] lowspeed_tau={lowspeed_tau:.4f} drift_tau={drift_tau:.4f}", flush=True)
    mode = "full-model" if args.use_full_model else f"out-of-fold={oof['count']}, ensemble={len(rows) - oof['count']}"
    print(f"[compose] prediction mode: {mode}", flush=True)

    if args.trace_json:
        trace_path = ROOT / args.trace_json
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({
            "raw_choices": dict(raw_counts),
            "final_choices": dict(final_counts),
            "fallback_rules": dict(rule_counts),
            "lowspeed_tau": lowspeed_tau,
            "drift_tau": drift_tau,
            "items": trace,
        }, indent=2))
        print(f"saved trace -> {trace_path}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--manifest_json", required=True)
    tr.add_argument("--gt_pt", required=True)
    tr.add_argument("--carrier_pt", required=True)
    tr.add_argument("--candidate", action="append", required=True)
    tr.add_argument("--text_pred", action="append", required=True)
    tr.add_argument("--out_ckpt", required=True)
    tr.add_argument("--coverage_trace", default="")
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--kfolds", type=int, default=5)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--reward_tau", type=float, default=0.5)
    tr.add_argument("--reward_mse_weight", type=float, default=0.3)
    tr.add_argument("--w_sem", type=float, default=1.0)
    tr.add_argument("--w_bleu", type=float, default=1.0)
    tr.add_argument("--w_bleu1", type=float, default=0.35)
    tr.add_argument("--w_bleu2", type=float, default=0.25)
    tr.add_argument("--w_bleu3", type=float, default=0.20)
    tr.add_argument("--w_bleu4", type=float, default=0.20)
    tr.add_argument("--w_bleu_dominance", type=float, default=0.0,
                    help="Extra reward on the weakest per-order BLEU proxy. "
                         "Use >0 when the target is to beat every USTC-MoE BLEU column.")
    tr.add_argument("--w_motion", type=float, default=0.6)
    tr.add_argument("--w_drift", type=float, default=0.3)
    tr.add_argument("--w_jerk", type=float, default=0.3)
    tr.add_argument("--w_dist", type=float, default=0.4)

    co = sub.add_parser("compose")
    co.add_argument("--ckpt", required=True)
    co.add_argument("--manifest_json", required=True)
    co.add_argument("--carrier_pt", required=True)
    co.add_argument("--candidate", action="append", required=True)
    co.add_argument("--out_pt", required=True)
    co.add_argument("--trace_json", default="")
    co.add_argument("--coverage_trace", default="")
    co.add_argument("--filter_keys_pt", default="")
    co.add_argument("--use_full_model", action="store_true")
    co.add_argument("--jepa_fallback", default=DEFAULT_JEPA_FALLBACK)
    co.add_argument("--drift_fallback", default=DEFAULT_DRIFT_FALLBACK)
    co.add_argument("--lowspeed_fallback", default=DEFAULT_LOWSPEED_FALLBACK)
    co.add_argument("--lowspeed_tau", type=float, default=0.0, help="0 = use dev-calibrated value")
    co.add_argument("--drift_tau", type=float, default=0.0, help="0 = use dev-calibrated value")

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        compose(args)


if __name__ == "__main__":
    main()
