"""Score a clip ranking against the copyable label (GraspGrade number 2 benchmark).

    python score_number2.py predictions.csv                  # held-out clips
    python score_number2.py predictions.csv --split tune
    python score_number2.py predictions.csv --label copyable_depth_2mm

predictions.csv: columns clip_id, score. Higher score means "more likely a
robot hand can copy this grasp". Every clip of the split needs a score.

Metrics:
  AUC with a 95% interval that resamples whole sequences (1,000 draws, seed
      20261026), because clips from one sequence are related,
  AUC per person and their mean (copyable shares differ between people, so a
      pooled AUC partly ranks people),
  share of copyable clips in the top 20% against the base rate,
  AUC on one-hand clips only.
Needs numpy, pandas and scikit-learn.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ci_by_sequence(y, s, seqs, seed=20261026, n=1000):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx = {q: np.where(seqs == q)[0] for q in uniq}
    boots = []
    for _ in range(n):
        i = np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def score(d, label="copyable"):
    y, s, seqs = d[label].values, d.score.values, d.sequence.values
    per = {p: round(float(roc_auc_score(g[label], g.score)), 3) for p, g in d.groupby("person") if g[label].nunique() == 2}
    top = s >= np.percentile(s, 80)
    one = ~d.two_hands.values.astype(bool)
    return {"clips": int(len(d)), "sequences": int(len(np.unique(seqs))), "label": label,
            "auc": round(float(roc_auc_score(y, s)), 3), "ci95_by_sequence": ci_by_sequence(y, s, seqs),
            "auc_per_person": per, "auc_per_person_mean": round(float(np.mean(list(per.values()))), 3),
            "top20_copyable_share": round(float(y[top].mean()), 3), "base_rate": round(float(y.mean()), 3),
            "auc_one_hand_clips": round(float(roc_auc_score(y[one], s[one])), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", default="held_out", choices=["held_out", "tune"])
    ap.add_argument("--role", help="score only clips of this role (score_train, item_train, held_out)")
    ap.add_argument("--label", default="copyable", choices=["copyable", "copyable_depth_2mm", "copyable_depth_10mm"])
    ap.add_argument("--clips", default=str(Path(__file__).resolve().parent / "number2" / "clips.csv"))
    args = ap.parse_args()
    clips = pd.read_csv(args.clips)
    clips = clips[clips.role == args.role] if args.role else clips[clips.split == args.split]
    pred = pd.read_csv(args.predictions)[["clip_id", "score"]]
    d = clips.merge(pred, on="clip_id", how="left")
    if d.score.isna().any():
        raise SystemExit(f"{int(d.score.isna().sum())} of {len(d)} clips have no score")
    print(json.dumps(score(d, args.label), indent=1))


if __name__ == "__main__":
    main()
