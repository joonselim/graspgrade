"""Number 2: train the score and run the comparisons of docs/MEASUREMENT_PLAN.md.

    python scripts/number2_score.py --check     # code check on tuning only: train item_train, test score_train
    python scripts/number2_score.py --final     # train score_train, test held_out (after all paid runs)

Inputs, per clip (out/number2/clips_final.csv):
  local checks   camera stability, action completeness (hot3d_local_checks.py)
  hand shape fit logistic regression on 10 MediaPipe hand features, trained on
                 item_train clips only, graded by its own tercile cut points
  SAM checks     visibility, contact clarity (number2_sam_checks.py, after SAM)
  Muse checklist finger occlusion and six object items (muse_hot3d.py --task checklist)
  Muse baseline  copyable probability (muse_hot3d.py --task baseline)
With --check, the Muse object items are replaced by the EXP-19 stand-in grades
and missing paid inputs are left out, so only the code path is tested.

Comparisons: our score vs Muse alone; object-only reference (copyable share
per object in the training clips); Muse alone vs Muse plus our checks.
Metrics: AUC with a 95% interval resampling sequences, AUC per person,
copyable share in the top 20% against the base rate, all clips and one-hand clips.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_hand_shape import FEATS as SHAPE, video_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
CHECKS = ROOT / "out" / "hot3d_checks"
KEY = ["sequence", "hand", "clip_start", "object"]
MUSE_ITEMS = ["finger_occlusion", "object_size", "object_shape", "object_thickness", "object_width",
              "object_handle", "free_space"]
STANDIN = {"object_size": "item_size", "object_shape": "item_shape", "object_thickness": "item_thickness",
           "object_width": "item_width", "object_handle": "item_handle", "free_space": "item_free_space"}


def terciles(train_scores):
    return np.percentile(train_scores, [33.3, 66.7])


def to_grade(x, cuts):
    return np.where(x >= cuts[1], 1.0, np.where(x >= cuts[0], 0.5, 0.0))


def hand_shape_fit(d, train_role):
    """The hand shape check: model and cut points from item_train clips only."""
    path = N2 / "hand_shape_video.csv"
    if path.exists():
        hs = pd.read_csv(path)
    else:
        parts = []
        for seq, g in d.groupby("sequence"):
            f = video_features(seq, g)
            parts.append(pd.DataFrame(f, columns=[f"v_{c}" for c in SHAPE]).assign(clip_id=g.clip_id.values))
        hs = pd.concat(parts)
        hs.to_csv(path, index=False)
    x = d[["clip_id", "role", "copyable"]].merge(hs, on="clip_id")
    cols = [f"v_{c}" for c in SHAPE]
    x[cols] = x[cols].fillna(x[x.role == train_role][cols].median())
    tr = x[x.role == train_role]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(tr[cols], tr.copyable)
    p = m.predict_proba(x[cols])[:, 1]
    cuts = terciles(p[(x.role == train_role).values])
    return pd.Series(to_grade(p, cuts), index=x.clip_id.values)


def assemble(check):
    d = pd.read_csv(N2 / "clips_final.csv")
    local = pd.read_csv(CHECKS / "clip_checks.csv")[KEY + ["grade_camera_stability", "grade_completeness"]]
    d = d.merge(local, on=KEY, how="left")
    d["grade_hand_shape_fit"] = d.clip_id.map(hand_shape_fit(d, "item_train"))
    items = ["grade_camera_stability", "grade_completeness", "grade_hand_shape_fit"]
    sam = N2 / "sam_checks.csv"
    if sam.exists():
        d = d.merge(pd.read_csv(sam), on="clip_id", how="left")
        items += ["grade_visibility", "grade_contact_clarity"]
    muse_dir = N2 / "muse" / "checklist"
    if check:
        st = pd.read_csv(CHECKS / "object_items.csv")[KEY + list(STANDIN.values())]
        d = d.merge(st, on=KEY, how="left").rename(columns={v: k for k, v in STANDIN.items()})
        items += list(STANDIN)
    elif muse_dir.exists():
        rows = []
        for f in muse_dir.glob("*.json"):
            r = json.loads(f.read_text())
            if r["parsed"]:
                rows.append({"clip_id": r["clip_id"], **{k: r["parsed"][k] for k in MUSE_ITEMS}})
        d = d.merge(pd.DataFrame(rows), on="clip_id", how="left")
        items += MUSE_ITEMS
    base = N2 / "muse" / "baseline"
    if base.exists():
        rows = [json.loads(f.read_text()) for f in base.glob("*.json")]
        b = pd.DataFrame([{"clip_id": r["clip_id"], "muse_baseline": r["parsed"]["copyable_probability"]}
                          for r in rows if r["parsed"]])
        d = d.merge(b, on="clip_id", how="left")
    return d, items


def metrics(d, score, seed=20261026):
    y, s = d.copyable.values, d[score].values
    ok = ~np.isnan(s)
    y, s, seqs = y[ok], s[ok], d.sequence.values[ok]
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx = {q: np.where(seqs == q)[0] for q in uniq}
    boots = []
    for _ in range(1000):
        i = np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    top = s >= np.percentile(s, 80)
    sub = d[ok]
    one = ~sub.two_hands.values.astype(bool)
    return {"clips": int(ok.sum()), "auc": round(float(roc_auc_score(y, s)), 3),
            "ci95_by_sequence": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "auc_per_person": {p: round(float(roc_auc_score(g.copyable, g[score])), 3)
                               for p, g in sub.groupby("person") if g.copyable.nunique() == 2},
            "top20_copyable_share": round(float(y[top].mean()), 3), "base_rate": round(float(y.mean()), 3),
            "auc_one_hand_clips": round(float(roc_auc_score(y[one], s[one])), 3)}


def fit_score(train, test, cols):
    m = LogisticRegression(C=0.5, max_iter=2000).fit(train[cols], train.copyable)
    return m.predict_proba(test[cols])[:, 1], dict(zip(cols, np.round(m.coef_[0], 3).tolist()))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--final", action="store_true")
    args = ap.parse_args()
    d, items = assemble(args.check)
    train_role, test_role = ("item_train", "score_train") if args.check else ("score_train", "held_out")
    train, test = d[d.role == train_role].copy(), d[d.role == test_role].copy()
    items = [c for c in items if train[c].notna().all() and test[c].notna().all()]
    out = {"mode": "check (tuning only, stand-ins)" if args.check else "final", "train": train_role,
           "test": test_role, "items": items}
    test["our_score"], out["weights"] = fit_score(train, test, items)
    out["our_score"] = metrics(test, "our_score")
    share = train.groupby("object").copyable.mean()
    test["object_reference"] = test.object.map(share).fillna(train.copyable.mean())
    out["object_reference"] = metrics(test, "object_reference")
    if "muse_baseline" in d and test.muse_baseline.notna().any() and train.muse_baseline.notna().any():
        out["muse_alone"] = metrics(test, "muse_baseline")
        tr, te = train.dropna(subset=["muse_baseline"]), test.dropna(subset=["muse_baseline"]).copy()
        te["plug_in"], out["plug_in_weights"] = fit_score(tr, te, ["muse_baseline"] + items)
        out["muse_plus_checks"] = metrics(te, "plug_in")
    name = "number2_check.json" if args.check else "number2_final.json"
    (N2 / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
