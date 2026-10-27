"""Before any paid call: how much of the copyable label can cheap signals predict? (tuning split only)

    python scripts/hot3d_signal_check.py

Uses out/hot3d_checks/clip_checks.csv (hot3d_local_checks.py). Held-out clips are not touched.
Leave one tuning person out (6 folds):
  - object-only reference: copyable share of each object in the other five
    people, given to the left-out person's clips (unseen object: overall share);
  - the three local checks in a logistic regression;
  - object share plus the three local checks.
Also the AUC of each local check inside each object, to see signal that
object identity does not explain. AUC intervals resample sequences.
Writes out/hot3d_checks/signal_check.json and docs/experiments/img/exp12_*.png.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
IMG = ROOT / "docs" / "experiments" / "img"
GRADES = ["grade_camera_stability", "grade_retargeting", "grade_completeness"]
RAW = ["camera_motion_p90_px", "retarget_value", "reach_seen", "release_seen", "hand_seen_in_contact"]


def boot_ci(df, score, n=1000, seed=20261026):
    rng = np.random.default_rng(seed)
    seqs = df.sequence.unique()
    by = {s: g for s, g in df.groupby("sequence")}
    vals = []
    for _ in range(n):
        g = pd.concat([by[s] for s in rng.choice(seqs, len(seqs))])
        if g.copyable.nunique() == 2:
            vals.append(roc_auc_score(g.copyable, g[score]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    d = pd.read_csv(OUT / "clip_checks.csv")
    t = d[d.split == "tune"].copy()
    t["copyable"] = t.copyable.astype(int)
    single = {c: round(float(roc_auc_score(t.copyable, t[c])), 3) for c in GRADES + RAW if t[c].notna().all()}

    t["p_object"] = np.nan
    t["p_local"] = np.nan
    t["p_both"] = np.nan
    folds = {}
    for p in sorted(t.person.unique()):
        tr, te = t[t.person != p], t[t.person == p]
        share = tr.groupby("object").copyable.mean()
        prior = te.object.map(share).fillna(tr.copyable.mean())
        t.loc[te.index, "p_object"] = prior
        lr = LogisticRegression(max_iter=1000).fit(tr[GRADES], tr.copyable)
        t.loc[te.index, "p_local"] = lr.predict_proba(te[GRADES])[:, 1]
        trp = tr.object.map(share)  # in-fold prior for training rows (slightly optimistic for the training side only)
        lr2 = LogisticRegression(max_iter=1000).fit(np.c_[trp, tr[GRADES]], tr.copyable)
        t.loc[te.index, "p_both"] = lr2.predict_proba(np.c_[prior, te[GRADES]])[:, 1]
        folds[p] = {"clips": len(te), "copyable_share": round(float(te.copyable.mean()), 3),
                    "object_only_auc": round(float(roc_auc_score(te.copyable, prior)), 3),
                    "unseen_objects": int(te.object.map(share).isna().sum())}
    pooled = {k: {"auc": round(float(roc_auc_score(t.copyable, t[k])), 3), "ci95_by_sequence": boot_ci(t, k)}
              for k in ("p_object", "p_local", "p_both")}

    per_person = {k: {p: round(float(roc_auc_score(g.copyable, g[k])), 3) for p, g in t.groupby("person")}
                  for k in ("p_object", "p_local", "p_both")}
    for k in per_person:
        pooled[k]["auc_mean_over_people"] = round(float(np.mean(list(per_person[k].values()))), 3)
        pooled[k]["auc_per_person"] = per_person[k]
    within = {}
    for c in GRADES:
        aucs, ws = [], []
        for _, g in t.groupby("object"):
            if g.copyable.nunique() == 2 and g[c].nunique() > 1:
                aucs.append(roc_auc_score(g.copyable, g[c]))
                ws.append(len(g))
        within[c] = round(float(np.average(aucs, weights=ws)), 3)

    out = {"tuning_clips": len(t), "copyable_share": round(float(t.copyable.mean()), 3),
           "single_check_auc": single, "leave_one_person_out": pooled, "per_person_object_only": folds,
           "within_object_auc_weighted": within}
    (OUT / "signal_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    plot(t, single, pooled)


def plot(t, single, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("camera stability (grade)", single["grade_camera_stability"], None),
            ("retargeting recoverability (grade)", single["grade_retargeting"], None),
            ("action completeness (grade)", single["grade_completeness"], None),
            ("3 local checks, logistic regression", pooled["p_local"]["auc_mean_over_people"], None),
            ("object-only reference", pooled["p_object"]["auc_mean_over_people"], None),
            ("object plus 3 local checks", pooled["p_both"]["auc_mean_over_people"], None)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (name, auc, ci) in zip(y, rows):
        c = "#868e96" if "object" not in name else "#1c7ed6"
        ax.barh(yi, auc - 0.5, left=0.5, color=c, height=0.6)
        if ci:
            ax.plot(ci, [yi, yi], color="black", lw=1.5)
        ax.text(max(auc, ci[1] if ci else auc) + 0.01, yi, f"{auc:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("AUC for copyable on tuning clips. Model rows: leave one person out, mean of 6 people")
    ax.set_title(f"What predicts the copyable label before any paid call ({len(t)} tuning clips)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "exp12_signal_check.png", dpi=130)


if __name__ == "__main__":
    main()
