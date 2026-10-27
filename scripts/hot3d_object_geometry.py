"""Option A check (docs/MEASUREMENT_PLAN.md): can object geometry predict the v2 copyable label? No API call.

    python scripts/hot3d_object_geometry.py

Per object, from its 3D model: oriented bounding box sides (largest, middle,
smallest), flatness, elongation, volume, fill. Logistic regression on tuning
clips (v2 label), tested two ways:
  1. leave one person out, AUC per person and the mean over the six people,
  2. leave one object out, pooled AUC with a 95% interval over sequences.
The object identity reference is computed the same ways (it cannot score an
unseen object, so it is 0.5 in test 2 by construction).
Writes out/hot3d_checks/object_geometry.json and docs/experiments/img/exp18_object_geometry.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ASSETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
LAB = ROOT / "out" / "hot3d_copyable" / "tune_v2_clips_labeled.csv"
FEATS = ["side_large_cm", "side_middle_cm", "side_small_cm", "flatness", "elongation", "volume_cm3", "fill"]


def geometry(uid):
    m = trimesh.load(ASSETS / f"{uid}.glb", force="mesh")
    ext = np.sort(m.bounding_box_oriented.extents)[::-1]
    vol = abs(m.volume) if m.is_watertight else m.convex_hull.volume
    return {"side_large_cm": 100 * ext[0], "side_middle_cm": 100 * ext[1], "side_small_cm": 100 * ext[2],
            "flatness": ext[2] / ext[0], "elongation": ext[1] / ext[0], "volume_cm3": 1e6 * vol,
            "fill": vol / float(np.prod(ext)), "watertight": bool(m.is_watertight)}


def object_table(names):
    uid_of = {}
    for meta in ARIA.glob("*/ground_truth/metadata.json"):
        m = json.loads(meta.read_text())
        uid_of.update(dict(zip(m["object_names"], m["object_uids"])))
    rows = {n: geometry(int(uid_of[n])) for n in names}
    return pd.DataFrame(rows).T


def fit_predict(tr, te):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    m.fit(tr[FEATS].astype(float), tr.copyable)
    return m.predict_proba(te[FEATS].astype(float))[:, 1]


def boot_ci(t, s, n=1000, seed=20261026):
    rng = np.random.default_rng(seed)
    seqs = t.sequence.unique()
    idx = {q: np.where(t.sequence.values == q)[0] for q in seqs}
    vals = []
    for _ in range(n):
        i = np.concatenate([idx[q] for q in rng.choice(seqs, len(seqs))])
        if t.copyable.values[i].min() != t.copyable.values[i].max():
            vals.append(roc_auc_score(t.copyable.values[i], s[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    t = pd.read_csv(LAB)
    geo = object_table(sorted(t.object.unique()))
    t = t.join(geo[FEATS].astype(float), on="object")

    lopo_geo, lopo_id = np.zeros(len(t)), np.zeros(len(t))
    for p in sorted(t.person.unique()):
        tr, te = t.person != p, t.person == p
        lopo_geo[te.values] = fit_predict(t[tr], t[te])
        share = t[tr].groupby("object").copyable.mean()
        lopo_id[te.values] = t[te].object.map(share).fillna(t[tr].copyable.mean()).values
    per = lambda s: {p: round(float(roc_auc_score(g.copyable, s[g.index])), 3) for p, g in t.groupby("person")}
    pg, pi = per(lopo_geo), per(lopo_id)

    looo = np.zeros(len(t))
    for o in sorted(t.object.unique()):
        tr, te = t.object != o, t.object == o
        looo[te.values] = fit_predict(t[tr], t[te])
    looo_auc = round(float(roc_auc_score(t.copyable, looo)), 3)

    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(t[FEATS], t.copyable)
    coef = dict(zip(FEATS, np.round(m[-1].coef_[0], 3).tolist()))
    single = {f: round(float(roc_auc_score(t.copyable, t[f])), 3) for f in FEATS}
    g_mean, l = float(np.mean(list(pg.values()))), looo_auc
    verdict = "pass" if g_mean >= 0.65 and l >= 0.65 else ("fail" if g_mean < 0.60 or l < 0.60 else "weak")
    obj = t.groupby("object").agg(copyable=("copyable", "mean"), clips=("copyable", "size")).join(geo)
    out = {
        "clips": len(t), "objects": int(t.object.nunique()), "copyable_share": round(float(t.copyable.mean()), 3),
        "leave_one_person_out": {
            "geometry_auc_mean_over_people": round(g_mean, 3), "geometry_per_person": pg,
            "geometry_pooled": round(float(roc_auc_score(t.copyable, lopo_geo)), 3),
            "object_identity_auc_mean_over_people": round(float(np.mean(list(pi.values()))), 3),
            "object_identity_per_person": pi},
        "leave_one_object_out": {"geometry_auc_pooled": looo_auc, "geometry_ci95_by_sequence": boot_ci(t, looo),
                                 "object_identity_auc": 0.5},
        "single_feature_auc": single, "coefficients_all_tuning": coef, "verdict": verdict,
        "non_watertight_objects": [o for o, w in geo.watertight.items() if not w],
        "objects": obj.round(3).to_dict("index"),
    }
    (OUT / "object_geometry.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "objects"}, indent=1, default=float))
    plot(out, obj)


def plot(out, obj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    rows = [("object identity, leave person out", out["leave_one_person_out"]["object_identity_auc_mean_over_people"], "#868e96"),
            ("object geometry, leave person out", out["leave_one_person_out"]["geometry_auc_mean_over_people"], "#1c7ed6"),
            ("object identity, leave object out", 0.5, "#868e96"),
            ("object geometry, leave object out", out["leave_one_object_out"]["geometry_auc_pooled"], "#1c7ed6")]
    y = np.arange(len(rows))[::-1]
    for yi, (n, v, c) in zip(y, rows):
        axes[0].barh(yi, v - 0.5, left=0.5, color=c, height=0.6)
        axes[0].text(max(v, 0.5) + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    lo, hi = out["leave_one_object_out"]["geometry_ci95_by_sequence"]
    axes[0].plot([lo, hi], [y[3], y[3]], color="black", lw=1.5)
    axes[0].axvline(0.65, color="#2b8a3e", ls="--", lw=1, label="pass line 0.65")
    axes[0].axvline(0.60, color="#c92a2a", ls="--", lw=1, label="fail line 0.60")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8)
    axes[0].set_xlim(0.45, 0.85)
    axes[0].set_xlabel("AUC for copyable v2, tuning clips")
    axes[0].legend(fontsize=7, loc="lower right")
    sc = axes[1].scatter(obj.side_large_cm, obj.side_small_cm, s=20 + obj.clips / 3, c=obj.copyable,
                         cmap="RdYlGn", vmin=0, vmax=1, edgecolor="black", lw=0.5)
    for name, r in obj.iterrows():
        axes[1].annotate(name, (r.side_large_cm, r.side_small_cm), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("largest side of the object (cm)")
    axes[1].set_ylabel("smallest side of the object (cm)")
    axes[1].set_title("Copyable share per object (colour), dot size = clips")
    fig.colorbar(sc, ax=axes[1], label="copyable share")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp18_object_geometry.png", dpi=125)


if __name__ == "__main__":
    main()
