"""Muse Spark errors on tuning, and the one prompt revision (EXP-30). Tuning only.

    python scripts/prompt_revision_check.py

1. Number 2 checklist: Muse grades against the EXP-19 stand-ins (grade tables),
   for version 1 (graded items) and version 2 (sides in cm, graded by our
   code). Version 2 cm estimates against the 3D model boxes. R2 and R3 leave
   one person out, R3 leave one object out. Muse alone (R1) behaviour.
2. Number 1 arm A: share of true releases (glove gaps) in which Muse says
   "not touching" at least once, by gap length, and boundary F1 with the shift
   learned on the other two episodes, for version 1 and version 2.
3. The rule from the measurement plan ("One prompt revision on tuning") picks
   the version for held-out.
Writes out/prompt_revision.json and docs/experiments/img/exp30_*.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_cut import SIDES, Score, ground_truth, segments  # noqa: E402
from hot3d_object_geometry import object_table  # noqa: E402
from muse_contact import a_masks, arm_rows, estimate_shifts, load_a  # noqa: E402
from number2_score import MUSE_ITEMS, N2, OBJECT_ITEMS, muse_items, standins  # noqa: E402
from tuning_muse_check import looo, lopo, per_person  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "experiments" / "img"
EPS = [0, 30, 53]
GAP_BINS = [0, 0.2, 0.5, 1, 2, 5, 100]
VERSIONS = {1: "checklist", 2: "checklist_v2"}


def number2():
    d = pd.read_csv(N2 / "clips_final.csv")
    d = d[d.role == "score_train"].reset_index(drop=True)
    st = standins(d)
    out, grades = {}, {}
    for v, task in VERSIONS.items():
        g = muse_items(d, task)
        ok = g[MUSE_ITEMS].notna().all(axis=1)
        g = g[ok].reset_index(drop=True)
        s = st[ok].reset_index(drop=True)
        g["R2"] = g[MUSE_ITEMS].mean(axis=1)
        g["R3"] = lopo(g, MUSE_ITEMS)
        g["R3_unseen"] = looo(g, MUSE_ITEMS)
        grades[v] = (g, s)
        out[f"version_{v}"] = {
            "clips": int(len(g)), "missing": int((~ok).sum()),
            "R2_equal_weights": per_person(g, "R2"), "R3_learned": per_person(g, "R3"),
            "R3_unseen_objects": per_person(g, "R3_unseen"),
            "muse_equals_standin": {k: round(float((g[k].values == s[k].values).mean()), 3) for k in OBJECT_ITEMS},
            "grade_tables": {k: pd.crosstab(s[k], g[k]).to_dict() for k in ("object_size", "object_width")}}
    g2, _ = grades[2]
    cm = pd.DataFrame([{"clip_id": r["clip_id"], **{k: r["parsed"][k] for k in ("longest_cm", "middle_cm", "shortest_cm")}}
                       for r in (json.loads(f.read_text()) for f in (N2 / "muse" / "checklist_v2").glob("*.json"))
                       if r["parsed"]])
    g2 = g2.merge(cm, on="clip_id")
    geo =object_table(sorted(g2.object.unique()))[["side_large_cm", "side_middle_cm", "side_small_cm"]].astype(float)
    true = geo.loc[g2.object]
    ratio = {name: g2[col].values / true[tcol].values for name, col, tcol in
             (("longest", "longest_cm", "side_large_cm"), ("middle", "middle_cm", "side_middle_cm"),
              ("shortest", "shortest_cm", "side_small_cm"))}
    out["version_2_cm_over_true"] = {k: {"median": round(float(np.median(r)), 2),
                                         "p25": round(float(np.percentile(r, 25)), 2),
                                         "p75": round(float(np.percentile(r, 75)), 2)} for k, r in ratio.items()}
    rows = [json.loads(f.read_text()) for f in (N2 / "muse" / "baseline").glob("*.json")]
    b = pd.DataFrame([{"clip_id": r["clip_id"], "p": r["parsed"]["copyable_probability"]} for r in rows if r["parsed"]])
    x = d.merge(b, on="clip_id")
    o = x.groupby("object").agg(p=("p", "mean"), copyable=("copyable", "mean"))
    out["muse_alone"] = {"share_at_0.85_or_more": round(float((x.p >= 0.85).mean()), 3),
                         "share_at_0.25_or_less": round(float((x.p <= 0.25).mean()), 3),
                         "object_level_rank_correlation": round(float(o.p.rank().corr(o.copyable.rank())), 3),
                         "auc": round(float(roc_auc_score(x.copyable, x.p)), 3)}
    plot_number2(grades, out)
    return out


def plot_number2(grades, out):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    lab = ["low", "mid", "high"]
    for j, (k, v) in enumerate(((k, v) for k in ("object_size", "object_width") for v in (1, 2))):
        g, s = grades[v]
        t = pd.crosstab(s[k], g[k]).reindex(index=[0.0, 0.5, 1.0], columns=[0.0, 0.5, 1.0], fill_value=0)
        ax = axes[j]
        ax.imshow(t.values, cmap="Blues")
        for a in range(3):
            for c in range(3):
                ax.text(c, a, int(t.values[a, c]), ha="center", va="center", fontsize=9,
                        color="white" if t.values[a, c] > t.values.max() / 2 else "black")
        ax.set_xticks(range(3), lab, fontsize=8)
        ax.set_yticks(range(3), lab, fontsize=8)
        ax.set_xlabel("Muse Spark grade", fontsize=8)
        ax.set_ylabel("3D model grade", fontsize=8)
        same = out[f"version_{v}"]["muse_equals_standin"][k]
        ax.set_title(f"{k.replace('object_', '')}, version {v}: {same:.0%} equal", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "exp30_grade_tables.png", dpi=120)
    plt.close(fig)


def gap_catch(ans, gt):
    rows = []
    for ep in EPS:
        n = len(gt[ep]["label"]["left"])
        raw = a_masks(ep, n, ans.get(ep, {}))
        for s in SIDES:
            segs = segments(gt[ep]["tactile_clean"][s])
            for (a, b), (c, _) in zip(segs[:-1], segs[1:]):
                rows.append({"gap_s": (c - b) / 30, "caught": bool((~raw[s][b:c]).any())})
    r = pd.DataFrame(rows)
    r["bin"] = pd.cut(r.gap_s, GAP_BINS)
    return {str(k): {"share": round(float(g.caught.mean()), 3), "gaps": int(len(g))}
            for k, g in r.groupby("bin", observed=True)}


def arm_a(version, gt):
    ans = load_a(EPS, version)
    rows = {"arm_A": {}, "arm_A_shift": {}}
    for ep in EPS:
        others = [e for e in EPS if e != ep]
        a_shift, _ = estimate_shifts(others, {e: ans.get(e, {}) for e in others}, None, None)
        r = arm_rows([ep], gt, ans, None, None, a_shift)
        for name in rows:
            rows[name].update(r[name])
    res = {}
    for name, r in rows.items():
        sc = Score()
        for (ep, s), segs in r.items():
            sc.add(segs, gt[ep]["tactile_clean"][s])
        x = sc.result()
        res[name] = {k: x[k] for k in ("boundary_f1_5", "boundary_f1_15", "contact_iou", "boundaries_pred",
                                       "boundaries_ref", "start_error_median", "end_error_median")}
    res["shift_on_all_three"] = estimate_shifts(EPS, ans, None, None)[0]
    res["releases_caught"] = gap_catch(ans, gt)
    return res


def number1():
    gt = ground_truth(EPS)
    out = {f"version_{v}": arm_a(v, gt) for v in (1, 2)}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = list(out["version_1"]["releases_caught"])
    x = np.arange(len(bins))
    for j, (v, c) in enumerate(((1, "#1971c2"), (2, "#e8590c"))):
        rc = out[f"version_{v}"]["releases_caught"]
        ax.bar(x + (j - 0.5) * 0.38, [rc[k]["share"] for k in bins], 0.38, color=c,
               label=f"version {v}: F1 ±5 with shift {out[f'version_{v}']['arm_A_shift']['boundary_f1_5']:.3f}")
    ax.set_xticks(x, [f"{b}\n{out['version_1']['releases_caught'][b]['gaps']} gaps" for b in bins], fontsize=7)
    ax.set_xlabel("length of the release between two glove contacts (s)")
    ax.set_ylabel("share Muse Spark marks as not touching")
    ax.set_ylim(0, 1.05)
    ax.set_title("Arm A on tuning episodes 0, 30, 53: does Muse Spark see short releases?", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "exp30_releases.png", dpi=120)
    plt.close(fig)
    return out


def main():
    out = {"number2": number2(), "number1": number1()}
    r3 = {v: out["number2"][f"version_{v}"]["R3_learned"]["per_person_mean"] for v in (1, 2)}
    f1 = {v: out["number1"][f"version_{v}"]["arm_A_shift"]["boundary_f1_5"] for v in (1, 2)}
    out["decision"] = {
        "checklist_R3_mean_over_people": r3, "checklist_for_held_out": "checklist_v2" if r3[2] > r3[1] else "checklist",
        "arm_A_f1_5_with_shift": f1, "arm_A_version_for_held_out": 2 if f1[2] > f1[1] else 1}
    (ROOT / "out" / "prompt_revision.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
