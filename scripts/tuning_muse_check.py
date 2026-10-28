"""Tuning check with real Muse Spark answers (EXP-29): timing shift and checklist against learning.

    python scripts/tuning_muse_check.py

Number 1, arm A on tuning episodes 0, 30, 53 (P4 first part): the arm A shift
is learned on two episodes and applied to the third, for each episode in turn,
and A + shift is compared with A on the cleaned glove.
Number 2, the 400 score_train clips (P7a): R1 Muse alone, R2 the mean of the
seven Muse checklist grades, R3 a logistic regression on them, scored leave
one person out (R3 trained on the other five people), with paired sequence
bootstrap differences. Leave one object out for R3 against the object share
reference (0.5 for an unseen object). Tuning only, nothing on held-out.
Writes out/tuning_muse_check.json and docs/experiments/img/exp29_*.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_cut import SIDES, Score, ground_truth  # noqa: E402
from muse_contact import arm_rows, estimate_shifts, load_a, paired_bootstrap  # noqa: E402
from number2_score import MUSE_ITEMS, N2, OBJECT_ITEMS, muse_items, paired_auc, standins  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "experiments" / "img"
EPS = [0, 30, 53]


def number1():
    gt = ground_truth(EPS)
    ans = load_a(EPS)
    raw = Score()
    rows = {"arm_A": {}, "arm_A_shift": {}}
    shifts = {}
    for ep in EPS:
        others = [e for e in EPS if e != ep]
        a_shift, _ = estimate_shifts(others, {e: ans.get(e, {}) for e in others}, others, None, None)
        shifts[ep] = a_shift
        r = arm_rows([ep], gt, ans, None, None, a_shift)
        for name in rows:
            rows[name].update(r[name])
    for (ep, s), segs in rows["arm_A"].items():
        raw.add(segs, gt[ep]["tactile_clean"][s])
    all_shift, _ = estimate_shifts(EPS, ans, EPS, None, None)
    out = {"answered_frames": {ep: len(ans.get(ep, {})) for ep in EPS},
           "median_error_unshifted_frames": {"start": raw.result()["start_error_median"],
                                             "end": raw.result()["end_error_median"]},
           "shift_learned_on_other_two": {str(k): v for k, v in shifts.items()},
           "shift_learned_on_all_three_for_held_out": all_shift}
    for t in ("tactile_clean", "tactile", "label"):
        out[t] = {}
        for name, r in rows.items():
            sc = Score()
            for (ep, s), segs in r.items():
                sc.add(segs, gt[ep][t][s])
            res = sc.result()
            out[t][name] = {k: res[k] for k in ("boundary_f1_5", "boundary_f1_15", "contact_iou",
                                                "start_error_median", "end_error_median",
                                                "boundaries_pred", "boundaries_ref")}
    out["paired_bootstrap_3_episodes"] = paired_bootstrap(rows, gt, "arm_A", "arm_A_shift", EPS)
    plot_number1(gt, rows)
    return out


def plot_number1(gt, rows):
    fig, axes = plt.subplots(len(EPS), 1, figsize=(12, 2.3 * len(EPS)), squeeze=False)
    for ax, ep in zip(axes[:, 0], EPS):
        y, ticks = 0, []
        n = len(gt[ep]["tactile_clean"]["left"])
        t = np.arange(n) / 30
        for s in SIDES:
            for name, c, lab in (("glove", "#2b8a3e", f"{s}: glove (truth)"), ("arm_A", "#1971c2", f"{s}: Muse alone"),
                                 ("arm_A_shift", "#e8590c", f"{s}: Muse alone + shift")):
                if name == "glove":
                    m = gt[ep]["tactile_clean"][s]
                else:
                    m = np.zeros(n, bool)
                    for a, b in rows[name][(ep, s)]:
                        m[a:b] = True
                ax.fill_between(t, y, y + 0.8, where=m, color=c, step="mid", lw=0)
                ticks.append((y + 0.4, lab))
                y += 1
            y += 0.4
        ax.set_yticks([p for p, _ in ticks], [lab for _, lab in ticks], fontsize=7)
        ax.set_xlim(0, t[-1])
        ax.set_title(f"Episode {ep}, shift learned on the other two episodes", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(IMG / "exp29_arm_a_timelines.png", dpi=110)
    plt.close(fig)


def lopo(t, cols):
    p = np.zeros(len(t))
    for person in t.person.unique():
        tr, te = t.person != person, t.person == person
        m = LogisticRegression(C=0.5, max_iter=2000).fit(t.loc[tr, cols], t.loc[tr, "copyable"])
        p[te.values] = m.predict_proba(t.loc[te, cols])[:, 1]
    return p


def looo(t, cols):
    p = np.zeros(len(t))
    for o in t.object.unique():
        tr, te = t.object != o, t.object == o
        m = LogisticRegression(C=0.5, max_iter=2000).fit(t.loc[tr, cols], t.loc[tr, "copyable"])
        p[te.values] = m.predict_proba(t.loc[te, cols])[:, 1]
    return p


def per_person(t, col):
    per = {p: round(float(roc_auc_score(g.copyable, g[col])), 3) for p, g in t.groupby("person") if g.copyable.nunique() == 2}
    return {"pooled": round(float(roc_auc_score(t.copyable, t[col])), 3),
            "per_person_mean": round(float(np.mean(list(per.values()))), 3), "per_person": per}


def number2():
    d = pd.read_csv(N2 / "clips_final.csv")
    d = d[d.role == "score_train"].reset_index(drop=True)
    g = muse_items(d)
    rows = [json.loads(f.read_text()) for f in (N2 / "muse" / "baseline").glob("*.json")]
    b = pd.DataFrame([{"clip_id": r["clip_id"], "R1_muse_alone": r["parsed"]["copyable_probability"]}
                      for r in rows if r["parsed"]])
    g = g.merge(b, on="clip_id", how="left")
    st = standins(d).reset_index(drop=True)
    ok = g[MUSE_ITEMS].notna().all(axis=1) & g.R1_muse_alone.notna()
    g, st = g[ok].reset_index(drop=True), st[ok].reset_index(drop=True)
    g["R2_equal_weights"] = g[MUSE_ITEMS].mean(axis=1)
    g["R3_learned"] = lopo(g, MUSE_ITEMS)
    st["standin_R3"] = lopo(st, OBJECT_ITEMS)
    g["standin_R3"] = st["standin_R3"]
    share = {}
    for p in g.person.unique():  # object share reference, leave one person out
        tr = g[g.person != p]
        share.update(dict(zip(g[g.person == p].index, g[g.person == p].object.map(
            tr.groupby("object").copyable.mean()).fillna(tr.copyable.mean()))))
    g["object_reference"] = pd.Series(share)
    out = {"clips": int(len(g)), "dropped_no_answer": int((~ok).sum()), "copyable_share": round(float(g.copyable.mean()), 3),
           "leave_one_person_out": {c: per_person(g, c) for c in
                                    ("R1_muse_alone", "R2_equal_weights", "R3_learned", "standin_R3", "object_reference")},
           "paired": [paired_auc(g, "R2_equal_weights", "R3_learned"), paired_auc(g, "R1_muse_alone", "R2_equal_weights"),
                      paired_auc(g, "R1_muse_alone", "R3_learned"), paired_auc(g, "R3_learned", "object_reference")],
           "muse_equals_standin": {k: round(float((g[k].values == st[k].values).mean()), 3) for k in OBJECT_ITEMS},
           "single_item_auc": {k: round(float(roc_auc_score(g.copyable, g[k])), 3) for k in MUSE_ITEMS},
           "R3_weights_all": dict(zip(MUSE_ITEMS, np.round(LogisticRegression(C=0.5, max_iter=2000).fit(
               g[MUSE_ITEMS], g.copyable).coef_[0], 3).tolist()))}
    g["R3_unseen_object"] = looo(g, MUSE_ITEMS)
    out["unseen_objects"] = {"R3_leave_one_object_out": per_person(g, "R3_unseen_object"),
                             "R1_muse_alone_same_clips": per_person(g, "R1_muse_alone"),
                             "object_reference": 0.5,
                             "paired": paired_auc(g, "R1_muse_alone", "R3_unseen_object")}
    plot_number2(out)
    return out


def plot_number2(out):
    l = out["leave_one_person_out"]
    names = [("R1_muse_alone", "Muse alone"), ("R2_equal_weights", "checklist,\nequal weights"),
             ("R3_learned", "checklist,\nlearned weights"), ("standin_R3", "stand-in grades,\nlearned"),
             ("object_reference", "object name\nreference")]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(names))
    ax.bar(x - 0.18, [l[k]["pooled"] for k, _ in names], 0.36, color="#1971c2", label="pooled")
    ax.bar(x + 0.18, [l[k]["per_person_mean"] for k, _ in names], 0.36, color="#74c0fc", label="mean over people")
    u = out["unseen_objects"]["R3_leave_one_object_out"]["pooled"]
    ax.scatter([2], [u], marker="D", color="#e8590c", zorder=3, label=f"learned, unseen objects ({u:.3f})")
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_xticks(x, [n for _, n in names], fontsize=8)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUC, copyable v2")
    ax.set_title(f"Number 2 on the {out['clips']} score_train tuning clips, real Muse Spark answers, "
                 "leave one person out", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "exp29_number2_rows.png", dpi=120)
    plt.close(fig)


def main():
    out = {"number1": number1(), "number2": number2()}
    (ROOT / "out" / "tuning_muse_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
