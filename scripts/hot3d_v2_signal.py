"""EXP-12 and EXP-13 again, on the copyable v2 label (EXP-16). Tuning clips only, no API call.

    python scripts/hot3d_v2_signal.py

Same models as before, leave one tuning person out, AUC per person and the
mean over the six people:
  - the three free checks (camera stability, retargeting recoverability,
    action completeness) alone and in a logistic regression,
  - object-only reference (copyable share of each object in the other people),
  - video hand shape (10 MediaPipe features during contact), alone and with object,
  - mocap hand shape (upper bound), alone and with object.
Also with the v2 label at depth limits 2 mm and 10 mm, to see whether the
picture depends on the depth limit.
Writes out/hot3d_checks/v2_signal.json and docs/experiments/img/exp17_v2_signal.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hot3d_copyable_v2 as v2  # noqa: E402
from hot3d_hand_shape import FEATS, lopo, within_object  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "out" / "hot3d_checks"
GRADES = ["grade_camera_stability", "grade_retargeting", "grade_completeness"]
KEY = ["sequence", "hand", "clip_start", "object"]


def object_only(t):
    s = pd.Series(np.nan, index=t.index)
    for p in sorted(t.person.unique()):
        tr, te = t[t.person != p], t[t.person == p]
        s[te.index] = te.object.map(tr.groupby("object").copyable.mean()).fillna(tr.copyable.mean())
    return {p: round(float(roc_auc_score(g.copyable, s[g.index])), 3) for p, g in t.groupby("person")}, s


def models(t):
    res = {}
    single = {g: round(float(roc_auc_score(t.copyable, t[g])), 3) for g in GRADES}
    for name, (cols, obj) in {
        "3 free checks": (GRADES, False),
        "object only": (None, None),
        "video hand shape": ([f"v_{f}" for f in FEATS], False),
        "video hand shape + object": ([f"v_{f}" for f in FEATS], True),
        "3 free checks + video hand shape + object": (GRADES + [f"v_{f}" for f in FEATS], True),
        "mocap hand shape (upper bound)": ([f"m_{f}" for f in FEATS], False),
        "mocap hand shape + object": ([f"m_{f}" for f in FEATS], True),
    }.items():
        per, s = object_only(t) if cols is None else lopo(t, cols, obj)
        res[name] = {"auc_mean_over_people": round(float(np.mean(list(per.values()))), 3),
                     "per_person": per, "within_object_auc": within_object(t, s) if cols else None}
    return single, res


def main():
    hs = pd.read_csv(CHECKS / "hand_shape.csv")
    lab = pd.read_csv(v2.OUT / "tune_v2_clips.csv")
    g = dict(np.load(v2.OUT / "tune_v2_geometry.npz"))
    for depth, col in ((0.005, "v2"), (0.002, "v2_depth2"), (0.01, "v2_depth10")):
        lab[col] = v2.label(g, depth=depth).astype(int)
    t = hs.drop(columns=["copyable"]).merge(lab[KEY + ["v2", "v2_depth2", "v2_depth10"]], on=KEY, how="inner")
    ok = t[[f"v_{f}" for f in FEATS] + [f"m_{f}" for f in FEATS]].notna().all(1)
    t = t[ok].reset_index(drop=True)
    out = {"clips": int(len(t)), "dropped_no_hand": int((~ok).sum())}
    for col in ("v2", "v2_depth2", "v2_depth10"):
        tt = t.copy()
        tt["copyable"] = tt[col]
        single, res = models(tt)
        out[col] = {"copyable_share": round(float(tt.copyable.mean()), 3), "single_free_check_auc": single,
                    "leave_one_person_out": res}
        print(col, "share", out[col]["copyable_share"], "single", single)
        for k, r in res.items():
            print(f"   {k}: {r['auc_mean_over_people']}  within object {r['within_object_auc']}  {r['per_person']}")
    (CHECKS / "v2_signal.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(out["v2"]["leave_one_person_out"])
    fig, ax = plt.subplots(figsize=(10.5, 4))
    y = np.arange(len(names))[::-1]
    for dy, col, c, lab in ((0.25, "v2_depth2", "#adb5bd", "depth 2 mm"), (0, "v2", "#1c7ed6", "depth 5 mm (fixed)"),
                            (-0.25, "v2_depth10", "#495057", "depth 10 mm")):
        vals = [out[col]["leave_one_person_out"][n]["auc_mean_over_people"] for n in names]
        ax.barh(y + dy, [v - 0.5 for v in vals], left=0.5, height=0.24, color=c, label=lab)
        if col == "v2":
            for yi, v in zip(y, vals):
                ax.text(max(v, 0.5) + 0.005, yi, f"{v:.3f}", va="center", fontsize=7)
    ax.axvline(0.5, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("AUC for copyable v2, tuning clips, leave one person out, mean of 6 people")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp17_v2_signal.png", dpi=125)


if __name__ == "__main__":
    main()
