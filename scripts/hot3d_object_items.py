"""Object items for the checklist, checked with stand-in grades (docs/MEASUREMENT_PLAN.md, EXP-19). No API call.

    python scripts/hot3d_object_items.py

Six items graded high (1), mid (0.5), low (0), high = easier for a large
four-finger robot hand. Anchors were fixed in the plan before computing.
Stand-ins for what Muse Spark would answer:
  - size, shape, thickness, width: from the 3D model oriented box,
  - handle: judged by eye from one frame per object (exp19_object_sheet.jpg),
  - free space: gap from the grasped object to the nearest other object in
    the mocap poses at the contact frame.
Logistic regression on the six grades, tuning clips, v2 label, leave one
person out and leave one object out.
Writes out/hot3d_checks/object_items.csv, object_items.json and docs/experiments/img/exp19_object_items.png.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, pose  # noqa: E402
from hot3d_copyable import Surface  # noqa: E402
from hot3d_object_geometry import boot_ci, object_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
LAB = ROOT / "out" / "hot3d_copyable" / "tune_v2_clips_labeled.csv"
HAND_CM = 19.0
ITEMS = ["item_size", "item_shape", "item_thickness", "item_width", "item_handle", "item_free_space"]
# judged by eye from exp19_object_sheet.jpg before looking at labels
HANDLE = {"coffee_pot": 1.0, "dumbbell_5lb": 1.0, "flask": 1.0, "mug_patterned": 1.0, "mug_white": 1.0,
          "potato_masher": 1.0, "spatula_red": 1.0, "spoon_wooden": 1.0, "vase": 1.0, "whiteboard_eraser": 0.5}


def grade(x, hi, lo, higher_is_better):
    """hi and lo are the two cut points in the plan's units."""
    if higher_is_better:
        return 1.0 if x >= hi else (0.5 if x >= lo else 0.0)
    return 1.0 if x <= hi else (0.5 if x <= lo else 0.0)


def free_space(args):
    """Gap (cm) from the grasped object to the nearest other object at each clip's contact frame."""
    seq, clips = args
    gt = ARIA / seq / "ground_truth"
    meta = json.loads((gt / "metadata.json").read_text())
    uid_of = dict(zip(meta["object_names"], meta["object_uids"]))
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    objs = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"])
    pts = {int(u): Surface(int(u), 3000).pts for u in meta["object_uids"]}
    out = []
    for c in clips.itertuples():
        t = ts[int(c.frame)]
        own = int(uid_of[c.object])
        world = {}
        for u, p in pts.items():
            if (t, u) in objs.index:
                o = objs.loc[(t, u)]
                T = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
                world[u] = (T[:3, :3] @ p.T).T + T[:3, 3]
        if own not in world or len(world) < 2:
            out.append(np.nan)
            continue
        tree = cKDTree(world[own])
        out.append(100 * min(float(tree.query(p)[0].min()) for u, p in world.items() if u != own))
    return clips.index.values, out


def main():
    t = pd.read_csv(LAB)
    geo = object_table(sorted(t.object.unique())).astype({"side_large_cm": float, "side_middle_cm": float,
                                                          "side_small_cm": float})
    with Pool(10) as pool:
        res = pool.map(free_space, [(s, g) for s, g in t.groupby("sequence")])
    gap = pd.Series(np.nan, index=t.index)
    for idx, vals in res:
        gap[idx] = vals
    t["free_space_cm"] = gap
    g = geo.loc[t.object]
    t["item_size"] = [grade(x / HAND_CM, 0.8, 1.3, False) for x in g.side_large_cm]
    t["item_shape"] = [grade(m / l, 0.6, 0.3, True) for m, l in zip(g.side_middle_cm, g.side_large_cm)]
    t["item_thickness"] = [grade(x, 4.0, 2.0, True) for x in g.side_small_cm]
    t["item_width"] = [grade(x, 8.0, 14.0, False) for x in g.side_middle_cm]
    t["item_handle"] = t.object.map(HANDLE).fillna(0.0)
    t["item_free_space"] = [grade(x, 10.0, 4.0, True) if np.isfinite(x) else 0.5 for x in t.free_space_cm]
    t.to_csv(OUT / "object_items.csv", index=False)

    single = {i: round(float(roc_auc_score(t.copyable, t[i])), 3) for i in ITEMS}
    share = {i: t[i].value_counts(normalize=True).round(3).sort_index().to_dict() for i in ITEMS}

    def fit_predict(tr, te):
        m = LogisticRegression(C=0.5, max_iter=2000).fit(tr[ITEMS], tr.copyable)
        return m.predict_proba(te[ITEMS])[:, 1]

    lopo = np.zeros(len(t))
    for p in sorted(t.person.unique()):
        tr, te = t.person != p, t.person == p
        lopo[te.values] = fit_predict(t[tr], t[te])
    per = {p: round(float(roc_auc_score(gp.copyable, lopo[gp.index])), 3) for p, gp in t.groupby("person")}
    looo = np.zeros(len(t))
    for o in sorted(t.object.unique()):
        tr, te = t.object != o, t.object == o
        looo[te.values] = fit_predict(t[tr], t[te])
    lopo_mean = float(np.mean(list(per.values())))
    looo_auc = float(roc_auc_score(t.copyable, looo))
    verdict = "pass" if lopo_mean >= 0.65 and looo_auc >= 0.65 else (
        "fail" if lopo_mean < 0.60 or looo_auc < 0.60 else "weak")
    full = LogisticRegression(C=0.5, max_iter=2000).fit(t[ITEMS], t.copyable)
    out = {"clips": len(t), "copyable_share": round(float(t.copyable.mean()), 3),
           "free_space_cm_median": round(float(t.free_space_cm.median()), 1),
           "single_item_auc": single, "grade_shares": share,
           "leave_one_person_out_mean": round(lopo_mean, 3), "leave_one_person_out_per_person": per,
           "leave_one_object_out_pooled": round(looo_auc, 3), "leave_one_object_out_ci95": boot_ci(t, looo),
           "exact_geometry_for_reference": {"leave_one_person_out": 0.695, "leave_one_object_out": 0.673},
           "weights_all_tuning": dict(zip(ITEMS, np.round(full.coef_[0], 3).tolist())), "verdict": verdict}
    (OUT / "object_items.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    plot(t, out)


def plot(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    names = [i.replace("item_", "").replace("_", " ") for i in ITEMS]
    rates = [[t[t[i] == v].copyable.mean() * 100 if (t[i] == v).any() else np.nan for v in (0.0, 0.5, 1.0)] for i in ITEMS]
    x = np.arange(len(ITEMS))
    for k, (lab, c) in enumerate((("low", "#c92a2a"), ("mid", "#fab005"), ("high", "#2b8a3e"))):
        axes[0].bar(x + (k - 1) * 0.27, [r[k] for r in rates], 0.27, color=c, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("copyable v2 (%)")
    axes[0].legend(fontsize=7, title="grade", title_fontsize=7)
    axes[0].set_title("Copyable share by stand-in grade (tuning clips)")
    rows = [("exact geometry, leave person out", 0.695), ("exact geometry, leave object out", 0.673),
            ("six items, leave person out", out["leave_one_person_out_mean"]),
            ("six items, leave object out", out["leave_one_object_out_pooled"])]
    y = np.arange(len(rows))[::-1]
    for yi, (n, v) in zip(y, rows):
        axes[1].barh(yi, v - 0.5, left=0.5, color="#868e96" if "exact" in n else "#1c7ed6", height=0.6)
        axes[1].text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    axes[1].axvline(0.65, color="#2b8a3e", ls="--", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([r[0] for r in rows], fontsize=8)
    axes[1].set_xlim(0.45, 0.8)
    axes[1].set_xlabel("AUC for copyable v2 (dashed: pass line 0.65)")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp19_object_items.png", dpi=125)


if __name__ == "__main__":
    main()
