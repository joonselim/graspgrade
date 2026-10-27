"""Compare copyable label methods on tuning clips, by the criteria fixed in docs/MEASUREMENT_PLAN.md.

    python scripts/hot3d_label_compare.py

Methods (hot3d_copyable.py run --source clips --method ...):
  ratio    vector retargeting scaled by Allegro length / person length (EXP-10)
  scale1   vector retargeting, scale 1.0
  position position retargeting of fingertips and finger bases, free hand base
Criteria, written before the results:
  fidelity  on solid human grasps, median distance Allegro fingertip to human fingertip
  hand size rank correlation of hand length and label inside each object, near 0
  share     copyable share inside 20% to 80%
Rule: among methods with |hand size correlation| <= 0.1 and share in the band, best fidelity.
Writes out/hot3d_copyable/label_compare.json and docs/experiments/img/exp15_*.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hot3d_copyable as hc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_copyable"
IMG = ROOT / "docs" / "experiments" / "img"
METHODS = {"ratio": "vector, scale = length ratio (EXP-10)", "scale1": "vector, scale 1.0",
           "position": "position, free base"}
NEAR, DEEP, RADIUS = 0.015, 0.01, 0.02
CANS = ("can_", "carton_", "bottle_")
ROUND = ("mug_white", "mug_patterned", "bowl", "plate_bamboo", "flask", "vase")


def load(method):
    tag = "tune_clips" if method == "ratio" else f"tune_clips_{method}"
    df = pd.read_csv(OUT / f"{tag}_events.csv")
    sd = np.load(OUT / f"{tag}_sd.npy")
    df["copyable"] = hc.labels(sd, NEAR, DEEP, RADIUS).astype(int)
    z = int(np.argmin(hc.GRID_R))
    others = sd[:, z, 1:4]
    df["other_tip_signed_mm"] = 1000 * others[np.arange(len(sd)), np.abs(others).argmin(1)]
    df["thumb_signed_mm"] = 1000 * sd[:, z, 0]
    return df, sd


def within_object_corr(df):
    rs, ws = [], []
    for _, g in df.groupby("object"):
        if g.copyable.nunique() == 2 and g.human_hand_len_mm.nunique() > 1:
            rs.append(spearmanr(g.human_hand_len_mm, g.copyable)[0])
            ws.append(len(g))
    return float(np.average(rs, weights=ws))


def summarize(method, df):
    solid = (df.human_tips_within_15mm >= 2) & (df.human_thumb_sd_mm.abs() <= 15)
    person = df.groupby("person").agg(share=("copyable", "mean"), hand=("human_hand_len_mm", "first"))
    cans = df.object.str.startswith(CANS)
    rnd = df.object.isin(ROUND)
    obj = df.groupby("object").copyable.agg(["mean", "size"])
    obj = obj[obj["size"] >= 25].sort_values("mean")
    return {
        "method": METHODS[method], "clips": len(df),
        "copyable_share": round(float(df.copyable.mean()), 3),
        "fidelity_tip_to_human_tip_mm_on_solid_grasps": round(float(df.tip_to_human_tip_mm_median[solid].median()), 1),
        "fidelity_thumb_mm_on_solid_grasps": round(float(df.thumb_to_human_thumb_mm[solid].median()), 1),
        "hand_size_corr_within_object": round(within_object_corr(df), 3),
        "hand_size_corr_people_spearman": round(float(spearmanr(person.hand, person.share)[0]), 3),
        "share_by_person": person.share.round(3).to_dict(),
        "hand_len_mm_by_person": person.hand.round(1).to_dict(),
        "copyable_share_on_solid_human_grasps": round(float(df.copyable[solid].mean()), 3),
        "wrist_to_human_wrist_mm_median": round(float(df.wrist_to_human_wrist_mm.median()), 1),
        "retarget_error_median": round(float(df.retarget_vector_error.median()), 3),
        "cans_cartons_bottles": {"clips": int(cans.sum()), "copyable": round(float(df.copyable[cans].mean()), 3),
                                 "other_tip_signed_mm_median": round(float(df.other_tip_signed_mm[cans].median()), 1)},
        "mugs_bowl_plate_flask_vase": {"clips": int(rnd.sum()), "copyable": round(float(df.copyable[rnd].mean()), 3),
                                       "other_tip_signed_mm_median": round(float(df.other_tip_signed_mm[rnd].median()), 1)},
        "lowest_objects": obj.head(5)["mean"].round(2).to_dict(),
        "highest_objects": obj.tail(5)["mean"].round(2).to_dict(),
    }


def choose(res):
    ok = {m: r for m, r in res.items()
          if abs(r["hand_size_corr_within_object"]) <= 0.1 and 0.2 <= r["copyable_share"] <= 0.8}
    if not ok:
        return None, "no method passes the hand size and share checks"
    best = min(ok, key=lambda m: ok[m]["fidelity_tip_to_human_tip_mm_on_solid_grasps"])
    return best, f"passing: {sorted(ok)}; best fidelity: {best}"


def plot_summary(res, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = list(res)
    x = np.arange(len(names))
    groups = [("all clips", lambda r: r["copyable_share"]),
              ("cans, cartons, bottles", lambda r: r["cans_cartons_bottles"]["copyable"]),
              ("mugs, bowl, plate, flask, vase", lambda r: r["mugs_bowl_plate_flask_vase"]["copyable"])]
    for i, (lab, f) in enumerate(groups):
        axes[0].bar(x + (i - 1) * 0.27, [100 * f(res[m]) for m in names], 0.27, label=lab)
    axes[0].axhspan(20, 80, color="#fff3bf", zorder=0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("copyable (%)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Copyable share by object group")
    axes[1].bar(x, [res[m]["fidelity_tip_to_human_tip_mm_on_solid_grasps"] for m in names], color="#1c7ed6")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("mm")
    axes[1].set_title("Allegro fingertip to human fingertip\n(solid human grasps, median, lower is better)")
    for m, c in zip(names, ("#868e96", "#e8590c", "#2b8a3e")):
        r = res[m]
        p = sorted(r["share_by_person"])
        axes[2].plot([r["hand_len_mm_by_person"][q] for q in p], [100 * r["share_by_person"][q] for q in p],
                     "o", color=c, label=f"{m} (within object r = {r['hand_size_corr_within_object']:+.2f})")
    axes[2].set_xlabel("person hand length, wrist to index tip (mm)")
    axes[2].set_ylabel("copyable (%)")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Copyable share per tuning person")
    fig.tight_layout()
    fig.savefig(IMG / "exp15_label_methods.png", dpi=125)


def plot_cans(data, n=4):
    """Can, carton and bottle grasps drawn for each method."""
    import cv2
    from hot3d_calib_check import ARIA, ORIENT, pose
    from hot3d_hands import HandModel, camera, load_poses, to_pixels
    base = data["ratio"][0]
    pick = base[base.object.str.startswith(CANS)].sample(n, random_state=20261026)
    rows = []
    for m in METHODS:
        df, sd = data[m]
        cells = []
        for _, e in pick.iterrows():
            i = int(np.where((df.sequence == e.sequence) & (df.clip_start == e.clip_start) & (df.hand == e.hand))[0][0])
            seq, k, side = e.sequence, int(e.frame), e.hand
            gt = ARIA / seq / "ground_truth"
            meta = json.loads((gt / "metadata.json").read_text())
            uid = int(dict(zip(meta["object_names"], meta["object_uids"]))[e.object])
            qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
            ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
            hid = "0" if side == "left" else "1"
            model = HandModel(seq)
            kp = hc.mediapipe_points(model, load_poses(seq, [ts[k]])[ts[k]][hid], hc.FLIP[hid])
            robot = hc.Allegro(side, float(df.scale.iloc[i]) if m == "ratio" else 1.0,
                               kind="position" if m == "position" else "vector")
            pts, _ = robot.fit(kp)
            ok = hc.GRID_R <= RADIUS + 1e-9
            tip = (sd[i, :, :4] <= NEAR) & (sd[i, :, :4] >= -DEEP)
            score = np.where(ok, tip.sum(1) + 10 * (tip[:, 0] & tip[:, 1:].any(1) & (sd[i, :, 4] >= -DEEP)), -1)
            pts = pts + hc.GRID[int(np.argmax(score))]
            params, T_dc, head = camera(seq)
            o = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"]).loc[(ts[k], uid)]
            T_wo = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
            sp = hc.Surface(uid, 4000).pts
            sp_w = (T_wo[:3, :3] @ sp.T).T + T_wo[:3, 3]
            cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            _, img = cap.read()
            cap.release()
            for p, c, r in ((sp_w, (0, 200, 0), 2), (kp, (255, 255, 255), 6), (pts, (0, 140, 255), 7),
                            (pts[[robot.thumb] + robot.others], (0, 0, 255), 13)):
                uv, z = to_pixels(p, ts[k], params, T_dc, head)
                u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
                for x, y, zz in zip(u, v, z):
                    if zz > 0 and 0 <= x < 1408 and 0 <= y < 1408:
                        cv2.circle(img, (int(x), int(y)), r, c, -1)
            uv, _ = to_pixels(kp, ts[k], params, T_dc, head)
            u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
            half = 300
            x0 = int(np.clip(np.median(u) - half, 0, 1408 - 2 * half))
            y0 = int(np.clip(np.median(v) - half, 0, 1408 - 2 * half))
            crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
            lab = f"{m}: {'copyable' if df.copyable.iloc[i] else 'not copyable'}"
            for col, w in (((0, 0, 0), 6), ((255, 255, 255), 2)):
                cv2.putText(crop, lab, (12, 40), 0, 1.1, col, w)
                cv2.putText(crop, e.object, (12, 580), 0, 1.0, col, w)
            cells.append(cv2.resize(crop, (360, 360)))
        rows.append(np.hstack(cells))
    cv2.imwrite(str(IMG / "exp15_can_grasps.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    data = {m: load(m) for m in METHODS}
    res = {m: summarize(m, data[m][0]) for m in METHODS}
    best, why = choose(res)
    out = {"criteria": "fixed before results, see docs/MEASUREMENT_PLAN.md", "methods": res,
           "chosen": best, "why": why}
    (OUT / "label_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    plot_summary(res, data)
    plot_cans(data)


if __name__ == "__main__":
    main()
