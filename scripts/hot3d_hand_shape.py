"""Hand shape during contact as a number 2 check: does it predict copyable? (tuning split, no API call)

    python scripts/hot3d_hand_shape.py

The copyable label asks whether the Allegro hand fits the grasp, so the
check should look at the grasp itself. Per clip, during the contact, from
21 hand points in MediaPipe order:
  - opening: thumb tip to index tip and to middle tip, over hand size
    (wrist to middle finger base),
  - finger curl: for each finger, 1 - (base to tip) / (length along the joints),
  - spread: angle between the index and little finger directions,
  - closure: mean fingertip distance to the palm centre, over hand size,
  - pinch vs power: index curl minus the mean curl of ring and little finger.
Clip value = median over the contact frames (10 fps).

Two sources, same features:
  - video: MediaPipe world landmarks (the check we can use),
  - mocap: UmeTrack landmarks (an upper bound, how much hand shape could
    tell if the video were perfect; never an input to our score).
Scored on the 2,385 tuning clips, leave one person out, mean AUC over people.
Writes out/hot3d_checks/hand_shape.csv, hand_shape.json and docs/experiments/img/exp13_*.png.
"""

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
from hot3d_calib_check import ARIA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
IMG = ROOT / "docs" / "experiments" / "img"
STEP = 3
FINGERS = {"index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16], "little": [17, 18, 19, 20]}
FEATS = ["open_thumb_index", "open_thumb_middle", "curl_thumb", "curl_index", "curl_middle", "curl_ring",
         "curl_little", "spread_deg", "closure", "pinch_vs_power"]
UME_TO_MP = [5, 6, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]  # index 1 unused


def shape(P):
    """P: N x 21 x 3 in MediaPipe order. Returns N x len(FEATS)."""
    size = np.linalg.norm(P[:, 9] - P[:, 0], axis=1)
    out = {"open_thumb_index": np.linalg.norm(P[:, 4] - P[:, 8], axis=1) / size,
           "open_thumb_middle": np.linalg.norm(P[:, 4] - P[:, 12], axis=1) / size}

    def curl(ids):
        seg = sum(np.linalg.norm(P[:, ids[i + 1]] - P[:, ids[i]], axis=1) for i in range(len(ids) - 1))
        return 1 - np.linalg.norm(P[:, ids[-1]] - P[:, ids[0]], axis=1) / seg

    out["curl_thumb"] = curl([2, 3, 4])
    for f, ids in FINGERS.items():
        out[f"curl_{f}"] = curl(ids)
    a, b = P[:, 8] - P[:, 5], P[:, 20] - P[:, 17]
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    out["spread_deg"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    palm = P[:, [0, 5, 9, 13, 17]].mean(1)
    out["closure"] = np.linalg.norm(P[:, [8, 12, 16, 20]] - palm[:, None], axis=2).mean(1) / size
    out["pinch_vs_power"] = out["curl_index"] - (out["curl_ring"] + out["curl_little"]) / 2
    return np.stack([out[f] for f in FEATS], 1)


def video_features(seq, clips):
    z = np.load(OUT / "video" / f"{seq}.npz")
    fr, lm3d = z["frames"], z["lm3d"]
    rows = []
    for c in clips.itertuples():
        slot = 0 if c.hand == "left" else 1
        sel = (fr >= c.contact_start) & (fr < c.contact_start + c.contact_frames) & ~np.isnan(lm3d[:, slot, 0, 0])
        v = np.nanmedian(shape(lm3d[sel, slot].astype(float)), 0) if sel.sum() else np.full(len(FEATS), np.nan)
        rows.append(v)
    return np.array(rows)


def mocap_features(seq, clips):
    from hot3d_hands import HandModel
    model = HandModel(seq)
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    want = {}
    for c in clips.itertuples():
        for k in range(c.contact_start, c.contact_start + c.contact_frames, STEP):
            want[ts[k]] = k
    ang = {}
    with open(ARIA / seq / "hand_data" / "umetrack_hand_pose_trajectory.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["timestamp_ns"] in want:
                for h, hp in r["hand_poses"].items():
                    ang[(want[r["timestamp_ns"]], h)] = hp["joint_angles"]
    rows = []
    for c in clips.itertuples():
        h = "0" if c.hand == "left" else "1"
        ks = [k for k in range(c.contact_start, c.contact_start + c.contact_frames, STEP) if (k, h) in ang]
        if not ks:
            rows.append(np.full(len(FEATS), np.nan))
            continue
        lm = model.landmarks_batch(np.array([ang[(k, h)] for k in ks]))[:, UME_TO_MP]
        rows.append(np.median(shape(lm), 0))
    return np.array(rows)


def lopo(t, cols, with_object=False):
    """Leave one person out. Returns per-person AUC and the out-of-fold scores."""
    score = pd.Series(np.nan, index=t.index)
    for p in sorted(t.person.unique()):
        tr, te = t[t.person != p], t[t.person == p]
        Xtr, Xte = tr[cols].values, te[cols].values
        if with_object:
            share = tr.groupby("object").copyable.mean()
            Xtr = np.c_[Xtr, tr.object.map(share)]
            Xte = np.c_[Xte, te.object.map(share).fillna(tr.copyable.mean())]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(Xtr, tr.copyable)
        score[te.index] = m.predict_proba(Xte)[:, 1]
    per = {p: round(float(roc_auc_score(g.copyable, score[g.index])), 3) for p, g in t.groupby("person")}
    return per, score


def within_object(t, s):
    aucs, ws = [], []
    for _, g in t.groupby("object"):
        if g.copyable.nunique() == 2:
            aucs.append(roc_auc_score(g.copyable, s[g.index]))
            ws.append(len(g))
    return round(float(np.average(aucs, weights=ws)), 3)


def main():
    d = pd.read_csv(OUT / "clip_checks.csv")
    t = d[d.split == "tune"].copy().reset_index(drop=True)
    t["copyable"] = t.copyable.astype(int)
    vid, moc = [], []
    for seq, g in t.groupby("sequence", sort=False):
        vid.append(pd.DataFrame(video_features(seq, g), index=g.index, columns=[f"v_{f}" for f in FEATS]))
        moc.append(pd.DataFrame(mocap_features(seq, g), index=g.index, columns=[f"m_{f}" for f in FEATS]))
    t = t.join(pd.concat(vid)).join(pd.concat(moc))
    t.to_csv(OUT / "hand_shape.csv", index=False)
    ok = t[[f"v_{f}" for f in FEATS] + [f"m_{f}" for f in FEATS]].notna().all(1)
    t = t[ok].copy()
    share = t.copyable.mean()

    single = {}
    for f in FEATS:
        single[f] = {"video": round(float(roc_auc_score(t.copyable, t[f"v_{f}"])), 3),
                     "mocap": round(float(roc_auc_score(t.copyable, t[f"m_{f}"])), 3)}
    res, scores = {}, {}
    sets = {"object only": ([], True),
            "video hand shape": ([f"v_{f}" for f in FEATS], False),
            "video hand shape + object": ([f"v_{f}" for f in FEATS], True),
            "mocap hand shape (upper bound)": ([f"m_{f}" for f in FEATS], False),
            "mocap hand shape + object": ([f"m_{f}" for f in FEATS], True)}
    for name, (cols, obj) in sets.items():
        if not cols:
            s = pd.Series(np.nan, index=t.index)
            for p in sorted(t.person.unique()):
                tr, te = t[t.person != p], t[t.person == p]
                s[te.index] = te.object.map(tr.groupby("object").copyable.mean()).fillna(tr.copyable.mean())
            per = {p: round(float(roc_auc_score(g.copyable, s[g.index])), 3) for p, g in t.groupby("person")}
        else:
            per, s = lopo(t, cols, obj)
        scores[name] = s
        res[name] = {"auc_mean_over_people": round(float(np.mean(list(per.values()))), 3), "per_person": per,
                     "within_object_auc": within_object(t, s) if cols else None}
    out = {"clips": int(len(t)), "dropped_no_hand": int((~ok).sum()), "copyable_share": round(float(share), 3),
           "single_feature_auc": single, "leave_one_person_out": res}
    (OUT / "hand_shape.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    plot(t, res, single)


def plot(t, res, single):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    names = list(res)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        v = res[n]["auc_mean_over_people"]
        c = "#1c7ed6" if "video" in n else ("#868e96" if n == "object only" else "#2b8a3e")
        axes[0].barh(yi, v - 0.5, left=0.5, color=c, height=0.6)
        axes[0].text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    axes[0].axvline(0.5, color="black", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlim(0.45, 0.9)
    axes[0].set_xlabel("AUC for copyable, leave one person out, mean of 6 people")
    axes[0].set_title("Models on tuning clips")
    f = "open_thumb_middle"
    for src, c in (("v", "#1c7ed6"), ("m", "#2b8a3e")):
        for lab, ls in ((1, "-"), (0, "--")):
            x = t[t.copyable == lab][f"{src}_{f}"]
            axes[1].hist(x, bins=np.linspace(0, 2.2, 45), histtype="step", color=c, ls=ls, density=True,
                         label=f"{'video' if src == 'v' else 'mocap'}, {'copyable' if lab else 'not copyable'}")
    axes[1].set_xlabel("thumb to middle fingertip opening / hand size, during contact")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=7)
    axes[1].set_title(f"Hand opening (single AUC video {single[f]['video']:.2f}, mocap {single[f]['mocap']:.2f})")
    fig.tight_layout()
    fig.savefig(IMG / "exp13_hand_shape.png", dpi=125)


if __name__ == "__main__":
    main()
