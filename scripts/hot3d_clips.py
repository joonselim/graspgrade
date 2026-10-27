"""Cut the number 2 clips in HOT3D from the mocap hand to object distance (no API call).

    python scripts/hot3d_clips.py

The plan (risk 2) says number 2 clips must not come from our own contact
pipeline, so that cutting errors do not leak into the score check. Here a
grasp is when at least two fingertips of one hand are within 15 mm of an
object's surface (UmeTrack fingertips, EXP-09). Per hand and object:
  - contact frames are joined across gaps up to 0.5 s and kept if at least 0.3 s,
  - the clip runs from 1 s before the contact starts (reach) to 1 s after it
    ends (release),
  - the contact frame for the copyable label is the middle of the contact,
  - the object counts as moved when it travels at least 3 cm during the contact.
All 30 fps RGB frames are used (stream 214-1). Writes out/hot3d_clips/clips.csv.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ASSETS  # noqa: E402
from hot3d_hands import FLIPS, HandModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_clips"
NEAR = 0.015
MIN_TIPS = 2
MERGE_GAP, MIN_LEN, PAD = 15, 9, 30
MOVED = 0.03
REACH = 0.35   # m, only test objects whose centre is this close to the wrist
FLIP = {"0": "none", "1": "x"}
SIDE = {"0": "left", "1": "right"}


def segments(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def hand_tracks(seq, ts):
    """Per hand: fingertips (N x 5 x 3, world m), wrist (N x 3), valid (N)."""
    model = HandModel(seq)
    idx = {t: i for i, t in enumerate(ts)}
    n = len(ts)
    raw = {h: {"ang": np.zeros((n, 22)), "t": np.zeros((n, 3)), "q": np.tile([1.0, 0, 0, 0], (n, 1)),
               "ok": np.zeros(n, bool)} for h in SIDE}
    with open(ARIA / seq / "hand_data" / "umetrack_hand_pose_trajectory.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            i = idx.get(r["timestamp_ns"])
            if i is None:
                continue
            for h, hp in r["hand_poses"].items():
                if h in raw:
                    raw[h]["ang"][i] = hp["joint_angles"]
                    raw[h]["t"][i] = hp["wrist_xform"]["t_xyz"]
                    raw[h]["q"][i] = hp["wrist_xform"]["q_wxyz"]
                    raw[h]["ok"][i] = True
    out = {}
    for h, r in raw.items():
        lm = model.landmarks_batch(r["ang"]) * FLIPS[FLIP[h]] / 1000.0
        R = Rotation.from_quat(r["q"][:, [1, 2, 3, 0]]).as_matrix()
        world = np.einsum("nij,nkj->nki", R, lm) + r["t"][:, None, :]
        out[h] = {"tips": world[:, :5], "wrist": world[:, 5], "ok": r["ok"]}
    return out


def object_tracks(seq, ts):
    gt = ARIA / seq / "ground_truth"
    d = pd.read_csv(gt / "dynamic_objects.csv")
    d = d[d["timestamp[ns]"].isin(set(ts))]
    out = {}
    for uid, g in d.groupby("object_uid"):
        g = g.set_index("timestamp[ns]").reindex(ts)
        q = g[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values
        t = g[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values
        ok = ~np.isnan(q).any(1) & ~np.isnan(t).any(1)
        R = np.tile(np.eye(3), (len(ts), 1, 1))
        R[ok] = Rotation.from_quat(q[ok][:, [1, 2, 3, 0]]).as_matrix()
        out[int(uid)] = {"R": R, "t": np.nan_to_num(t), "ok": ok}
    return out


def cut(seq):
    gt = ARIA / seq / "ground_truth"
    meta = json.loads((gt / "metadata.json").read_text())
    name_of = {int(u): n for u, n in zip(meta["object_uids"], meta["object_names"])}
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    rgb = qa[qa["stream_id"] == "214-1"].sort_values("timestamp[ns]")
    ts, qa_ok = rgb["timestamp[ns]"].values, rgb["mask"].astype(bool).values
    n = len(ts)
    hands, objs = hand_tracks(seq, ts), object_tracks(seq, ts)
    contact = {}
    for uid, o in objs.items():
        m = trimesh.load(ASSETS / f"{uid}.glb", force="mesh")
        tree = cKDTree(np.asarray(trimesh.sample.sample_surface(m, 20000, seed=0)[0]))
        for h, hd in hands.items():
            near = hd["ok"] & o["ok"] & (np.linalg.norm(hd["wrist"] - o["t"], axis=1) < REACH)
            cnt = np.zeros(n, int)
            dmin = np.full(n, np.inf)
            if near.any():
                # world -> object frame: R^T (p - t)
                p = np.einsum("nji,nkj->nki", o["R"][near], hd["tips"][near] - o["t"][near][:, None, :])
                dist, _ = tree.query(p.reshape(-1, 3))
                dist = dist.reshape(-1, 5)
                cnt[near] = (dist < NEAR).sum(1)
                dmin[near] = dist.min(1)
            contact[(uid, h)] = (cnt >= MIN_TIPS, dmin)
    rows = []
    for (uid, h), (c, dmin) in contact.items():
        segs = []
        for a, b in segments(c):
            if segs and a - segs[-1][1] <= MERGE_GAP:
                segs[-1] = (segs[-1][0], b)
            else:
                segs.append((a, b))
        other = contact.get((uid, "1" if h == "0" else "0"), (np.zeros(n, bool), None))[0]
        for a, b in segs:
            if b - a < MIN_LEN:
                continue
            o = objs[uid]
            pos = o["t"][a:b][o["ok"][a:b]]
            travel = float(np.linalg.norm(pos - pos[0], axis=1).max()) if len(pos) else 0.0
            s, e = max(0, a - PAD), min(n, b + PAD)
            rows.append({"sequence": seq, "object": name_of.get(uid, str(uid)), "object_uid": uid,
                         "hand": SIDE[h], "contact_start": int(a), "contact_frames": int(b - a),
                         "contact_frame": int((a + b) // 2), "clip_start": int(s), "clip_frames": int(e - s),
                         "object_travel_m": round(travel, 3), "moved": travel >= MOVED,
                         "other_hand_touches_pct": round(100 * float(other[a:b].mean()), 1),
                         "clip_qa_pass_pct": round(100 * float(qa_ok[s:e].mean()), 1)})
    return rows


def plot(seq, obj_name):
    """Fingertip distance for one object, and every clip in the sequence coloured by its copyable label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = ARIA / seq / "ground_truth"
    meta = json.loads((gt / "metadata.json").read_text())
    uid = int(dict(zip(meta["object_names"], meta["object_uids"]))[obj_name])
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    hands, objs = hand_tracks(seq, ts), object_tracks(seq, ts)
    m = trimesh.load(ASSETS / f"{uid}.glb", force="mesh")
    tree = cKDTree(np.asarray(trimesh.sample.sample_surface(m, 20000, seed=0)[0]))
    o = objs[uid]
    lab = pd.read_csv(OUT / "clips_labeled.csv")
    lab = lab[lab.sequence == seq]
    t = np.arange(len(ts)) / 30
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2), gridspec_kw={"height_ratios": [1.2, 1]})
    for h, c in (("0", "#1c7ed6"), ("1", "#e8590c")):
        hd = hands[h]
        p = np.einsum("nji,nkj->nki", o["R"], hd["tips"] - o["t"][:, None, :])
        d = tree.query(p.reshape(-1, 3))[0].reshape(-1, 5)
        d2 = np.sort(d, 1)[:, 1]
        d2[~(hd["ok"] & o["ok"])] = np.nan
        axes[0].plot(t, 100 * d2, color=c, lw=0.8, label=f"{SIDE[h]} hand, 2nd closest fingertip")
    axes[0].axhline(100 * NEAR, color="black", ls="--", lw=1, label="15 mm")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.1, 100)
    axes[0].set_ylabel("distance to object (cm)")
    axes[0].set_title(f"{seq}: {obj_name}. Contact = two fingertips within 15 mm")
    axes[0].legend(fontsize=7, loc="upper right")
    names = sorted(lab.object.unique())
    for _, r in lab.iterrows():
        y = names.index(r.object) + (0.4 if r.hand == "right" else 0)
        axes[1].barh(y, r.clip_frames / 30, left=r.clip_start / 30, height=0.35,
                     color="#2b8a3e" if r.copyable else "#c92a2a")
    axes[1].set_yticks(np.arange(len(names)) + 0.2)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("seconds   (green: copyable, red: not copyable, lower bar left hand, upper bar right hand)")
    for ax in axes:
        ax.set_xlim(0, t[-1])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp11_clips_timeline.png", dpi=120)


def main():
    if sys.argv[1:2] == ["--plot"]:
        plot(sys.argv[2], sys.argv[3])
        return
    split = json.loads((ROOT / "splits" / "hot3d_aria.json").read_text())
    part = {s: p for p in ("tune", "held_out") for s in split[p]}
    with Pool(12) as pool:
        rows = [r for rs in pool.imap_unordered(cut, sorted(part)) for r in rs]
    df = pd.DataFrame(rows)
    df["split"] = df.sequence.map(part)
    df = df.sort_values(["sequence", "clip_start"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "clips.csv", index=False)
    use = df[df.moved & (df.clip_qa_pass_pct >= 90)]
    summary = {}
    for p in ("tune", "held_out"):
        g, u = df[df.split == p], use[use.split == p]
        summary[p] = {"contacts": len(g), "moved_object": int(g.moved.sum()),
                      "moved_and_qa_90pct": len(u),
                      "contact_seconds_median": round(float(u.contact_frames.median()) / 30, 2),
                      "clip_seconds_median": round(float(u.clip_frames.median()) / 30, 2),
                      "clip_seconds_p90": round(float(u.clip_frames.quantile(0.9)) / 30, 2),
                      "both_hands_on_object_pct": round(100 * float((u.other_hand_touches_pct > 50).mean()), 1)}
    (OUT / "clips_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
