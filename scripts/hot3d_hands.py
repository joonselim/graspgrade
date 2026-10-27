"""HOT3D hand landmarks (21 points, fingertips included) from UmeTrack poses, no MANO.

    python scripts/hot3d_hands.py check P0001_10a27bf7 P0002_016222d1 P0011_72efb935

hand_data/umetrack_hand_user_profile.json holds a full kinematic hand model
(22 joints with rest positions in mm, rotation axes, parents, and 21 landmarks
bound to bones). umetrack_hand_pose_trajectory.jsonl holds, per timestamp and
hand, 22 joint angles and the wrist pose in the world frame (m). Forward
kinematics: joint j turns by its angle about its axis through its rest
position, on top of its parent. A landmark moves with its bone.

Landmark order: 0-4 fingertips (thumb, index, middle, ring, little), 5 wrist,
6-7 thumb, 8-19 the other fingers (proximal, intermediate, distal), 20 palm.

The profile is one hand. The other hand is its mirror image. `check` finds
which hand index needs mirroring and along which axis, by projecting the
landmarks (EXP-08 lens model) into the raw image and counting how many fall
inside the dataset hand box (box2d_hands.csv). It also measures fingertip to
object surface distance during the grasp events of EXP-07.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ASSETS, IMG, ORIENT, fisheye624, pose  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_hands"
TIPS = [0, 1, 2, 3, 4]
FLIPS = {"none": np.array([1, 1, 1]), "x": np.array([-1, 1, 1]),
         "y": np.array([1, -1, 1]), "z": np.array([1, 1, -1])}
BONES = [(5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (5, 11), (11, 12), (12, 13), (13, 2),
         (5, 14), (14, 15), (15, 16), (16, 3), (5, 17), (17, 18), (18, 19), (19, 4)]


class HandModel:
    def __init__(self, seq):
        h = json.loads((ARIA / seq / "hand_data" / "umetrack_hand_user_profile.json").read_text())["hand_model"]
        self.parent = np.array(h["joint_parent"], int)
        self.rest = np.array(h["joint_rest_positions"], float)
        self.axes = np.array(h["joint_rotation_axes"], float)
        self.frame_of = np.array(h["joint_frame_index"], int)
        self.lm_rest = np.array(h["landmark_rest_positions"], float)
        self.lm_bone = np.array(h["landmark_rest_bone_indices"], int)
        self.lm_w = np.array(h["landmark_rest_bone_weights"], float)
        self.scale = float(h.get("hand_scale", 1.0))
        self.n_frames = np.array(h["dense_bone_weights"]).shape[1]

    def frames(self, angles):
        """4x4 transform per bone frame, in the hand frame (mm). Frames no joint drives stay identity."""
        g = [None] * len(self.parent)
        out = [np.eye(4) for _ in range(self.n_frames)]
        for j in range(len(self.parent)):
            local = np.eye(4)
            local[:3, :3] = Rotation.from_rotvec(self.axes[j] * angles[j]).as_matrix()
            local[:3, 3] = self.rest[j] - local[:3, :3] @ self.rest[j]
            g[j] = local if self.parent[j] == 255 else g[self.parent[j]] @ local
            if self.frame_of[j] != 255:
                out[self.frame_of[j]] = g[j]
        return out

    def landmarks(self, angles):
        fr = self.frames(angles)
        pts = np.zeros((len(self.lm_rest), 3))
        for i, p in enumerate(self.lm_rest):
            for b, w in zip(self.lm_bone[i], self.lm_w[i]):
                if w:
                    pts[i] += w * (fr[b][:3, :3] @ p + fr[b][:3, 3])
        return pts * self.scale

    def world(self, hand_pose, flip):
        """21 landmarks in the world frame (m)."""
        p = self.landmarks(np.array(hand_pose["joint_angles"])) * FLIPS[flip] / 1000.0
        T = pose(hand_pose["wrist_xform"]["q_wxyz"], hand_pose["wrist_xform"]["t_xyz"])
        return (T[:3, :3] @ p.T).T + T[:3, 3]


def load_poses(seq, ts=None):
    keep = set(ts) if ts is not None else None
    poses = {}
    with open(ARIA / seq / "hand_data" / "umetrack_hand_pose_trajectory.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if keep is None or r["timestamp_ns"] in keep:
                poses[r["timestamp_ns"]] = r["hand_poses"]
    return poses


def camera(seq):
    gt = ARIA / seq / "ground_truth"
    cam = next(c for c in json.loads((gt / "camera_models.json").read_text()) if c["label"] == "camera-rgb")
    T_dc = pose(cam["T_Device_Camera"]["quaternion_wxyz"], cam["T_Device_Camera"]["translation_xyz"])
    head = pd.read_csv(gt / "headset_trajectory.csv").set_index("timestamp[ns]")
    return np.array(cam["projectionParams"]), T_dc, head


def to_pixels(pw, t, params, T_dc, head):
    h = head.loc[t]
    T_wd = pose(h[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, h[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
    pc = (np.linalg.inv(T_wd @ T_dc) @ np.c_[pw, np.ones(len(pw))].T).T[:, :3]
    return fisheye624(pc, params), pc[:, 2]


def box_check(seq, model, n=120):
    """Share of projected landmarks inside the dataset hand box, per hand index and mirror axis."""
    gt = ARIA / seq / "ground_truth"
    hb = pd.read_csv(gt / "box2d_hands.csv")
    hb = hb[(hb["stream_id"] == "214-1") & (hb["visibility_ratio[%]"] > 0.9)].dropna()
    params, T_dc, head = camera(seq)
    ts = sorted(set(hb["timestamp[ns]"]) & set(head.index))
    ts = [ts[i] for i in np.linspace(0, len(ts) - 1, n).astype(int)]
    poses = load_poses(seq, ts)
    res = {}
    for _, b in hb[hb["timestamp[ns]"].isin(ts)].iterrows():
        t, hid = b["timestamp[ns]"], str(int(b["hand_index"]))
        if hid not in poses.get(t, {}):
            continue
        x0, x1, y0, y1 = b["x_min[pixel]"], b["x_max[pixel]"], b["y_min[pixel]"], b["y_max[pixel]"]
        pad = 0.05 * max(x1 - x0, y1 - y0)
        for flip in FLIPS:
            uv, z = to_pixels(model.world(poses[t][hid], flip), t, params, T_dc, head)
            inside = (z > 0) & (uv[:, 0] >= x0 - pad) & (uv[:, 0] <= x1 + pad) & (uv[:, 1] >= y0 - pad) & (uv[:, 1] <= y1 + pad)
            res.setdefault((hid, flip), []).append(float(inside.mean()))
    return {f"hand {h}, mirror {f}": round(float(np.mean(v)), 3) for (h, f), v in sorted(res.items())}


def surface_points(uid, n=20000):
    m = trimesh.load(ASSETS / f"{uid}.glb", force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return np.asarray(pts)


def tip_distances(seq, model, flips, events):
    """Closest fingertip to object surface (m): at mid-move and 1.5 s before the move starts."""
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    meta = json.loads((gt / "metadata.json").read_text())
    uid_of = dict(zip(meta["object_names"], meta["object_uids"]))
    objs = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"])
    poses = load_poses(seq, ts)
    trees, rows = {}, []
    for _, e in events.iterrows():
        uid = int(uid_of[e["object"]])
        if uid not in trees:
            trees[uid] = cKDTree(surface_points(uid))
        for when, k in (("during", e["move_start"] + e["move_frames"] // 2), ("before", e["move_start"] - 45)):
            if k < 0 or k >= len(ts) or (ts[k], uid) not in objs.index:
                continue
            o = objs.loc[(ts[k], uid)]
            T_wo = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
            T_ow = np.linalg.inv(T_wo)
            best = None
            for hid, hp in poses.get(ts[k], {}).items():
                tips = model.world(hp, flips[hid])[TIPS]
                d, _ = trees[uid].query((T_ow[:3, :3] @ tips.T).T + T_ow[:3, 3])
                cand = (float(d.min()), int((d < 0.015).sum()), hid)
                best = cand if best is None or cand[0] < best[0] else best
            if best:
                rows.append({"sequence": seq, "object": e["object"], "when": when,
                             "closest_tip_m": best[0], "tips_within_15mm": best[1], "hand": best[2]})
    return rows


def overlay(seq, model, flips, k=None):
    params, T_dc, head = camera(seq)
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    poses = load_poses(seq, ts)
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    cells = []
    picks = [i for i in np.linspace(len(ts) * 0.2, len(ts) * 0.8, 12).astype(int) if len(poses.get(ts[i], {})) == 2][:3]
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        for hid, hp in poses[ts[i]].items():
            uv, z = to_pixels(model.world(hp, flips[hid]), ts[i], params, T_dc, head)
            u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
            c = (255, 160, 0) if hid == "0" else (0, 200, 255)
            for a, b in BONES:
                cv2.line(img, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), c, 4)
            for j in TIPS:
                cv2.circle(img, (int(u[j]), int(v[j])), 10, (0, 0, 255), -1)
        cells.append(cv2.resize(img, (560, 560)))
    cap.release()
    path = IMG / f"exp09_{seq}_hands.jpg"
    cv2.imwrite(str(path), np.hstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path.relative_to(ROOT))


def main():
    seqs = sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    report, flips_found = {}, {}
    for seq in seqs:
        model = HandModel(seq)
        bc = box_check(seq, model)
        flips = {h: max(FLIPS, key=lambda f: bc.get(f"hand {h}, mirror {f}", 0)) for h in ("0", "1")}
        report[seq] = {"inside_hand_box": bc, "best_mirror": flips}
        flips_found[seq] = flips
        print(seq, json.dumps(report[seq]), flush=True)
    events = pd.read_csv(ROOT / "out" / "hot3d_clips" / "events.csv")
    rows = []
    for seq in seqs:
        model = HandModel(seq)
        ev = events[(events.sequence == seq) & (events.wrist_min_dist_m <= 0.25)]
        rows += tip_distances(seq, model, flips_found[seq], ev)
        report[seq]["overlay"] = overlay(seq, model, flips_found[seq])
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tip_distances.csv", index=False)
    summ = {}
    for when, g in d.groupby("when"):
        summ[when] = {"n": len(g), "closest_tip_mm_median": round(1000 * g.closest_tip_m.median(), 1),
                      "closest_tip_mm_p90": round(1000 * g.closest_tip_m.quantile(0.9), 1),
                      "share_within_15mm": round(float((g.closest_tip_m < 0.015).mean()), 3),
                      "share_2plus_tips_within_15mm": round(float((g.tips_within_15mm >= 2).mean()), 3)}
    report["tip_to_object"] = summ
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
