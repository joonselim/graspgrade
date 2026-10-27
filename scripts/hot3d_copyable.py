"""Copyable label for HOT3D grasps: can an Allegro hand copy this grasp? (no API call)

    python scripts/hot3d_copyable.py run --split tune      # compute per event, cache
    python scripts/hot3d_copyable.py report --split tune   # rates for a grid of thresholds

Definition from the plan: map the mocap hand to an Allegro hand and put the
Allegro wrist at the human wrist. At the contact frame the grasp is copyable
when the Allegro thumb tip and at least one other Allegro fingertip are near
the object surface, and no part of the Allegro hand is deep inside the
object. The wrist may move a few cm to find such a placement. The distance
thresholds are set on the tuning split and then fixed.

Steps per grasp event (EXP-07 events, wrist near and QA passed):
1. Contact frame = middle of the object move. The hand is the one whose
   fingertips are closest to the object (EXP-09 landmarks).
2. UmeTrack landmarks to the MediaPipe 21 point order, wrist frame as in
   dex-retargeting's single_hand_detector, vector retargeting (wrist to four
   fingertips) with the dex-retargeting Allegro config for that hand. The
   size factor is Allegro wrist-to-index-tip length over the person's.
3. Allegro forward kinematics, every link point placed in the world through
   the same wrist frame, then signed distance to the object surface
   (negative = inside), for every wrist shift on a 1 cm grid up to 4 cm.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ASSETS, pose  # noqa: E402
from hot3d_hands import TIPS, HandModel, load_poses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEX = ROOT / "ext" / "dex-retargeting"
OUT = ROOT / "out" / "hot3d_copyable"
SIDE = {"0": "left", "1": "right"}
FLIP = {"0": "none", "1": "x"}
OP2MANO = {"right": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
           "left": np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])}
STEP_CM, MAX_CM = 1, 4
GRID = np.array([(x, y, z) for x in range(-MAX_CM, MAX_CM + 1) for y in range(-MAX_CM, MAX_CM + 1)
                 for z in range(-MAX_CM, MAX_CM + 1) if x * x + y * y + z * z <= MAX_CM ** 2], float) / 100
GRID_R = np.linalg.norm(GRID, axis=1)
# MediaPipe order from UmeTrack landmarks. Index 1 (thumb base) is filled from the joint chain.
MP_FROM_UME = [5, None, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]


def mediapipe_points(model, hand_pose, flip):
    """21 points in MediaPipe order, world frame (m)."""
    from hot3d_hands import FLIPS
    from scipy.spatial.transform import Rotation
    angles = np.array(hand_pose["joint_angles"])
    lm = model.landmarks(angles)
    # thumb base: rest position of joint 1 moved by joint 0
    local = np.eye(4)
    local[:3, :3] = Rotation.from_rotvec(model.axes[0] * angles[0]).as_matrix()
    local[:3, 3] = model.rest[0] - local[:3, :3] @ model.rest[0]
    cmc = local[:3, :3] @ model.rest[1] + local[:3, 3]
    pts = np.array([cmc if i is None else lm[i] for i in MP_FROM_UME]) * FLIPS[flip] / 1000.0
    T = pose(hand_pose["wrist_xform"]["q_wxyz"], hand_pose["wrist_xform"]["t_xyz"])
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def wrist_frame(kp):
    """dex-retargeting SingleHandDetector.estimate_frame_from_hand_points, copied."""
    points = kp[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


class Allegro:
    def __init__(self, side, scale, kind="vector"):
        from dex_retargeting.constants import HandType, RetargetingType, RobotName, get_default_config_path
        from dex_retargeting.retargeting_config import RetargetingConfig
        RetargetingConfig.set_default_urdf_dir(str(DEX / "assets" / "robots" / "hands"))
        hand = HandType.right if side == "right" else HandType.left
        self.kind = kind
        if kind == "position":
            # offline config: fingertips and finger bases matched in position, free 6-DoF hand base
            path = get_default_config_path(RobotName.allegro, RetargetingType.position, hand)
            override = {"low_pass_alpha": 1.0}
        else:
            path = get_default_config_path(RobotName.allegro, RetargetingType.vector, hand)
            override = {"low_pass_alpha": 1.0, "scaling_factor": float(scale)}
        self.ret = RetargetingConfig.load_from_file(path, override=override).build()
        opt = self.ret.optimizer
        self.robot = opt.robot
        if kind == "position":
            self.idx = np.asarray(opt.target_link_human_indices)
            self.task_links = list(opt.body_names)
            human_index_of_link = dict(zip(self.task_links, self.idx))
        else:
            self.idx = opt.target_link_human_indices
            self.task_links = list(opt.task_link_names)
            human_index_of_link = dict(zip(self.task_links, self.idx[1]))
        import pinocchio as pin
        # BODY frames only: get_link_index looks names up as BODY, a joint frame name gives an invalid id.
        # The dummy links of the free base are not part of the hand.
        self.links = [f.name for f in self.robot.model.frames
                      if f.type == pin.FrameType.BODY and not f.name.startswith("dummy")]
        self.link_idx = [self.robot.get_link_index(n) for n in self.links]
        # thumb tip is human index 4, the other fingertips 8, 12, 16
        self.thumb = self.links.index(next(n for n, h in human_index_of_link.items() if h == 4))
        self.others = [self.links.index(n) for n, h in human_index_of_link.items() if h in (8, 12, 16)]
        self.others_human = [h for n, h in human_index_of_link.items() if h in (8, 12, 16)]
        self.wrist = self.links.index("wrist")
        self.side, self.scale = side, scale

    def index_length(self):
        self.robot.compute_forward_kinematics(np.zeros(self.robot.dof))
        p = {n: self.robot.get_link_pose(self.robot.get_link_index(n))[:3, 3] for n in self.task_links + ["wrist"]}
        link = self.task_links[list(self.idx[1]).index(8)]
        return float(np.linalg.norm(p[link] - p["wrist"]))

    def fit(self, kp_world):
        """Retarget one frame. Returns all link points in the world (m) and the retargeting error:
        vector kind, median relative vector error; position kind, median position error in m."""
        w = kp_world[0]
        kp = kp_world - w
        M = wrist_frame(kp) @ OP2MANO[self.side]
        joint_pos = kp @ M
        self.ret.reset()
        if self.kind == "position":
            ref = joint_pos[self.idx]
            qpos = self.ret.retarget(ref)
            self.robot.compute_forward_kinematics(qpos)
            pb = np.array([self.robot.get_link_pose(i)[:3, 3] for i in self.link_idx])
            got = np.array([pb[self.links.index(t)] for t in self.task_links])
            err = float(np.median(np.linalg.norm(got - ref, axis=1)))
            # the free base already places the hand, the points are in the wrist frame of the input
            return w + pb @ M.T, err
        ref = joint_pos[self.idx[1]] - joint_pos[self.idx[0]]
        qpos = self.ret.retarget(ref)
        self.robot.compute_forward_kinematics(qpos)
        pb = np.array([self.robot.get_link_pose(i)[:3, 3] for i in self.link_idx])
        wb = pb[self.wrist]
        got = np.array([pb[self.links.index(t)] - wb for t in self.task_links])
        err = float(np.median(np.linalg.norm(got - ref * self.scale, axis=1) / np.linalg.norm(ref * self.scale, axis=1)))
        return w + (pb - wb) @ M.T, err


class Surface:
    def __init__(self, uid, n=30000):
        m = trimesh.load(ASSETS / f"{uid}.glb", force="mesh")
        pts, face = trimesh.sample.sample_surface(m, n, seed=0)
        self.pts, self.nrm = np.asarray(pts), np.asarray(m.face_normals[face])
        self.tree = cKDTree(self.pts)

    def signed(self, p):
        """Signed distance (m) from points in the object frame, negative inside."""
        d, i = self.tree.query(p)
        s = np.sign(np.einsum("ij,ij->i", p - self.pts[i], self.nrm[i]))
        return d * np.where(s == 0, 1, s)


def run(split_name, source="events", method="ratio"):
    """source "events": EXP-07 object moves, contact frame = middle of the move, hand = closest.
    source "clips": hot3d_clips.py clips, contact frame and hand given by the clip.
    method "ratio": vector retargeting scaled by Allegro length / person length (EXP-10),
    "scale1": vector retargeting, scale 1.0, "position": position retargeting with a free base (EXP-15)."""
    split = json.loads((ROOT / "splits" / "hot3d_aria.json").read_text())
    seqs = set(split[split_name])
    if source == "events":
        ev = pd.read_csv(ROOT / "out" / "hot3d_clips" / "events.csv")
        ev = ev[ev.sequence.isin(seqs) & (ev.wrist_min_dist_m <= 0.25) & (ev.clip_qa_pass_pct >= 90)].copy()
        ev["frame"] = ev.move_start + ev.move_frames // 2
        ev["hand_given"] = None
    else:
        ev = pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips.csv")
        ev = ev[ev.sequence.isin(seqs) & ev.moved & (ev.clip_qa_pass_pct >= 90)].copy()
        ev["frame"] = ev.contact_frame
        ev["hand_given"] = ev.hand.map({v: k for k, v in SIDE.items()})
    tag = split_name if source == "events" else f"{split_name}_clips"
    if method != "ratio":
        tag = f"{tag}_{method}"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sd_store = [], []
    surfaces = {}
    for seq, g in ev.groupby("sequence"):
        gt = ARIA / seq / "ground_truth"
        meta = json.loads((gt / "metadata.json").read_text())
        uid_of = dict(zip(meta["object_names"], meta["object_uids"]))
        qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
        ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
        objs = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"])
        model = HandModel(seq)
        human_len = float(np.linalg.norm(model.lm_rest[1] - model.lm_rest[5])) / 1000.0
        hands = {}
        frames = [int(f) for f in g.frame]
        poses = load_poses(seq, ts[frames])
        for e, k in zip(g.itertuples(), frames):
            uid = int(uid_of[e.object])
            if uid not in surfaces:
                surfaces[uid] = Surface(uid)
            surf = surfaces[uid]
            if (ts[k], uid) not in objs.index or not poses.get(ts[k]):
                continue
            o = objs.loc[(ts[k], uid)]
            T_ow = np.linalg.inv(pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values,
                                      o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values))

            def to_obj(p):
                return (T_ow[:3, :3] @ p.T).T + T_ow[:3, 3]

            best = None
            for hid, hp in poses[ts[k]].items():
                if hid not in SIDE or (e.hand_given is not None and hid != e.hand_given):
                    continue
                kp = mediapipe_points(model, hp, FLIP[hid])
                tips_sd = surf.signed(to_obj(kp[[4, 8, 12, 16, 20]]))
                if best is None or np.abs(tips_sd).min() < np.abs(best[2]).min():
                    best = (hid, kp, tips_sd)
            if best is None:
                continue
            hid, kp, human_sd = best
            side = SIDE[hid]
            if (side, seq) not in hands:
                if method == "ratio":
                    a = Allegro(side, 1.0)
                    hands[(side, seq)] = Allegro(side, a.index_length() / human_len)
                else:
                    hands[(side, seq)] = Allegro(side, 1.0, kind="position" if method == "position" else "vector")
            robot = hands[(side, seq)]
            pts, err = robot.fit(kp)
            p_obj = to_obj(pts)
            # rotation part of world->object only, the grid is a world shift
            shifts = (T_ow[:3, :3] @ GRID.T).T
            sd = surf.signed((p_obj[None, :, :] + shifts[:, None, :]).reshape(-1, 3)).reshape(len(GRID), len(pts))
            tips = sd[:, [robot.thumb] + robot.others]
            # fidelity: Allegro fingertip to the matching human fingertip, and Allegro wrist to human wrist
            fid = np.linalg.norm(pts[[robot.thumb] + robot.others] - kp[[4] + robot.others_human], axis=1)
            sd_store.append(np.c_[tips, sd.min(1)].astype(np.float32))
            rows.append({**({"clip_start": int(e.clip_start), "clip_frames": int(e.clip_frames)} if source == "clips" else {}),
                         "sequence": seq, "person": seq.split("_")[0], "object": e.object, "hand": side,
                         "frame": k, "scale": round(robot.scale, 3), "retarget_vector_error": round(err, 3),
                         "human_hand_len_mm": round(1000 * human_len, 1),
                         "tip_to_human_tip_mm_median": round(1000 * float(np.median(fid)), 1),
                         "thumb_to_human_thumb_mm": round(1000 * float(fid[0]), 1),
                         "wrist_to_human_wrist_mm": round(1000 * float(np.linalg.norm(pts[robot.wrist] - kp[0])), 1),
                         "human_thumb_sd_mm": round(1000 * human_sd[0], 1),
                         "human_closest_tip_mm": round(1000 * np.abs(human_sd).min(), 1),
                         "human_tips_within_15mm": int((np.abs(human_sd) < 0.015).sum())})
        print(f"{seq}: {len(g)} events", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}_events.csv", index=False)
    np.save(OUT / f"{tag}_sd.npy", np.stack(sd_store))
    print(f"{len(df)} events, scale median {df.scale.median():.2f}, retarget error median {df.retarget_vector_error.median():.3f}")


def labels(sd, near, deep, radius):
    """sd: events x shifts x (thumb, 3 other tips, min over all links). True if any shift within radius works."""
    ok_tip = (sd[:, :, :4] <= near) & (sd[:, :, :4] >= -deep)
    good = ok_tip[:, :, 0] & ok_tip[:, :, 1:].any(2) & (sd[:, :, 4] >= -deep)
    good &= (GRID_R <= radius + 1e-9)[None, :]
    return good.any(1)


def report(split_name):
    df = pd.read_csv(OUT / f"{split_name}_events.csv")
    sd = np.load(OUT / f"{split_name}_sd.npy")
    grid = []
    for near in (0.01, 0.015, 0.02, 0.03):
        for deep in (0.005, 0.01, 0.02):
            for radius in (0.0, 0.02, 0.04):
                grid.append({"near_cm": near * 100, "deep_cm": deep * 100, "wrist_shift_cm": radius * 100,
                             "copyable_pct": round(100 * float(labels(sd, near, deep, radius).mean()), 1)})
    g = pd.DataFrame(grid)
    human = ((df.human_tips_within_15mm >= 2) & (df.human_thumb_sd_mm.abs() <= 15)).mean()
    out = {"events": len(df), "scale_median": float(df.scale.median()),
           "retarget_vector_error_median": float(df.retarget_vector_error.median()),
           "human_rule_pct (thumb and one more tip within 15 mm)": round(100 * float(human), 1),
           "grid": grid}
    (OUT / f"{split_name}_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    print(g.pivot_table(index=["near_cm", "deep_cm"], columns="wrist_shift_cm", values="copyable_pct").to_string())
    return df, sd


def figures(split_name, near=0.015, deep=0.01, radius=0.02):
    """Rate curves over thresholds, and four example grasps drawn on the video."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hot3d_calib_check import IMG, ORIENT
    from hot3d_hands import camera, to_pixels
    df = pd.read_csv(OUT / f"{split_name}_events.csv")
    sd = np.load(OUT / f"{split_name}_sd.npy")
    nears = np.arange(0.005, 0.0401, 0.0025)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for r, c in ((0.0, "#868e96"), (0.02, "#1c7ed6"), (0.04, "#2b8a3e")):
        ax.plot(nears * 100, [100 * labels(sd, n, deep, r).mean() for n in nears], marker=".", color=c,
                label=f"wrist may shift up to {r * 100:.0f} cm")
    ax.axhspan(20, 80, color="#fff3bf", zorder=0, label="target band 20 to 80%")
    ax.axvline(near * 100, color="#e8590c", ls="--", lw=1, label=f"chosen: near {near * 100:.1f} cm")
    ax.set_xlabel("near threshold for Allegro fingertips (cm), deep limit 1 cm")
    ax.set_ylabel("copyable grasps (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"HOT3D tuning split: {len(df)} grasps")
    fig.tight_layout()
    fig.savefig(IMG / "exp10_copyable_rate.png", dpi=130)

    lab = labels(sd, near, deep, radius)
    rng = np.random.default_rng(20261026)
    pick = list(rng.choice(np.where(lab)[0], 2, replace=False)) + list(rng.choice(np.where(~lab)[0], 2, replace=False))
    cells = []
    for i in pick:
        e = df.iloc[i]
        seq, k = e["sequence"], int(e["frame"])
        gt = ARIA / seq / "ground_truth"
        meta = json.loads((gt / "metadata.json").read_text())
        uid = int(dict(zip(meta["object_names"], meta["object_uids"]))[e["object"]])
        qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
        ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
        hid = {v: k2 for k2, v in SIDE.items()}[e["hand"]]
        model = HandModel(seq)
        hp = load_poses(seq, [ts[k]])[ts[k]][hid]
        kp = mediapipe_points(model, hp, FLIP[hid])
        robot = Allegro(e["hand"], float(e["scale"]))
        pts, _ = robot.fit(kp)
        # best shift: the allowed one with the most tips near, for drawing
        ok = (GRID_R <= radius + 1e-9)
        tip = (sd[i, :, :4] <= near) & (sd[i, :, :4] >= -deep)
        score = np.where(ok, tip.sum(1) + 10 * (tip[:, 0] & tip[:, 1:].any(1) & (sd[i, :, 4] >= -deep)), -1)
        pts = pts + GRID[int(np.argmax(score))]
        params, T_dc, head = camera(seq)
        objs = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"])
        o = objs.loc[(ts[k], uid)]
        T_wo = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
        sp = Surface(uid, 4000).pts
        sp_w = (T_wo[:3, :3] @ sp.T).T + T_wo[:3, 3]
        cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        _, img = cap.read()
        cap.release()
        layers = [(sp_w, (0, 200, 0), 2), (kp, (255, 255, 255), 5), (pts, (0, 140, 255), 7),
                  (pts[[robot.thumb] + robot.others], (0, 0, 255), 12)]
        for p, c, r in layers:
            uv, z = to_pixels(p, ts[k], params, T_dc, head)
            u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
            for x, y, zz in zip(u, v, z):
                if zz > 0 and 0 <= x < 1408 and 0 <= y < 1408:
                    cv2.circle(img, (int(x), int(y)), r, c, -1)
        uv, _ = to_pixels(pts, ts[k], params, T_dc, head)
        u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
        cx, cy, half = int(np.median(u)), int(np.median(v)), 330
        x0, y0 = int(np.clip(cx - half, 0, 1408 - 2 * half)), int(np.clip(cy - half, 0, 1408 - 2 * half))
        crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        label = ("copyable" if lab[i] else "not copyable") + f": {e['object']}"
        cv2.putText(crop, label, (15, 45), 0, 1.2, (0, 0, 0), 6)
        cv2.putText(crop, label, (15, 45), 0, 1.2, (255, 255, 255), 2)
        cells.append(cv2.resize(crop, (420, 420)))
    cv2.imwrite(str(IMG / "exp10_examples.jpg"), np.hstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "figures"])
    ap.add_argument("--split", default="tune", choices=["tune", "held_out"])
    ap.add_argument("--source", default="events", choices=["events", "clips"])
    ap.add_argument("--method", default="ratio", choices=["ratio", "scale1", "position"])
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.split, a.source, a.method)
    else:
        {"report": report, "figures": figures}[a.cmd](a.split if a.source == "events" else f"{a.split}_clips")


if __name__ == "__main__":
    main()
