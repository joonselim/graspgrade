"""Copyable label v2 (docs/MEASUREMENT_PLAN.md, "Copyable definition v2"), tuning clips only.

    python scripts/hot3d_copyable_v2.py run       # per clip geometry, cached
    python scripts/hot3d_copyable_v2.py report    # labels, acceptance checks, figures

Per clip at its contact frame:
  - position retargeting of the clip hand (Allegro, free base),
  - Allegro surface = points on the visual mesh of every link, placed by forward kinematics,
  - for every hand shift up to 2 cm (1 cm grid): signed distance of the four
    Allegro fingertips to the object, distance of each Allegro fingertip to
    the matching human fingertip, and the deepest Allegro mesh point.
Copyable when some shift has: grip (Allegro thumb and one more fingertip
within 1.5 cm of the surface), reach (every human finger that touches has its
Allegro fingertip within 2 cm of it), and no collision (no mesh point deeper
than 5 mm). Thresholds were fixed before computing.
"""

import json
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hot3d_copyable as hc  # noqa: E402
from hot3d_calib_check import ARIA, pose  # noqa: E402
from hot3d_hands import HandModel, load_poses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_copyable"
URDF_DIR = hc.DEX / "assets" / "robots" / "hands" / "allegro_hand"
SHIFT_MAX = 0.02
SHIFTS = hc.GRID[hc.GRID_R <= SHIFT_MAX + 1e-9]
NEAR, REACH, DEPTH = 0.015, 0.02, 0.005
HUMAN_TOUCH = 0.015
PTS_PER_LINK = 300


def link_meshes(side):
    """Visual mesh points per link, in the link frame."""
    root = ET.parse(URDF_DIR / f"allegro_hand_{side}.urdf").getroot()
    out = {}
    for link in root.findall("link"):
        pts = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            m = trimesh.load(URDF_DIR / mesh.get("filename"), force="mesh")
            if mesh.get("scale"):
                m.apply_scale([float(s) for s in mesh.get("scale").split()])
            p, _ = trimesh.sample.sample_surface(m, PTS_PER_LINK, seed=0)
            o = vis.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            pts.append((T[:3, :3] @ np.asarray(p).T).T + T[:3, 3])
        if pts:
            out[link.get("name")] = np.vstack(pts)
    return out


def clip_geometry(args):
    seq, clips = args
    gt = ARIA / seq / "ground_truth"
    meta = json.loads((gt / "metadata.json").read_text())
    uid_of = dict(zip(meta["object_names"], meta["object_uids"]))
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    objs = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"])
    model = HandModel(seq)
    human_len = float(np.linalg.norm(model.lm_rest[1] - model.lm_rest[5]))
    poses = load_poses(seq, ts[clips.contact_frame.values])
    robots = {s: hc.Allegro(s, 1.0, kind="position") for s in ("left", "right")}
    meshes = {s: link_meshes(s) for s in ("left", "right")}
    surfaces, rows, arrays = {}, [], []
    for c in clips.itertuples():
        k, hid = int(c.contact_frame), "0" if c.hand == "left" else "1"
        uid = int(uid_of[c.object])
        if (ts[k], uid) not in objs.index or hid not in poses.get(ts[k], {}):
            continue
        if uid not in surfaces:
            surfaces[uid] = hc.Surface(uid)
        surf = surfaces[uid]
        o = objs.loc[(ts[k], uid)]
        T_ow = np.linalg.inv(pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values,
                                  o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values))
        R_ow, t_ow = T_ow[:3, :3], T_ow[:3, 3]
        kp = hc.mediapipe_points(model, poses[ts[k]][hid], hc.FLIP[hid])
        robot = robots[c.hand]
        pts, err = robot.fit(kp)  # leaves the forward kinematics of this fit in robot.robot.data
        w = kp[0]
        M = hc.wrist_frame(kp - w) @ hc.OP2MANO[c.hand]
        mesh_w = []
        for name, p in meshes[c.hand].items():
            T = robot.robot.get_link_pose(robot.robot.get_link_index(name))
            mesh_w.append(w + ((T[:3, :3] @ p.T).T + T[:3, 3]) @ M.T)
        mesh_o = (R_ow @ np.vstack(mesh_w).T).T + t_ow
        tips_w = pts[[robot.thumb] + robot.others]
        human_tips_w = kp[[4] + robot.others_human]
        tips_o = (R_ow @ tips_w.T).T + t_ow
        human_sd = surf.signed((R_ow @ kp[[4, 8, 12, 16]].T).T + t_ow)
        sh_o = (R_ow @ SHIFTS.T).T  # a world shift seen in the object frame
        tip_sd = surf.signed((tips_o[None] + sh_o[:, None]).reshape(-1, 3)).reshape(len(SHIFTS), 4)
        reach = np.linalg.norm((tips_w[None] + SHIFTS[:, None]) - human_tips_w[None], axis=2)
        mesh_min = np.array([surf.signed(mesh_o + s).min() for s in sh_o])
        arrays.append({"tip_sd": tip_sd.astype(np.float32), "reach": reach.astype(np.float32),
                       "mesh_min": mesh_min.astype(np.float32),
                       "human_touch": (np.abs(human_sd) < HUMAN_TOUCH)})
        rows.append({"sequence": seq, "person": seq.split("_")[0], "object": c.object, "hand": c.hand,
                     "clip_start": int(c.clip_start), "clip_frames": int(c.clip_frames), "frame": k,
                     "human_hand_len_mm": round(human_len, 1), "position_error_mm": round(1000 * err, 1),
                     "tip_to_human_tip_mm": round(1000 * float(np.median(reach[np.argmin(np.linalg.norm(SHIFTS, axis=1))])), 1),
                     "wrist_to_human_wrist_mm": round(1000 * float(np.linalg.norm(pts[robot.wrist] - w)), 1),
                     "human_fingers_touching": int((np.abs(human_sd) < HUMAN_TOUCH).sum())})
    return rows, arrays


def cmd_run():
    split = json.loads((ROOT / "splits" / "hot3d_aria.json").read_text())
    clips = pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips.csv")
    clips = clips[clips.sequence.isin(split["tune"]) & clips.moved & (clips.clip_qa_pass_pct >= 90)]
    jobs = [(s, g) for s, g in clips.groupby("sequence")]
    rows, arrays = [], []
    with Pool(10) as pool:
        for i, (r, a) in enumerate(pool.imap(clip_geometry, jobs)):
            rows += r
            arrays += a
            print(f"{i + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tune_v2_clips.csv", index=False)
    np.savez_compressed(OUT / "tune_v2_geometry.npz",
                        tip_sd=np.stack([a["tip_sd"] for a in arrays]), reach=np.stack([a["reach"] for a in arrays]),
                        mesh_min=np.stack([a["mesh_min"] for a in arrays]),
                        human_touch=np.stack([a["human_touch"] for a in arrays]))
    print(len(rows), "clips")


def label(g, near=NEAR, reach=REACH, depth=DEPTH, parts=False):
    tip_sd, rch, mesh_min, touch = g["tip_sd"], g["reach"], g["mesh_min"], g["human_touch"]
    grip = (tip_sd[:, :, 0] <= near) & (tip_sd[:, :, 1:] <= near).any(2)
    reach_ok = ((rch <= reach) | ~touch[:, None, :]).all(2)
    free = mesh_min >= -depth
    lab = (grip & reach_ok & free).any(1)
    if parts:
        return lab, {"no shift gives grip": float((~grip.any(1)).mean()),
                     "no shift gives reach": float((~reach_ok.any(1)).mean()),
                     "no shift is collision free": float((~free.any(1)).mean())}
    return lab


def within_object_corr(df):
    rs, ws = [], []
    for _, g in df.groupby("object"):
        if g.copyable.nunique() == 2 and g.human_hand_len_mm.nunique() > 1:
            rs.append(spearmanr(g.human_hand_len_mm, g.copyable)[0])
            ws.append(len(g))
    return float(np.average(rs, weights=ws)) if rs else float("nan")


def cmd_report():
    df = pd.read_csv(OUT / "tune_v2_clips.csv")
    g = dict(np.load(OUT / "tune_v2_geometry.npz"))
    lab, parts = label(g, parts=True)
    df["copyable"] = lab.astype(int)
    share = float(df.copyable.mean())
    corr = within_object_corr(df)
    grid = {f"reach {r * 100:.0f} cm, depth {d * 1000:.0f} mm": round(float(label(g, reach=r, depth=d).mean()), 3)
            for r in (0.01, 0.02, 0.03) for d in (0.002, 0.005, 0.01)}
    obj = df.groupby("object").copyable.agg(["mean", "size"])
    obj = obj[obj["size"] >= 25].sort_values("mean")
    cans = df.object.str.startswith(("can_", "carton_", "bottle_"))
    out = {
        "clips": len(df), "copyable_share": round(share, 3),
        "share_in_20_80": bool(0.2 <= share <= 0.8),
        "hand_size_corr_within_object": round(corr, 3), "hand_size_ok": bool(abs(corr) <= 0.1),
        "accepted": bool(0.2 <= share <= 0.8 and abs(corr) <= 0.1),
        "fails_by_condition": {k: round(v, 3) for k, v in parts.items()},
        "sensitivity": grid,
        "share_by_person": df.groupby("person").copyable.mean().round(3).to_dict(),
        "hand_len_by_person": df.groupby("person").human_hand_len_mm.first().to_dict(),
        "cans_cartons_bottles": round(float(df.copyable[cans].mean()), 3),
        "lowest_objects": obj.head(6)["mean"].round(2).to_dict(),
        "highest_objects": obj.tail(6)["mean"].round(2).to_dict(),
        "tip_to_human_tip_mm_median": float(df.tip_to_human_tip_mm.median()),
        "wrist_to_human_wrist_mm_median": float(df.wrist_to_human_wrist_mm.median()),
        "deepest_mesh_point_mm_median_no_shift": round(float(-1000 * np.median(g["mesh_min"][:, np.argmin(np.linalg.norm(SHIFTS, axis=1))])), 1),
    }
    (OUT / "tune_v2_report.json").write_text(json.dumps(out, indent=1))
    df.to_csv(OUT / "tune_v2_clips_labeled.csv", index=False)
    print(json.dumps(out, indent=1))


def cmd_figures():
    """Summary chart, and example grasps for each outcome drawn on the video."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hot3d_calib_check import IMG, ORIENT
    from hot3d_hands import camera, to_pixels
    df = pd.read_csv(OUT / "tune_v2_clips_labeled.csv")
    g = dict(np.load(OUT / "tune_v2_geometry.npz"))
    rep = json.loads((OUT / "tune_v2_report.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    reaches, depths = (0.01, 0.02, 0.03), (0.002, 0.005, 0.01)
    for r, c in zip(reaches, ("#868e96", "#1c7ed6", "#2b8a3e")):
        axes[0].plot([d * 1000 for d in depths], [100 * label(g, reach=r, depth=d).mean() for d in depths],
                     "o-", color=c, label=f"reach {r * 100:.0f} cm")
    axes[0].axhspan(20, 80, color="#fff3bf", zorder=0)
    axes[0].plot([5], [100 * rep["copyable_share"]], "*", ms=14, color="#e8590c", label="fixed thresholds")
    axes[0].set_xlabel("allowed Allegro mesh depth inside the object (mm)")
    axes[0].set_ylabel("copyable (%)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Share over thresholds (tuning clips)")
    fails = rep["fails_by_condition"]
    axes[1].barh(list(fails)[::-1], [100 * v for v in list(fails.values())[::-1]], color="#c92a2a")
    axes[1].set_xlabel("% of clips where no hand shift passes the condition")
    axes[1].set_title("Why clips are not copyable")
    p = sorted(rep["share_by_person"])
    axes[2].plot([rep["hand_len_by_person"][q] for q in p], [100 * rep["share_by_person"][q] for q in p], "o",
                 color="#2b8a3e", label=f"v2 (within object r = {rep['hand_size_corr_within_object']:+.2f})")
    old = pd.read_csv(OUT / "tune_clips_events.csv")
    old["copyable"] = hc.labels(np.load(OUT / "tune_clips_sd.npy"), 0.015, 0.01, 0.02)
    po = old.groupby("person").copyable.mean()
    axes[2].plot([rep["hand_len_by_person"][q] for q in p], [100 * po[q] for q in p], "o", color="#868e96",
                 label="EXP-10 label")
    axes[2].set_xlabel("person hand length (mm)")
    axes[2].set_ylabel("copyable (%)")
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=7)
    axes[2].set_title("Copyable share per tuning person")
    fig.tight_layout()
    fig.savefig(IMG / "exp16_copyable_v2.png", dpi=125)

    lab, _ = label(g, parts=True)
    grip = ((g["tip_sd"][:, :, 0] <= NEAR) & (g["tip_sd"][:, :, 1:] <= NEAR).any(2))
    reach_ok = ((g["reach"] <= REACH) | ~g["human_touch"][:, None, :]).all(2)
    free = g["mesh_min"] >= -DEPTH
    kinds = {"copyable": np.where(lab)[0],
             "collision": np.where(~lab & (grip & reach_ok).any(1) & ~(grip & reach_ok & free).any(1))[0],
             "reach": np.where(~lab & ~reach_ok.any(1))[0]}
    rng = np.random.default_rng(20261026)
    cells = []
    for kind, idx in kinds.items():
        for i in rng.choice(idx, 2, replace=False):
            e = df.iloc[i]
            seq, k, side = e.sequence, int(e.frame), e.hand
            gt = ARIA / seq / "ground_truth"
            meta = json.loads((gt / "metadata.json").read_text())
            uid = int(dict(zip(meta["object_names"], meta["object_uids"]))[e.object])
            qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
            ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
            hid = "0" if side == "left" else "1"
            model = HandModel(seq)
            kp = hc.mediapipe_points(model, load_poses(seq, [ts[k]])[ts[k]][hid], hc.FLIP[hid])
            robot = hc.Allegro(side, 1.0, kind="position")
            pts, _ = robot.fit(kp)
            w = kp[0]
            M = hc.wrist_frame(kp - w) @ hc.OP2MANO[side]
            mesh_w = []
            for name, p in link_meshes(side).items():
                T = robot.robot.get_link_pose(robot.robot.get_link_index(name))
                mesh_w.append(w + ((T[:3, :3] @ p[::3].T).T + T[:3, 3]) @ M.T)
            mesh_w = np.vstack(mesh_w)
            # draw at the shift that passes, or that passes grip and reach, or no shift
            ok, gr = grip[i] & reach_ok[i] & free[i], grip[i] & reach_ok[i]
            s = int(np.argmax(ok)) if ok.any() else (int(np.argmax(gr)) if gr.any() else int(np.argmin(np.linalg.norm(SHIFTS, axis=1))))
            mesh_w = mesh_w + SHIFTS[s]
            o = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"]).loc[(ts[k], uid)]
            T_wo = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
            surf = hc.Surface(uid, 8000)
            T_ow = np.linalg.inv(T_wo)
            sd = surf.signed((T_ow[:3, :3] @ mesh_w.T).T + T_ow[:3, 3])
            sp_w = (T_wo[:3, :3] @ surf.pts[::2].T).T + T_wo[:3, 3]
            params, T_dc, head = camera(seq)
            cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            _, img = cap.read()
            cap.release()
            layers = [(sp_w, (0, 200, 0), 2), (mesh_w[sd >= -DEPTH], (0, 160, 255), 2),
                      (mesh_w[sd < -DEPTH], (0, 0, 255), 4), (kp, (255, 255, 255), 6)]
            for p_, c_, r_ in layers:
                if not len(p_):
                    continue
                uv, z = to_pixels(p_, ts[k], params, T_dc, head)
                u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
                for x, y, zz in zip(u, v, z):
                    if zz > 0 and 0 <= x < 1408 and 0 <= y < 1408:
                        cv2.circle(img, (int(x), int(y)), r_, c_, -1)
            uv, _ = to_pixels(kp, ts[k], params, T_dc, head)
            u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
            half = 300
            x0 = int(np.clip(np.median(u) - half, 0, 1408 - 2 * half))
            y0 = int(np.clip(np.median(v) - half, 0, 1408 - 2 * half))
            crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
            title = {"copyable": "copyable", "collision": "not copyable: collision",
                     "reach": "not copyable: reach"}[kind]
            for col, wd in (((0, 0, 0), 6), ((255, 255, 255), 2)):
                cv2.putText(crop, title, (12, 40), 0, 1.0, col, wd)
                cv2.putText(crop, e.object, (12, 580), 0, 1.0, col, wd)
            cells.append(cv2.resize(crop, (330, 330)))
    cv2.imwrite(str(IMG / "exp16_examples.jpg"), np.vstack([np.hstack(cells[0::2]), np.hstack(cells[1::2])]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    {"run": cmd_run, "report": cmd_report, "figures": cmd_figures}[sys.argv[1]]()


if __name__ == "__main__":
    main()
