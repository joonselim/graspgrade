"""Does undistorting the HOT3D fisheye frames help MediaPipe? (EXP-25, tuning persons, no API call)

    python scripts/hot3d_fisheye_check.py            # run and report
    python scripts/hot3d_fisheye_check.py report     # report from the cache
    FISHEYE_FOCAL=1.0 python scripts/hot3d_fisheye_check.py   # side check with another pinhole focal

One contact frame per clip (the middle one) for 400 clips of tuning persons
(seed 20261026). MediaPipe Hands (static image mode) runs on
  raw:          the preview frame as it is (what the project uses now),
  undistorted:  the same frame turned into a pinhole image with the dataset
                lens model (FISHEYE624), focal length 0.7 of the fisheye one
                so that most of the view stays inside the image.
Truth is the mocap hand: UmeTrack landmarks projected with the lens model to
the preview. MediaPipe points from the undistorted image are mapped back to
the preview, so both are measured in the same pixels.
Per frame and condition:
  found       a MediaPipe hand whose median point error is under 0.5 hand size,
  error_2d    median point error over hand size (wrist to middle finger base),
  shape_err   hand shape features (hot3d_hand_shape.py) from MediaPipe 3D points
              against the same features from mocap, per feature over its mocap
              spread, mean over features.
The decision rule is in docs/MEASUREMENT_PLAN.md (fisheye check).
Writes out/hot3d_checks/fisheye.csv, fisheye.json, docs/experiments/img/exp25_*.png.
"""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, fisheye624, pose  # noqa: E402
from hot3d_hand_shape import FEATS, UME_TO_MP, shape  # noqa: E402
from hot3d_hands import HandModel, camera, load_poses, to_pixels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
IMG = ROOT / "docs" / "experiments" / "img"
N2 = ROOT / "out" / "number2"
SIZE, N_CLIPS, SEED = 1408, 400, 20261026
PIN_SCALE = float(os.environ.get("FISHEYE_FOCAL", "0.7"))  # 0.7 is the main setting, others are side checks
TAG = "" if PIN_SCALE == 0.7 else f"_f{PIN_SCALE}"
POINTS = [i for i in range(21) if i != 1]  # MediaPipe point 1 has no UmeTrack match (UME_TO_MP)
FLIP = {"0": "none", "1": "x"}  # hand 0 left as is, hand 1 right mirrored (EXP-09)


def raw_to_preview(uv):
    return np.stack([SIZE - 1 - uv[:, 1], uv[:, 0]], 1)


class Pinhole:
    """Upright pinhole image with the same centre as the raw camera, focal PIN_SCALE x fisheye focal."""

    def __init__(self, params):
        self.params, self.f, self.c = params, PIN_SCALE * params[0], (SIZE - 1) / 2
        X, Y = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
        prev = self.to_preview(np.stack([X.ravel(), Y.ravel()], 1).astype(float))
        self.mapx = prev[:, 0].reshape(SIZE, SIZE).astype(np.float32)
        self.mapy = prev[:, 1].reshape(SIZE, SIZE).astype(np.float32)

    def to_preview(self, XY):
        """Upright pinhole pixel -> preview pixel. Upright (X, Y) is raw pinhole (Y, SIZE - 1 - X)."""
        x, y = XY[:, 1], SIZE - 1 - XY[:, 0]
        ray = np.stack([(x - self.c) / self.f, (y - self.c) / self.f, np.ones(len(x))], 1)
        return raw_to_preview(fisheye624(ray, self.params))

    def inside(self, pc):
        """True if all camera-frame points (N x 3, raw camera) land inside the upright pinhole image."""
        x = self.f * pc[:, 0] / pc[:, 2] + self.c
        y = self.f * pc[:, 1] / pc[:, 2] + self.c
        X, Y = SIZE - 1 - y, x
        return bool(((X >= 0) & (X < SIZE) & (Y >= 0) & (Y < SIZE)).all())

    def image(self, preview):
        return cv2.remap(preview, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def run_mediapipe(hands, img):
    """List of (2D points in pixels N x 2, 3D world points 21 x 3) per detected hand."""
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for l2, l3 in zip(res.multi_hand_landmarks or [], res.multi_hand_world_landmarks or []):
        out.append((np.array([(p.x * SIZE, p.y * SIZE) for p in l2.landmark]),
                    np.array([(p.x, p.y, p.z) for p in l3.landmark])))
    return out


def best_match(dets, gt2d, size):
    best = None
    for p2, p3 in dets:
        err = float(np.median(np.linalg.norm(p2[POINTS] - gt2d[POINTS], axis=1))) / size
        if best is None or err < best[0]:
            best = (err, p3)
    return best


def one_sequence(args):
    import mediapipe as mp
    seq, rows = args
    params, T_dc, head = camera(seq)
    pin = Pinhole(params)
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    frames = {int(r["frame"]): r for r in rows}
    poses = load_poses(seq, {ts[f] for f in frames})
    model = HandModel(seq)
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    out, shots = [], {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.5) as hands:
        for f, r in sorted(frames.items()):
            h = "0" if r["hand"] == "left" else "1"
            if ts[f] not in poses or h not in poses[ts[f]] or ts[f] not in head.index:
                continue
            world = model.world(poses[ts[f]][h], FLIP[h])
            uv, z = to_pixels(world, ts[f], params, T_dc, head)
            if (z <= 0.02).any():
                continue
            gt2d = raw_to_preview(uv)[UME_TO_MP]
            size = float(np.linalg.norm(gt2d[9] - gt2d[0]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok or size < 5:
                continue
            gt_shape = shape(model.landmarks(np.array(poses[ts[f]][h]["joint_angles"]))[UME_TO_MP][None])[0]
            und = pin.image(img)
            # fisheye radius over focal is close to the angle from the optical axis (radians)
            off = float(np.degrees(np.hypot(*(uv[5] - params[1:3])) / params[0]))
            hp = head.loc[ts[f]]
            T_wd = pose(hp[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, hp[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
            pc = (np.linalg.inv(T_wd @ T_dc) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
            row = {"sequence": seq, "clip_id": r["clip_id"], "frame": f, "hand": r["hand"], "hand_px": size,
                   "off_axis_deg": off, "inside_pinhole": pin.inside(pc)}
            for name, picture, back in (("raw", img, None), ("undistorted", und, pin.to_preview)):
                dets = run_mediapipe(hands, picture)
                if back is not None:
                    dets = [(back(p2), p3) for p2, p3 in dets]
                m = best_match(dets, gt2d, size)
                found = m is not None and m[0] < 0.5
                row[f"{name}_found"] = bool(found)
                row[f"{name}_error_2d"] = m[0] if m else np.nan
                row[f"{name}_shape"] = shape(m[1][None])[0].tolist() if found else None
            row["mocap_shape"] = gt_shape.tolist()
            out.append(row)
            if len(shots) < 1 and row["raw_found"] and row["undistorted_found"]:
                shots[f] = (img, und, gt2d)
    cap.release()
    return out, [(seq, f, *v) for f, v in shots.items()]


def pick_frames():
    c = pd.read_csv(N2 / "clips_final.csv")
    c = c[c.role != "held_out"].sample(N_CLIPS, random_state=SEED)
    c["frame"] = c.contact_start + c.contact_frames // 2
    return c


def run():
    c = pick_frames()
    jobs = [(s, g[["clip_id", "frame", "hand"]].to_dict("records")) for s, g in c.groupby("sequence")]
    rows, shots = [], []
    with Pool(10) as pool:
        for r, s in pool.imap_unordered(one_sequence, jobs):
            rows += r
            shots += s
    d = pd.DataFrame(rows)
    d.to_json(OUT / f"fisheye_rows{TAG}.json", orient="records")
    draw_example(shots)
    return d


def shape_error(d, name):
    """Per frame: mean over features of |MediaPipe - mocap| / mocap spread of that feature."""
    m = np.array(d.mocap_shape.tolist())
    spread = np.nanstd(m, 0)
    errs = []
    for v, g in zip(d[f"{name}_shape"], m):
        errs.append(np.nan if v is None else float(np.nanmean(np.abs(np.array(v) - g) / spread)))
    return np.array(errs)


def report(d):
    for name in ("raw", "undistorted"):
        d[f"{name}_shape_err"] = shape_error(d, name)
    both = d.raw_found & d.undistorted_found
    res = {"frames": int(len(d)), "clips_asked": N_CLIPS, "pinhole_focal_scale": PIN_SCALE,
           "off_axis_deg_median": round(float(d.off_axis_deg.median()), 1),
           "off_axis_deg_p90": round(float(d.off_axis_deg.quantile(0.9)), 1)}
    for name in ("raw", "undistorted"):
        res[name] = {"found": round(float(d[f"{name}_found"].mean()), 3),
                     "error_2d_median_found": round(float(d.loc[d[f"{name}_found"], f"{name}_error_2d"].median()), 3),
                     "error_2d_median_both": round(float(d.loc[both, f"{name}_error_2d"].median()), 3),
                     "shape_err_median_both": round(float(np.nanmedian(d.loc[both, f"{name}_shape_err"])), 3)}
    res["frames_found_in_both"] = int(both.sum())
    ins = d.inside_pinhole.astype(bool)
    res["hand_inside_pinhole_share"] = round(float(ins.mean()), 3)
    res["inside_pinhole_only"] = {"frames": int(ins.sum()),
                                  "raw_found": round(float(d.loc[ins, "raw_found"].mean()), 3),
                                  "undistorted_found": round(float(d.loc[ins, "undistorted_found"].mean()), 3)}
    bins = [0, 30, 45, 90]
    d["zone"] = pd.cut(d.off_axis_deg, bins, labels=["centre (<30 deg)", "middle (30-45 deg)", "edge (>45 deg)"])
    res["by_zone"] = {str(z): {"frames": int(len(g)), "raw_found": round(float(g.raw_found.mean()), 3),
                               "undistorted_found": round(float(g.undistorted_found.mean()), 3),
                               "raw_error_2d": round(float(g.loc[g.raw_found & g.undistorted_found, "raw_error_2d"].median()), 3),
                               "undistorted_error_2d": round(float(g.loc[g.raw_found & g.undistorted_found, "undistorted_error_2d"].median()), 3)}
                      for z, g in d.groupby("zone", observed=True)}
    r, u = res["raw"], res["undistorted"]
    res["decision"] = {
        "found_gain_points": round(100 * (u["found"] - r["found"]), 1),
        "error_2d_drop_pct": round(100 * (1 - u["error_2d_median_both"] / r["error_2d_median_both"]), 1),
        "shape_err_drop_pct": round(100 * (1 - u["shape_err_median_both"] / r["shape_err_median_both"]), 1),
    }
    dec = res["decision"]
    dec["undistort"] = bool(dec["found_gain_points"] >= 5 or dec["error_2d_drop_pct"] >= 20 or dec["shape_err_drop_pct"] >= 20)
    (OUT / f"fisheye{TAG}.json").write_text(json.dumps(res, indent=1))
    d.drop(columns=["raw_shape", "undistorted_shape", "mocap_shape"]).to_csv(OUT / f"fisheye{TAG}.csv", index=False)
    print(json.dumps(res, indent=1))
    plot_zones(d)


def plot_zones(d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    zs = list(d.zone.cat.categories)
    x = np.arange(len(zs))
    for k, (name, col) in enumerate((("raw", "#1971c2"), ("undistorted", "#e8590c"))):
        found = [d.loc[d.zone == z, f"{name}_found"].mean() for z in zs]
        both = d.raw_found & d.undistorted_found
        err = [d.loc[(d.zone == z) & both, f"{name}_error_2d"].median() for z in zs]
        axes[0].bar(x + (k - 0.5) * 0.38, found, 0.38, color=col, label=name)
        axes[1].bar(x + (k - 0.5) * 0.38, err, 0.38, color=col, label=name)
    n = [int((d.zone == z).sum()) for z in zs]
    for ax, t in ((axes[0], "Share of frames with the hand found"),
                  (axes[1], "Median 2D error over hand size\n(frames found in both)")):
        ax.set_xticks(x, [f"{z}\n{k} frames" for z, k in zip(zs, n)], fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1)
    fig.suptitle("MediaPipe on the raw fisheye preview vs the undistorted frame, by angle of the wrist from the "
                 "image centre", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / f"exp25_fisheye_zones{TAG}.png", dpi=120)
    plt.close(fig)


def draw_example(shots):
    if not shots:
        return
    shots = sorted(shots, key=lambda s: s[0])[:2]
    fig, axes = plt.subplots(len(shots), 2, figsize=(8, 4 * len(shots)), squeeze=False)
    for row, (seq, f, img, und, gt2d) in zip(axes, shots):
        row[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row[0].scatter(gt2d[POINTS, 0], gt2d[POINTS, 1], s=4, c="lime")
        row[0].set_title(f"{seq} frame {f}: raw preview, mocap hand in green", fontsize=8)
        row[1].imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
        row[1].set_title("same frame undistorted (pinhole, focal 0.7)", fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(IMG / f"exp25_fisheye_example{TAG}.png", dpi=110)
    plt.close(fig)


def main():
    if sys.argv[1:2] == ["report"]:
        d = pd.read_json(OUT / f"fisheye_rows{TAG}.json", orient="records")
    else:
        d = run()
    report(d)


if __name__ == "__main__":
    main()
