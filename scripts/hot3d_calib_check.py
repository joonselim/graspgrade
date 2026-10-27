"""Which image orientation does the HOT3D camera calibration use? (no API call)

    python scripts/hot3d_calib_check.py P0001_10a27bf7 P0002_016222d1 P0010_0ecbf39f

preview_rgb.mp4 is upright, but raw Aria images are rotated 90 degrees. The
calibration in ground_truth/camera_models.json (FISHEYE624) may belong to
either. For sampled frames we project each object's 3D model with the mocap
poses, take the bounding box of the projected points, and compare it (IoU)
with the dataset 2D box in box2d_objects.csv under three orientations:
as projected, rotated 90 degrees clockwise, and counter-clockwise. An overlay on the video frame checks the winner by eye.

Writes out/hot3d_calib/report.json and docs/experiments/img/exp08_*.jpg.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
ARIA = ROOT / "data" / "hot3d" / "aria"
ASSETS = ROOT / "data" / "hot3d" / "assets"
OUT = ROOT / "out" / "hot3d_calib"
IMG = ROOT / "docs" / "experiments" / "img"
SIZE = 1408
N_SAMPLES = 25
ORIENT = {
    "as_projected": lambda u, v: (u, v),
    "rot90_cw": lambda u, v: (SIZE - 1 - v, u),
    "rot90_ccw": lambda u, v: (v, SIZE - 1 - u),
}


def pose(q_wxyz, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = t
    return T


def fisheye624(p, params):
    """Aria FISHEYE624 projection of camera-frame points (N x 3) to pixels."""
    f, cx, cy = params[:3]
    k = params[3:9]
    p0, p1 = params[9:11]
    s = params[11:15]
    a, b = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    poly = 1 + sum(k[i] * th2 ** (i + 1) for i in range(6))
    scale = np.where(r > 1e-9, th * poly / np.maximum(r, 1e-9), 1.0)
    xr, yr = a * scale, b * scale
    rd2 = xr * xr + yr * yr
    xd = xr + (2 * xr * xr + rd2) * p0 + 2 * xr * yr * p1 + s[0] * rd2 + s[1] * rd2 * rd2
    yd = yr + 2 * xr * yr * p0 + (2 * yr * yr + rd2) * p1 + s[2] * rd2 + s[3] * rd2 * rd2
    return np.stack([f * xd + cx, f * yd + cy], 1)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_seq(seq):
    gt = ARIA / seq / "ground_truth"
    cam = next(c for c in json.loads((gt / "camera_models.json").read_text()) if c["label"] == "camera-rgb")
    T_dc = pose(cam["T_Device_Camera"]["quaternion_wxyz"], cam["T_Device_Camera"]["translation_xyz"])
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    head = pd.read_csv(gt / "headset_trajectory.csv").set_index("timestamp[ns]")
    objs = pd.read_csv(gt / "dynamic_objects.csv")
    boxes = pd.read_csv(gt / "box2d_objects.csv")
    boxes = boxes[(boxes["stream_id"] == "214-1") & (boxes["visibility_ratio[%]"] > 0.6)].dropna()
    return cam, T_dc, ts, head, objs, boxes


def project_object(verts, row_obj, T_wd, T_dc, params):
    T_wo = pose(row_obj[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, row_obj[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
    T_cw = np.linalg.inv(T_wd @ T_dc)
    pc = (T_cw @ T_wo @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    pc = pc[pc[:, 2] > 0.02]
    return fisheye624(pc, params) if len(pc) else None


def check(seq, meshes, draw=False):
    cam, T_dc, ts, head, objs, boxes = load_seq(seq)
    params = np.array(cam["projectionParams"])
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cand_ts = sorted(set(boxes["timestamp[ns]"]) & set(head.index))
    pick = [cand_ts[i] for i in np.linspace(0, len(cand_ts) - 1, N_SAMPLES).astype(int)]
    scores = {o: [] for o in ORIENT}
    shown = None
    for t in pick:
        h = head.loc[t]
        T_wd = pose(h[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, h[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
        ot = objs[objs["timestamp[ns]"] == t].set_index("object_uid")
        for _, b in boxes[boxes["timestamp[ns]"] == t].iterrows():
            uid = b["object_uid"]
            if uid not in ot.index or uid not in meshes:
                continue
            uv = project_object(meshes[uid], ot.loc[uid], T_wd, T_dc, params)
            if uv is None:
                continue
            gtb = (b["x_min[pixel]"], b["y_min[pixel]"], b["x_max[pixel]"], b["y_max[pixel]"])
            for o, fn in ORIENT.items():
                u, v = fn(uv[:, 0], uv[:, 1])
                scores[o].append(iou((u.min(), v.min(), u.max(), v.max()), gtb))
        if draw and shown is None and len(ot):
            shown = t
    result = {"video_frames": n_video, "rgb_timestamps": len(ts), "boxes_compared": len(scores["as_projected"]),
              "mean_iou": {o: round(float(np.mean(s)), 3) for o, s in scores.items()}}
    if draw:
        result["overlay"] = overlay(seq, cap, ts, shown, head, objs, boxes, meshes, T_dc, params)
    cap.release()
    return result


def frame_at(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    return img if ok else None


def overlay(seq, cap, ts, t, head, objs, boxes, meshes, T_dc, params):
    """GT boxes (green) and projected models (orange) on video frame k, one panel per orientation."""
    k = int(np.searchsorted(ts, t))
    h = head.loc[t]
    T_wd = pose(h[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, h[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
    ot = objs[objs["timestamp[ns]"] == t].set_index("object_uid")
    cells = []
    for orient in ORIENT:
        img = frame_at(cap, k).copy()
        for _, b in boxes[boxes["timestamp[ns]"] == t].iterrows():
            cv2.rectangle(img, (int(b["x_min[pixel]"]), int(b["y_min[pixel]"])),
                          (int(b["x_max[pixel]"]), int(b["y_max[pixel]"])), (0, 220, 0), 4)
        for uid in ot.index:
            if uid not in meshes:
                continue
            uv = project_object(meshes[uid], ot.loc[uid], T_wd, T_dc, params)
            if uv is None:
                continue
            u, v = ORIENT[orient](uv[:, 0], uv[:, 1])
            for x, y in zip(u[::7].astype(int), v[::7].astype(int)):
                if 0 <= x < SIZE and 0 <= y < SIZE:
                    cv2.circle(img, (x, y), 2, (0, 120, 255), -1)
        cv2.putText(img, orient, (30, 80), 0, 2.4, (0, 0, 0), 10)
        cv2.putText(img, orient, (30, 80), 0, 2.4, (255, 255, 255), 3)
        cells.append(cv2.resize(img, (560, 560)))
    path = IMG / f"exp08_{seq}_orientation.jpg"
    cv2.imwrite(str(path), np.hstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path.relative_to(ROOT))


def hand_check(seq, n=150, offsets=(-2, -1, 0, 1, 2)):
    """Independent check on the video itself: MediaPipe hand boxes on preview frame k
    against the dataset hand boxes at RGB timestamp k + offset, per orientation."""
    import mediapipe as mp
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    hb = pd.read_csv(gt / "box2d_hands.csv")
    hb = hb[(hb["stream_id"] == "214-1") & (hb["visibility_ratio[%]"] > 0.6)].dropna()
    by_t = {t: g[["x_min[pixel]", "y_min[pixel]", "x_max[pixel]", "y_max[pixel]"]].values for t, g in hb.groupby("timestamp[ns]")}
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ks = [k for k in range(2, n_video - 2) if ts[k] in by_t]
    ks = [ks[i] for i in np.linspace(0, len(ks) - 1, n).astype(int)]
    scores = {(o, d): [] for o in ORIENT for d in offsets}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for k in ks:
            img = frame_at(cap, k)
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            mpb = [(min(p.x for p in h.landmark) * SIZE, min(p.y for p in h.landmark) * SIZE,
                    max(p.x for p in h.landmark) * SIZE, max(p.y for p in h.landmark) * SIZE) for h in res.multi_hand_landmarks]
            for d in offsets:
                g = by_t.get(ts[k + d])
                if g is None:
                    continue
                for o, fn in ORIENT.items():
                    for x0, y0, x1, y1 in g:
                        ua, va = fn(np.array([x0, x1]), np.array([y0, y1]))
                        box = (ua.min(), va.min(), ua.max(), va.max())
                        scores[(o, d)].append(max(iou(box, m) for m in mpb))
    cap.release()
    return {f"{o} offset {d:+d}": round(float(np.mean(s)), 3) for (o, d), s in scores.items() if s}


def main():
    seqs = sys.argv[1:] or ["P0001_10a27bf7"]
    meshes = {}
    for p in ASSETS.glob("*.glb"):
        m = trimesh.load(p, force="mesh")
        v = np.asarray(m.vertices)
        meshes[int(p.stem)] = v[np.linspace(0, len(v) - 1, min(len(v), 3000)).astype(int)]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for i, seq in enumerate(seqs):
        report[seq] = check(seq, meshes, draw=True)
        report[seq]["mediapipe_hand_iou"] = hand_check(seq)
        print(seq, json.dumps(report[seq]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
