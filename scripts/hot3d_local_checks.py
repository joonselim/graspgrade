"""Three number 2 checklist items that need no API call, on the HOT3D clips.

    python scripts/hot3d_local_checks.py video      # one pass per sequence video, 10 fps, cached
    python scripts/hot3d_local_checks.py clips      # per clip values, grades, mocap checks

Video only. Mocap is used for checking, never as an input.

Video pass (preview_rgb.mp4, every 3rd frame, 10 fps):
  - camera motion between sampled frames: corners tracked with Lucas-Kanade
    on a 352 x 352 grey image, a similarity transform fitted with RANSAC.
    Motion = translation in pixels (at 352) plus rotation in degrees.
  - MediaPipe Hands (tracking mode, two hands). The label is flipped for this
    non-mirrored view: MediaPipe "Left" is the person's right hand.

Per clip (clip hand and frame ranges from hot3d_clips.py):
  - camera stability: 90th percentile of camera translation over the clip.
  - retargeting recoverability: MediaPipe world landmarks of the clip hand,
    through dex-retargeting (Allegro, default config with its 1.6 size
    factor, made for MediaPipe input), during the contact. Value =
    detection rate x (1 - median vector error), so a hand that is not found
    or cannot be matched scores low.
  - action completeness: whether the clip hand is seen and moves in the
    reach second before contact and the release second after it.
Grades high, mid, low = 1, 0.5, 0. For camera and retargeting the cut points
are the tuning-split terciles, fixed. Completeness has a fixed rule.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ORIENT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_checks"
STEP = 3
SMALL = 352
FLIP = {"Left": 1, "Right": 0}      # MediaPipe label -> hand slot (0 left, 1 right)
SIDES = ["left", "right"]


def video_pass(seq):
    import mediapipe as mp
    path = OUT / "video" / f"{seq}.npz"
    if path.exists():
        return seq
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    frames, cam, lm2d, lm3d, score = [], [], [], [], []
    prev = None
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % STEP:
                i += 1
                continue
            _, img = cap.retrieve()
            grey = cv2.cvtColor(cv2.resize(img, (SMALL, SMALL)), cv2.COLOR_BGR2GRAY)
            motion = (np.nan, np.nan)
            if prev is not None:
                pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
                if pts is not None and len(pts) >= 8:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, grey, pts, None)
                    good = st.ravel() == 1
                    if good.sum() >= 8:
                        M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                             ransacReprojThreshold=2.0)
                        if M is not None:
                            motion = (float(np.hypot(M[0, 2], M[1, 2])),
                                      float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
            prev = grey
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            a2, a3, sc = np.full((2, 21, 2), np.nan), np.full((2, 21, 3), np.nan), np.zeros(2)
            if res.multi_handedness:
                for h, l2, l3 in zip(res.multi_handedness, res.multi_hand_landmarks, res.multi_hand_world_landmarks):
                    c = h.classification[0]
                    slot = FLIP[c.label]
                    if c.score <= sc[slot]:
                        continue
                    sc[slot] = c.score
                    a2[slot] = [(p.x, p.y) for p in l2.landmark]
                    a3[slot] = [(p.x, p.y, p.z) for p in l3.landmark]
            frames.append(i)
            cam.append(motion)
            lm2d.append(a2)
            lm3d.append(a3)
            score.append(sc)
            i += 1
    cap.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=np.array(frames), cam=np.array(cam), lm2d=np.array(lm2d, np.float32),
                        lm3d=np.array(lm3d, np.float32), score=np.array(score, np.float32))
    return seq


def headset_speed(seq, frames):
    """Mocap check only: headset rotation in degrees per 0.1 s at the sampled frames."""
    from scipy.spatial.transform import Rotation
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    head = pd.read_csv(gt / "headset_trajectory.csv").set_index("timestamp[ns]")
    q = head.reindex(ts[np.minimum(frames, len(ts) - 1)])[["q_wo_x", "q_wo_y", "q_wo_z", "q_wo_w"]].values
    out = np.full(len(frames), np.nan)
    ok = ~np.isnan(q).any(1)
    for k in range(1, len(frames)):
        if ok[k] and ok[k - 1]:
            out[k] = np.degrees((Rotation.from_quat(q[k]) * Rotation.from_quat(q[k - 1]).inv()).magnitude())
    return out


def clip_values(args):
    seq, clips = args
    from hot3d_copyable import Allegro
    z = np.load(OUT / "video" / f"{seq}.npz")
    fr, cam, lm2d, lm3d = z["frames"], z["cam"], z["lm2d"], z["lm3d"]
    robots = {s: Allegro(s, 1.6) for s in SIDES}
    head = headset_speed(seq, fr)
    rows = []
    for c in clips.itertuples():
        slot = SIDES.index(c.hand)
        sel = (fr >= c.clip_start) & (fr < c.clip_start + c.clip_frames)
        con = (fr >= c.contact_start) & (fr < c.contact_start + c.contact_frames)
        reach = (fr >= c.clip_start) & (fr < c.contact_start)
        rel = (fr >= c.contact_start + c.contact_frames) & (fr < c.clip_start + c.clip_frames)
        seen = ~np.isnan(lm2d[:, slot, 0, 0])
        t = cam[sel, 0]
        # retargeting during contact
        errs = []
        for k in np.where(con & seen)[0]:
            kp = lm3d[k, slot].astype(float)
            _, err = robots[c.hand].fit(kp)
            errs.append(err)
        det = float(seen[con].mean()) if con.any() else 0.0
        err = float(np.median(errs)) if errs else 1.0

        def moved(m):
            w = lm2d[m & seen, slot, 0]
            return float(np.linalg.norm(w.max(0) - w.min(0))) if len(w) >= 2 else 0.0

        rows.append({"sequence": seq, "object": c.object, "hand": c.hand, "clip_start": c.clip_start,
                     "camera_motion_p90_px": float(np.nanpercentile(t, 90)) if np.isfinite(t).any() else np.nan,
                     "camera_rotation_p90_deg": float(np.nanpercentile(np.abs(cam[sel, 1]), 90)) if np.isfinite(t).any() else np.nan,
                     "headset_rotation_p90_deg": float(np.nanpercentile(head[sel], 90)) if np.isfinite(head[sel]).any() else np.nan,
                     "hand_seen_in_contact": det, "retarget_error": err,
                     "retarget_value": det * (1 - min(err, 1.0)),
                     "reach_seen": float(seen[reach].mean()) if reach.any() else 0.0,
                     "release_seen": float(seen[rel].mean()) if rel.any() else 0.0,
                     "reach_wrist_move": moved(reach), "release_wrist_move": moved(rel)})
    return rows


def handedness_check(seq, n=60):
    """Mocap check only: is MediaPipe's flipped label the same hand as the mocap hand nearest in the image?"""
    from hot3d_hands import HandModel, camera, load_poses, to_pixels
    z = np.load(OUT / "video" / f"{seq}.npz")
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    params, T_dc, headdf = camera(seq)
    model = HandModel(seq)
    both = np.where(~np.isnan(z["lm2d"][:, 0, 0, 0]) | ~np.isnan(z["lm2d"][:, 1, 0, 0]))[0]
    pick = both[np.linspace(0, len(both) - 1, min(n, len(both))).astype(int)]
    tt = [ts[min(z["frames"][k], len(ts) - 1)] for k in pick]
    poses = load_poses(seq, tt)
    right = total = 0
    for k, t in zip(pick, tt):
        if t not in poses or t not in headdf.index:
            continue
        mocap = {}
        for hid, hp in poses[t].items():
            uv, zz = to_pixels(model.world(hp, "none" if hid == "0" else "x")[[5]], t, params, T_dc, headdf)
            u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
            mocap[int(hid)] = np.array([u[0], v[0]]) / 1408
        for slot in (0, 1):
            w = z["lm2d"][k, slot, 0]
            if np.isnan(w[0]) or not mocap:
                continue
            nearest = min(mocap, key=lambda h: np.linalg.norm(mocap[h] - w))
            if np.linalg.norm(mocap[nearest] - w) < 0.08:
                total += 1
                right += nearest == slot
    return right, total


def grade(x, lo, hi, higher_is_better=True):
    if np.isnan(x):
        return 0.0
    if not higher_is_better:
        x, lo, hi = -x, -hi, -lo
    return 1.0 if x >= hi else (0.5 if x >= lo else 0.0)


def cmd_clips():
    clips = pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips_labeled.csv")
    meta = pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips.csv")[["sequence", "object", "hand", "clip_start",
                                                                   "contact_start", "contact_frames"]]
    clips = clips.merge(meta, on=["sequence", "object", "hand", "clip_start"], how="left")
    jobs = [(s, g) for s, g in clips.groupby("sequence")]
    with Pool(12) as pool:
        rows = [r for rs in pool.imap_unordered(clip_values, jobs) for r in rs]
    v = pd.DataFrame(rows)
    d = clips.merge(v, on=["sequence", "object", "hand", "clip_start"], how="left")
    tune = d[d.split == "tune"]
    cuts = {"camera": list(np.nanpercentile(tune.camera_motion_p90_px, [33.3, 66.7])),
            "retarget": list(np.nanpercentile(tune.retarget_value, [33.3, 66.7]))}
    d["grade_camera_stability"] = [grade(x, *cuts["camera"], higher_is_better=False) for x in d.camera_motion_p90_px]
    d["grade_retargeting"] = [grade(x, *cuts["retarget"]) for x in d.retarget_value]
    ends = ((d.reach_seen >= 0.5) & (d.reach_wrist_move >= 0.03)).astype(int) + \
           ((d.release_seen >= 0.5) & (d.release_wrist_move >= 0.03)).astype(int)
    d["grade_completeness"] = ends / 2
    d.to_csv(OUT / "clip_checks.csv", index=False)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    ok = tune.camera_motion_p90_px.notna() & tune.headset_rotation_p90_deg.notna()
    rho = spearmanr(tune.camera_motion_p90_px[ok], tune.headset_rotation_p90_deg[ok])[0]
    hc = [handedness_check(s) for s in sorted(d.sequence.unique())[::9]]
    t = d[d.split == "tune"]
    summary = {
        "clips": len(d), "cuts_from_tuning_terciles": cuts,
        "camera_vs_headset_rotation_spearman_tune": round(float(rho), 3),
        "mediapipe_label_matches_mocap_hand": f"{sum(r for r, _ in hc)} of {sum(n for _, n in hc)}",
        "grade_share_tune": {g: t[g].value_counts(normalize=True).round(3).sort_index().to_dict()
                             for g in ("grade_camera_stability", "grade_retargeting", "grade_completeness")},
        "hand_seen_in_contact_median_tune": round(float(t.hand_seen_in_contact.median()), 3),
        "retarget_error_median_tune": round(float(t.retarget_error.median()), 3),
        "copyable_auc_tune_single_items": {
            g: round(float(roc_auc_score(t.copyable, t[g])), 3)
            for g in ("grade_camera_stability", "grade_retargeting", "grade_completeness",
                      "camera_motion_p90_px", "retarget_value", "retarget_error")
            if t[g].notna().all()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def main():
    cmd = sys.argv[1]
    if cmd == "video":
        seqs = sorted(pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips_labeled.csv").sequence.unique())
        with Pool(14) as pool:
            for i, s in enumerate(pool.imap_unordered(video_pass, seqs)):
                print(f"{i + 1}/{len(seqs)} {s}", flush=True)
    else:
        cmd_clips()


if __name__ == "__main__":
    main()
