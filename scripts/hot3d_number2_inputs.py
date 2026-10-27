"""Prepare every number 2 input that needs no API call.

    python scripts/hot3d_number2_inputs.py select    # final clip set and roles
    python scripts/hot3d_number2_inputs.py windows   # overlapping clips joined into SAM windows, cost
    python scripts/hot3d_number2_inputs.py cut       # one 10 fps mp4 per SAM window
    python scripts/hot3d_number2_inputs.py frames    # six 720 px frames per clip for Muse Spark

Roles (docs/MEASUREMENT_PLAN.md):
  score_train  400 random tuning clips (seed 20261026), weights of the final score
  item_train   the other tuning clips, for checks that are themselves learned (hand shape fit)
  held_out     all held-out clips, for the AUC
Ground truth: copyable v2 (EXP-16, EXP-20).
SAM gets the held-out and score_train clips. Overlapping clips of one
sequence share one request (EXP-11 pricing). Frame k of a 10 fps window is
preview frame window_start + 3k.
Muse Spark frames per clip: clip start (reach), contact start, one third and
two thirds into the contact, contact end, clip end (release).
Outputs go to out/number2/.
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "number2"
STEP = 3
SAM_PRICE_FRAME, SAM_MIN_FRAMES, PROMPTS = 0.20 / 1000, 50, 2
FRAME_PX = 720
SEED = 20261026


def clip_id(r):
    return f"{r.sequence}_{r.hand}_{int(r.clip_start):05d}"


def cmd_select():
    lab = ROOT / "out" / "hot3d_copyable"
    tune = pd.read_csv(lab / "tune_v2_clips_labeled.csv")
    held = pd.read_csv(lab / "held_out_v2_clips_labeled.csv")
    meta = pd.read_csv(ROOT / "out" / "hot3d_clips" / "clips.csv")[
        ["sequence", "hand", "clip_start", "object", "contact_start", "contact_frames", "other_hand_touches_pct"]]
    tune["role"] = "item_train"
    tune.loc[tune.sample(400, random_state=SEED).index, "role"] = "score_train"
    held["role"] = "held_out"
    d = pd.concat([tune, held]).merge(meta, on=["sequence", "hand", "clip_start", "object"], how="left")
    d["clip_id"] = [clip_id(r) for r in d.itertuples()]
    d["two_hands"] = d.other_hand_touches_pct > 50
    cols = ["clip_id", "role", "sequence", "person", "object", "hand", "clip_start", "clip_frames",
            "contact_start", "contact_frames", "frame", "two_hands", "copyable"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(OUT / "clips_final.csv", index=False)
    print(d.groupby("role").agg(clips=("clip_id", "size"), copyable=("copyable", "mean"),
                                two_hands=("two_hands", "mean")).round(3))


def cmd_windows():
    d = pd.read_csv(OUT / "clips_final.csv")
    d = d[d.role != "item_train"]
    rows = []
    for seq, g in d.groupby("sequence"):
        cur = None
        for r in g.sort_values("clip_start").itertuples():
            a, b = int(r.clip_start), int(r.clip_start + r.clip_frames)
            if cur and a <= cur["end"]:
                cur["end"] = max(cur["end"], b)
                cur["clips"].append(r.clip_id)
            else:
                if cur:
                    rows.append(cur)
                cur = {"sequence": seq, "role": r.role, "start": a, "end": b, "clips": [r.clip_id]}
        rows.append(cur)
    w = pd.DataFrame(rows)
    w["window_id"] = [f"{r.sequence}_{r.start:05d}" for r in w.itertuples()]
    w["frames_10fps"] = np.ceil((w.end - w.start) / STEP).astype(int)
    w["billed"] = np.maximum(w.frames_10fps, SAM_MIN_FRAMES) * PROMPTS
    w["n_clips"] = w.clips.map(len)
    w["clips"] = w.clips.map(json.dumps)
    w.to_csv(OUT / "sam_windows.csv", index=False)
    s = w.groupby("role").agg(windows=("window_id", "size"), clips=("n_clips", "sum"), billed_frames=("billed", "sum"))
    s["cost_usd"] = (s.billed_frames * SAM_PRICE_FRAME).round(2)
    print(s, "\ntotal $", round(float(s.cost_usd.sum()), 2))


def cut_one(r):
    dst = OUT / "sam_windows" / f"{r.window_id}.mp4"
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = next((ARIA / r.sequence).glob("*preview_rgb.mp4"))
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{r.start / 30:.4f}", "-i", str(src),
                    "-frames:v", str(r.frames_10fps), "-vf", f"select='not(mod(n,{STEP}))',setpts=N/(10*TB)",
                    "-r", "10", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(tmp)], check=True)
    tmp.rename(dst)
    return 1


def cmd_cut():
    w = pd.read_csv(OUT / "sam_windows.csv")
    with ThreadPoolExecutor(8) as pool:
        made = sum(pool.map(cut_one, w.itertuples()))
    # check a sample of frame counts
    bad = 0
    for r in w.sample(min(20, len(w)), random_state=SEED).itertuples():
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", str(OUT / "sam_windows" / f"{r.window_id}.mp4")],
                               capture_output=True, text=True).stdout.strip() or 0)
        bad += n != r.frames_10fps
    print(f"made {made} windows, {len(w)} in total, frame count mismatches in a sample of 20: {bad}")


def frame_indices(r):
    a, cs, cf = int(r.clip_start), int(r.contact_start), int(r.contact_frames)
    b = a + int(r.clip_frames) - 1
    return [a, cs, cs + cf // 3, cs + 2 * cf // 3, cs + cf - 1, b]


def frames_for_sequence(args):
    seq, g = args
    probe = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    n_video = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    want = {}
    for r in g.itertuples():
        for j, k in enumerate(frame_indices(r)):
            # the video has one frame fewer than the RGB timestamps (EXP-08), so the last index may not exist
            want.setdefault(min(k, n_video - 1), []).append((r.clip_id, j))
    todo = {k: v for k, v in want.items()
            if not all((OUT / "muse_frames" / cid / f"f{j}.jpg").exists() for cid, j in v)}
    if not todo:
        return 0
    cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
    i, last, n = 0, max(todo), 0
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in todo:
            _, img = cap.retrieve()
            img = cv2.resize(img, (FRAME_PX, FRAME_PX), interpolation=cv2.INTER_AREA)
            for cid, j in todo[i]:
                p = OUT / "muse_frames" / cid / f"f{j}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                n += 1
        i += 1
    cap.release()
    return n


def cmd_frames():
    from multiprocessing import Pool
    d = pd.read_csv(OUT / "clips_final.csv")
    d = d[d.role != "item_train"]
    with Pool(10) as pool:
        n = sum(pool.map(frames_for_sequence, list(d.groupby("sequence"))))
    have = sum(1 for _ in (OUT / "muse_frames").glob("*/f*.jpg"))
    print(f"wrote {n} frames, {have} on disk for {len(d)} clips (expected {6 * len(d)})")


def main():
    {"select": cmd_select, "windows": cmd_windows, "cut": cmd_cut, "frames": cmd_frames}[sys.argv[1]]()


if __name__ == "__main__":
    main()
