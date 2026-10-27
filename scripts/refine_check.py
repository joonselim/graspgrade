"""Local data checks for OpenGraph that need no Meta API call.

    python scripts/refine_check.py

1. Tactile ground truth. Per hand, "touching" = sum of the 80 taxels above
   --threshold. Compared with the dataset labels (annotation.hand), frame by
   frame and onset by onset. At 0, episodes 63 to 71 disagree on about a
   quarter of frames because those gloves read small pressure at rest.
2. Label noise. Very short contact segments and very short gaps between
   contacts, which a real boundary detector should not be asked to find.
3. Camera shake from the head IMU gyro (rad/s), per episode.
4. SAM cost if clips are sent at 10 fps instead of 30 fps.

Writes out/refine_check/summary.json and per_episode.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ego-tactile" / "data" / "chunk-000" / "file-000.parquet"
OUT = ROOT / "out" / "refine_check"
SHORT = 5          # frames, the boundary tolerance
SAM_PRICE_FRAME = 0.20 / 1000


def segments(mask):
    """Start and end (exclusive) of each run of True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def onset_offsets(ref, test):
    """For each onset in ref, signed frame offset to the nearest onset in test."""
    r = np.array([s for s, _ in segments(ref)])
    t = np.array([s for s, _ in segments(test)])
    if not len(r) or not len(t):
        return []
    return [int(t[np.argmin(np.abs(t - x))] - x) for x in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=800)
    thr = ap.parse_args().threshold
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["episode_index", "frame_index", "annotation.hand", "observation.tactile.left",
            "observation.tactile.right", "observation.imu.head"]
    d = pd.read_parquet(DATA, columns=cols)

    rows, offsets = [], {"left": [], "right": []}
    agree = {"left": [0, 0], "right": [0, 0]}
    seg_len, gap_len = [], []
    for ep, e in d.groupby("episode_index"):
        e = e.sort_values("frame_index")
        lab = e["annotation.hand"].values
        row = {"episode": int(ep), "frames": len(e)}
        for h in ("left", "right"):
            tac = np.stack(e[f"observation.tactile.{h}"].values).sum(1) > thr
            lb = np.isin(lab, [h, "both"])
            agree[h][0] += int((tac == lb).sum())
            agree[h][1] += len(e)
            offsets[h] += onset_offsets(lb, tac)
            row[f"{h}_label_vs_tactile_agree_pct"] = round(100 * float((tac == lb).mean()), 2)
        anyc = lab != "none"
        segs = segments(anyc)
        seg_len += [b - a for a, b in segs]
        gap_len += [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        # imu.head names: ax ay az gx gy gz grx gry grz, the last three are the gyro
        gyro = np.linalg.norm(np.stack(e["observation.imu.head"].values)[:, 6:9], axis=1)
        row["head_gyro_median"] = round(float(np.median(gyro)), 3)
        row["head_gyro_p95"] = round(float(np.percentile(gyro, 95)), 3)
        row["contact_segments"] = len(segs)
        rows.append(row)

    per_ep = pd.DataFrame(rows)
    per_ep.to_csv(OUT / "per_episode.csv", index=False)

    def off_stats(o):
        o = np.array(o)
        return {"onsets": int(len(o)), "median": float(np.median(o)), "within_0": round(100 * float(np.mean(o == 0)), 1),
                "within_5": round(100 * float(np.mean(np.abs(o) <= SHORT)), 1),
                "p90_abs": float(np.percentile(np.abs(o), 90))}

    seg_len, gap_len = np.array(seg_len), np.array(gap_len)
    frames = int(per_ep.frames.sum())
    summary = {
        "frames": frames,
        "tactile_threshold": thr,
        "label_vs_tactile_frame_agree_pct": {h: round(100 * a / n, 2) for h, (a, n) in agree.items()},
        "label_onset_to_tactile_onset_frames": {h: off_stats(o) for h, o in offsets.items()},
        "contact_segments": int(len(seg_len)),
        "segments_shorter_than_5_frames": int((seg_len < SHORT).sum()),
        "gaps_shorter_than_5_frames": int((gap_len < SHORT).sum()),
        "segment_length_frames_p10_p50_p90": [float(np.percentile(seg_len, q)) for q in (10, 50, 90)],
        "head_gyro_p95_by_episode_min_median_max": [float(per_ep.head_gyro_p95.min()),
                                                    float(per_ep.head_gyro_p95.median()),
                                                    float(per_ep.head_gyro_p95.max())],
        "shakiest_episodes": per_ep.nlargest(5, "head_gyro_p95")[["episode", "head_gyro_p95"]].to_dict("records"),
        "sam_cost_all_frames_30fps_usd": round(frames * SAM_PRICE_FRAME, 2),
        "sam_cost_all_frames_10fps_usd": round(frames / 3 * SAM_PRICE_FRAME, 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
