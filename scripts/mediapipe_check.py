"""MediaPipe hand detection rate on OpenGraph episodes (gloved hands).

    python scripts/mediapipe_check.py --episodes 0 1

Uses the 720p cuts from scripts/cut_episodes.py, so frame i of the mp4 is
frame_index i in the parquet. Each frame is joined with the tactile-derived
annotation.hand label ("left", "right", "both", "none"), which tells us which
hands are touching something and so should be in view.

Handedness: MediaPipe labels hands as if the image were a mirror. These videos
are not mirrored, so a hand MediaPipe calls "Left" is the person's right hand.
We flip the label before comparing.
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CUTS = ROOT / "data" / "episodes_720p"
DATA = ROOT / "data" / "ego-tactile" / "data" / "chunk-000" / "file-000.parquet"
OUT = ROOT / "out" / "mediapipe_check"
FLIP = {"Left": "right", "Right": "left"}


def run_episode(idx, labels):
    cap = cv2.VideoCapture(str(CUTS / f"episode_{idx:06d}.mp4"))
    rows = []
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  model_complexity=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as hands:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found, wrist_x = set(), {}
            if res.multi_handedness:
                for h, lm in zip(res.multi_handedness, res.multi_hand_landmarks):
                    side = FLIP[h.classification[0].label]
                    found.add(side)
                    # normalized 0..1, left edge = 0. Lets us tell a wrong label
                    # from a hand that is simply out of frame.
                    wrist_x[side] = lm.landmark[0].x
            rows.append({"frame_index": i, "n_hands": len(found),
                         "det_left": "left" in found, "det_right": "right" in found,
                         "wrist_x_left": wrist_x.get("left"), "wrist_x_right": wrist_x.get("right")})
            i += 1
    cap.release()
    df = pd.DataFrame(rows).merge(labels, on="frame_index", how="left")
    return df


def summarize(df):
    lab = df["annotation.hand"]
    need_r = lab.isin(["right", "both"])
    need_l = lab.isin(["left", "both"])
    touching = lab != "none"
    pct = lambda s: round(100.0 * float(s.mean()), 1) if len(s) else None
    return {
        "frames": int(len(df)),
        "any_hand_pct": pct(df.n_hands > 0),
        "any_hand_while_touching_pct": pct((df.n_hands > 0)[touching]),
        "right_found_when_right_touching_pct": pct(df.det_right[need_r]),
        "left_found_when_left_touching_pct": pct(df.det_left[need_l]),
        "both_found_when_both_touching_pct": pct((df.det_left & df.det_right)[lab == "both"]),
        # sanity check on the flip: if the label mapping were backwards, these
        # "wrong side only" rates would be high
        "only_wrong_side_when_right_touching_pct": pct((df.det_left & ~df.det_right)[lab == "right"]),
        "label_mix_pct": {k: round(100.0 * v, 1) for k, v in lab.value_counts(normalize=True).items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ann = pd.read_parquet(DATA, columns=["episode_index", "frame_index", "annotation.hand"])
    report = {}
    for idx in args.episodes:
        labels = ann[ann.episode_index == idx].drop(columns="episode_index")
        df = run_episode(idx, labels)
        df.to_parquet(OUT / f"episode_{idx:06d}.parquet")
        report[idx] = summarize(df)
        print(f"episode {idx}: {json.dumps(report[idx])}", flush=True)

    allf = pd.concat(pd.read_parquet(OUT / f"episode_{i:06d}.parquet") for i in args.episodes)
    report["all"] = summarize(allf)
    print("\nall:", json.dumps(report["all"], indent=1))
    (OUT / "summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
