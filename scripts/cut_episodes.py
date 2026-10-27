"""Cut the OpenGraph LeRobot v3.0 videos into one 720p H.264 mp4 per episode.

    python scripts/cut_episodes.py                 # all 72 episodes
    python scripts/cut_episodes.py --episodes 0 7  # just these

Why this exists:
  - LeRobot v3.0 packs several episodes into one mp4. The episode metadata gives
    each episode's file and start time inside that file.
  - The source codec is AV1. The OpenCV pip wheel cannot decode AV1, but the
    system ffmpeg can (libdav1d). Re-encoding to H.264 fixes OpenCV.
  - 720p cuts the SAM 3.1 upload size and cost.

Each output is checked: its frame count must equal the episode length from the
metadata, so frame i of the mp4 is frame_index i in the parquet.
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ego-tactile"
OUT = ROOT / "data" / "episodes_720p"
VKEY = "observation.images.ego"


def load_episodes():
    ep = pd.read_parquet(SRC / "meta/episodes/chunk-000/file-000.parquet")
    return ep[[
        "episode_index", "length",
        f"videos/{VKEY}/chunk_index", f"videos/{VKEY}/file_index",
        f"videos/{VKEY}/from_timestamp", f"videos/{VKEY}/to_timestamp",
    ]].rename(columns=lambda c: c.split("/")[-1])


def count_frames(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def cut_one(row):
    idx, length = int(row.episode_index), int(row.length)
    out = OUT / f"episode_{idx:06d}.mp4"
    if out.exists() and count_frames(out) == length:
        return idx, length, length, "skip"
    src = SRC / f"videos/{VKEY}/chunk-{int(row.chunk_index):03d}/file-{int(row.file_index):03d}.mp4"
    # -ss after -i decodes from the file start and drops frames up to the start
    # time. Slower than input seeking but frame exact, which the tactile
    # alignment needs. -frames:v pins the count to the metadata length.
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-c:v", "libdav1d", "-i", str(src),
        "-ss", f"{row.from_timestamp:.6f}", "-frames:v", str(length),
        "-vf", "scale=-2:720", "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    got = count_frames(out)
    return idx, length, got, "ok" if got == length else "MISMATCH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eps = load_episodes()
    if args.episodes:
        eps = eps[eps.episode_index.isin(args.episodes)]

    results = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for idx, want, got, status in pool.map(cut_one, eps.itertuples()):
            print(f"episode {idx:3d}  frames {got}/{want}  {status}", flush=True)
            results.append({"episode_index": idx, "length": want, "frames": got, "status": status})

    bad = [r for r in results if r["status"] == "MISMATCH"]
    (OUT / "cut_report.json").write_text(json.dumps(results, indent=1))
    print(f"\n{len(results)} episodes, {len(bad)} frame-count mismatches")


if __name__ == "__main__":
    main()
