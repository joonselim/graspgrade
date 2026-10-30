"""Make one photo and one short clip per object of splits/softness_labels.csv, for the /label page.

    ~/graspgrade/.venv/bin/python tools/label_media.py --graspgrade ~/graspgrade --out label-media

Run in WSL. Reads only local files: out/soft (OpenGraph contact segments and
frames), out/number2 (HOT3D clips and Muse frames), data/episodes_720p and the
HOT3D preview videos. Writes label-media/<id>.jpg, label-media/<id>.mp4 and
label-media/items.json (the rows the page shows, in the order of the sheet).
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import pandas as pd

PAD = 15        # frames of context before and after the touch (30 fps)
MAX_SEC = 8     # clip length cap


def clip(src, start, frames, dst):
    start = max(0, start - PAD)
    dur = min((frames + 2 * PAD) / 30, MAX_SEC)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start / 30:.3f}", "-i", str(src),
                    "-t", f"{dur:.3f}", "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def photo(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=640:-2",
                    "-q:v", "4", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspgrade", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gg, out = a.graspgrade, a.out
    out.mkdir(parents=True, exist_ok=True)
    sheet = list(csv.DictReader(open(gg / "splits" / "softness_labels.csv", encoding="utf-8")))

    og = pd.read_csv(gg / "out" / "soft" / "opengraph_segments.csv")
    hc = pd.read_csv(gg / "out" / "number2" / "clips_final.csv")
    hc = hc[hc.role.isin(["score_train", "held_out"])]
    aria = gg / "data" / "hot3d" / "aria"

    items = []
    for i, row in enumerate(sheet):
        key = f"{row['dataset']}_{i:03d}"
        jpg, mp4 = out / f"{key}.jpg", out / f"{key}.mp4"
        if row["dataset"] == "hot3d":
            c = hc[hc.object == row["object"]].sort_values("contact_frames", ascending=False).iloc[0]
            src_photo = gg / "out" / "number2" / "muse_frames" / c.clip_id / "f2.jpg"
            src_video = next((aria / c.sequence).glob("*preview_rgb.mp4"))
            start, frames, hand = int(c.clip_start), int(c.clip_frames), c.hand
            # the clip already holds a second of reach and release
            pad_start, pad_frames = start + PAD, max(frames - 2 * PAD, 1)
        else:
            s = og[og.object == row["object"]].sort_values(["object_purity", "frames"], ascending=False).iloc[0]
            src_photo = gg / "out" / "soft" / "frames" / s["item"] / "f1.jpg"
            src_video = gg / "data" / "episodes_720p" / f"episode_{int(s.episode):06d}.mp4"
            pad_start, pad_frames, hand = int(s.start), int(s.frames), s.hand
        if not jpg.exists():
            photo(src_photo, jpg)
        if not mp4.exists():
            clip(src_video, pad_start, pad_frames, mp4)
        items.append({"id": key, "dataset": row["dataset"], "object": row["object"], "hand": hand,
                      "tune": int(row["tune"] or 0), "held_out": int(row["held_out"] or 0)})
        print(key, row["object"])

    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{len(items)} items, {size:.1f} MB in {out}")


if __name__ == "__main__":
    main()
