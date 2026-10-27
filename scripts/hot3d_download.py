"""Download selected HOT3D files from the signed-URL JSONs, with sha1 checks.

    python scripts/hot3d_download.py --assets
    python scripts/hot3d_download.py --sequences P0001_10a27bf7
    python scripts/hot3d_download.py --all-sequences --jobs 4

By default only the groups GraspGrade needs are fetched. Full Aria is 525 GB,
of which 329 GB is main_vrs. The default set is about 20 GB for all 198 Aria
sequences:
  video_main_rgb  RGB preview mp4 (the SAM / MediaPipe input)
  ground_truth    camera models, headset and object poses, visibility masks
  hand_data       MANO and UmeTrack hand poses

The URL JSONs come from projectaria.com and expire 14 days after issue.
They are gitignored under data/. Never print the URLs, they are signed.
"""

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
HOT3D = ROOT / "data" / "hot3d"
DEFAULT_GROUPS = ["video_main_rgb", "ground_truth", "hand_data"]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry, dest_dir, group):
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / entry["filename"]
    if out.exists() and out.stat().st_size == entry["file_size_bytes"] and sha1(out) == entry["sha1sum"]:
        return out, "skip"
    tmp = out.with_suffix(out.suffix + ".part")
    with urlopen(entry["download_url"], timeout=60) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
    if sha1(tmp) != entry["sha1sum"]:
        tmp.unlink()
        return out, "SHA1 MISMATCH"
    tmp.rename(out)
    if out.suffix == ".zip":
        # each zip unpacks into a folder named after its group
        with zipfile.ZipFile(out) as z:
            z.extractall(dest_dir / group)
    return out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Aria", choices=["Aria", "Quest"])
    ap.add_argument("--sequences", nargs="*", default=[])
    ap.add_argument("--all-sequences", action="store_true")
    ap.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    ap.add_argument("--assets", action="store_true", help="object 3D models (.glb), 159 MB")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    if args.assets:
        a = json.loads((HOT3D / "Hot3DAssets_download_urls.json").read_text())["sequences"]
        jobs.append((a["assets"]["assets"], HOT3D, "assets"))

    seqs = json.loads((HOT3D / f"Hot3D{args.dataset}_download_urls.json").read_text())["sequences"]
    names = sorted(seqs) if args.all_sequences else args.sequences
    for name in names:
        for g in args.groups:
            if g in seqs[name]:
                jobs.append((seqs[name][g], HOT3D / args.dataset.lower() / name, g))

    total = sum(e["file_size_bytes"] for e, _, _ in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB", flush=True)
    bad = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for out, status in pool.map(lambda j: fetch(*j), jobs):
            bad += status != "ok" and status != "skip"
            print(f"  {status:14s} {out.relative_to(ROOT)}", flush=True)
    print(f"done, {bad} failures")


if __name__ == "__main__":
    main()
