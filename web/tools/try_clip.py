"""Turn the frames saved by web/tools/try_record.ps1 into the short Try clip for the summary page.

Each frame is held until the next one arrived, so the clip keeps the real timing.

    python3 -I tools/try_clip.py FRAMES_DIR assets/try_en.mp4 [--skip 0.6]

Writes the mp4 (no sound, 30 fps, 1440 px wide) and a poster jpg next to it.
"""
import subprocess
import sys
from pathlib import Path


def main():
    frames, out = Path(sys.argv[1]), Path(sys.argv[2])
    skip = float(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0.0
    rows = [line.split() for line in (frames / "times.txt").read_text().splitlines() if line.strip()]
    times = [float(t) for _, t in rows]
    start = times[0] + skip
    lines = []
    for i, (n, _) in enumerate(rows):
        if times[i] < start and (i + 1 < len(rows) and times[i + 1] <= start):
            continue
        end = times[i + 1] if i + 1 < len(rows) else times[i] + 1.5  # hold the last frame
        lines += [f"file '{(frames / f'f{n}.jpg').as_posix()}'", f"duration {end - max(times[i], start):.4f}"]
    lines.append(lines[-2])  # the concat demuxer needs the last file twice to keep its duration
    listing = frames / "list.txt"
    listing.write_text("\n".join(lines) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", "fps=30,scale=1440:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-crf", "27",
                    "-preset", "slow", "-movflags", "+faststart", "-an", str(out)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-frames:v", "1", "-q:v", "4",
                    str(out.with_suffix(".jpg"))], check=True)
    print("wrote", out, f"{(times[-1] - start + 1.5):.1f} s", out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
