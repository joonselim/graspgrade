"""Send whole OpenGraph episodes to SAM 3.1 at 10 fps, one request per episode and prompt.

    python scripts/sam_episodes.py --split tune                  # plan and cost only
    python scripts/sam_episodes.py --episodes 2,5 --send         # paid
    python scripts/sam_episodes.py --split held_out --allow-held-out --send

Without --send nothing is sent: it makes the 10 fps clips and prints how many
frames would be billed. EXP-05 showed that 10 fps gives the same hand and
grasped-object tracks as 30 fps. 10 fps frame k is source frame 3k.

Output per episode in out/sam10/episode_XXXXXX/:
  clip10.mp4, sam_<prompt>_events.jsonl.gz, sam_<prompt>.json (time, usage).
Done prompts are skipped, so a failed run can simply be started again. Only a
response with status "completed" counts as done. Any other status keeps its
events as sam_<prompt>_events.<status>.jsonl.gz, and a rerun sends it again.
SAM takes up to 15,000 frames per request. Our longest episode is 1,457 frames at 10 fps.
"""

import argparse
import gzip
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "episodes_720p"
OUT = ROOT / "out" / "sam10"
STEP = 3
PROMPTS = ["hand", "small object"]
SAM_PRICE_FRAME, SAM_MIN_FRAMES = 0.20 / 1000, 50


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def n_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def make_clip(ep):
    d = OUT / f"episode_{ep:06d}"
    d.mkdir(parents=True, exist_ok=True)
    clip = d / "clip10.mp4"
    if not clip.exists():
        tmp = d / "clip10.tmp.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SRC / f"episode_{ep:06d}.mp4"),
                        "-vf", f"select='not(mod(n,{STEP}))',setpts=N/({30 // STEP}*TB)", "-r", str(30 // STEP),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(tmp)], check=True)
        tmp.rename(clip)
    return clip


def send(client, ep, clip, prompt):
    d = clip.parent
    tag = prompt.replace(" ", "_")
    t0 = time.perf_counter()
    with open(clip, "rb") as fh:
        f = client.files.create(file=fh, purpose="user_data")
    t_upload = time.perf_counter() - t0
    t1 = time.perf_counter()
    stream = client.responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": f.id},
        ]}],
        stream=True,
    )
    events, final = [], None
    for ev in stream:
        e = ev.model_dump()
        events.append(e)
        if ev.type in ("response.completed", "response.incomplete", "response.failed"):
            final = e.get("response")
    t_stream = time.perf_counter() - t1
    status = final.get("status") if final else None
    # only a completed response counts as done: anything else is kept under another name, so a rerun sends it again
    name = f"sam_{tag}_events.jsonl.gz" if status == "completed" else f"sam_{tag}_events.{status or 'no_status'}.jsonl.gz"
    tmp = d / f"sam_{tag}_events.tmp.gz"
    with gzip.open(tmp, "wt") as fh:
        fh.write("\n".join(json.dumps(e) for e in events))
    tmp.rename(d / name)
    usage = final.get("usage") if final else None
    info = {"episode": ep, "prompt": prompt, "status": status,
            "upload_seconds": round(t_upload, 2), "stream_seconds": round(t_stream, 2),
            "frames_billed": (usage or {}).get("video_frames_processed"), "usage": usage}
    (d / f"sam_{tag}.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "held_out"])
    ap.add_argument("--episodes", help="comma list, overrides --split")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--send", action="store_true", help="really call the API (costs credit)")
    ap.add_argument("--allow-held-out", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    if args.episodes:
        eps = [int(x) for x in args.episodes.split(",")]
    elif args.split:
        eps = split[args.split]
    else:
        ap.error("give --split or --episodes")
    held = sorted(set(eps) & set(split["held_out"]))
    if held and not args.allow_held_out:
        ap.error(f"held-out episodes {held} need --allow-held-out")
    prompts = args.prompts.split(",")

    jobs, billed = [], 0
    for ep in eps:
        clip = make_clip(ep)
        n = n_frames(clip)
        for p in prompts:
            if (clip.parent / f"sam_{p.replace(' ', '_')}_events.jsonl.gz").exists():
                continue
            jobs.append((ep, clip, p))
            billed += max(n, SAM_MIN_FRAMES)
    print(f"{len(eps)} episodes, {len(jobs)} requests to send, {billed} frames, "
          f"about ${billed * SAM_PRICE_FRAME:.2f}", flush=True)
    if not args.send or not jobs:
        return

    load_env()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    total, failed = 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(send, client, *j): j for j in jobs}
        for fut in as_completed(futs):
            ep, _, p = futs[fut]
            try:
                info = fut.result()
                total += info["frames_billed"] or 0
                if info["status"] != "completed":
                    failed.append((ep, p, info["status"]))
                print(f"ep {ep} {p}: {info['status']}, {info['frames_billed']} frames, "
                      f"{info['stream_seconds']} s", flush=True)
            except Exception as exc:  # keep going, a rerun picks up what is missing
                failed.append((ep, p, repr(exc)[:200]))
                print(f"ep {ep} {p}: FAILED {exc!r}"[:300], flush=True)
    print(f"billed {total} frames, ${total * SAM_PRICE_FRAME:.2f}, failed {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
