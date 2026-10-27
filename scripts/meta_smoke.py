"""Smoke test for the Meta Model API: one Muse Spark image call, one SAM 3.1 clip.

    python scripts/meta_smoke.py --muse --sam
    python scripts/meta_smoke.py --sam --sam-prompt "small object" \
        --episode 53 --start 136 --out out/meta_smoke/ep53

Default input is OpenGraph episode 1 (tuning split), frames 270 to 419. The right hand
reaches an object and touches it at frame 321 (tactile label). The image is
frame 330. Everything goes to out/meta_smoke/: the inputs, the raw responses,
and report.json with timing, usage and cost.

Prices are from dev.meta.ai/docs/pricing-rate-limits on 2026-10-27:
muse-spark-1.3 $1.25 / 1M input, $4.25 / 1M output.
sam-3.1 video $0.20 / 1,000 frames, minimum 50 frames per request.
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "episodes_720p" / "episode_000001.mp4"
OUT = ROOT / "out" / "meta_smoke"
START, N_FRAMES, IMAGE_FRAME = 270, 150, 330
FPS = 30

MUSE_PRICE_IN, MUSE_PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6
SAM_PRICE_FRAME, SAM_MIN_FRAMES = 0.20 / 1000, 50

MUSE_PROMPT = (
    "This is a first-person video frame. The person wears beige tactile gloves. "
    "Answer in JSON with keys: right_hand_touching (true/false), left_hand_touching "
    "(true/false), object (short noun), fingertips_visible (high/mid/low), reason "
    "(one sentence)."
)


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def prepare_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    clip, image = OUT / "clip.mp4", OUT / "frame.jpg"
    # -ss before -i seeks to the frame exactly when re-encoding, so the clip
    # holds source frames START .. START+N_FRAMES-1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{START / FPS:.4f}", "-i", str(SRC),
                    "-frames:v", str(N_FRAMES), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(clip)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{IMAGE_FRAME / FPS:.4f}", "-i", str(SRC),
                    "-frames:v", "1", "-q:v", "3", str(image)], check=True)
    return clip, image


def run_muse(client, image):
    b64 = base64.b64encode(image.read_bytes()).decode()
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="muse-spark-1.3",
        reasoning_effort="low",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": MUSE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    dt = time.perf_counter() - t0
    (OUT / "muse_response.json").write_text(r.model_dump_json(indent=1))
    u = r.usage
    return {
        "model": r.model,
        "seconds": round(dt, 2),
        "image_bytes": image.stat().st_size,
        "usage": u.model_dump() if u else None,
        "cost_usd": round(u.prompt_tokens * MUSE_PRICE_IN + u.completion_tokens * MUSE_PRICE_OUT, 6) if u else None,
        "content": r.choices[0].message.content,
        "finish_reason": r.choices[0].finish_reason,
    }


async def _replay(events):
    for e in events:
        yield e


def parse_sam(events):
    from meta_sam_parser import (decode_mask_to_raster, parse_responses_stream,
                                 video_segmentation_format)

    async def go():
        stream = parse_responses_stream(_replay(events), video_segmentation_format())
        async for _ in stream:
            pass
        return await stream.final_result()

    res = asyncio.run(go())
    boxes = [r for r in res.records if r.kind == "box"]
    masks = [r for r in res.records if r.kind == "mask"]
    frames = sorted({b.frame.frame_index for b in boxes if b.frame})
    per_frame = {}
    for b in boxes:
        per_frame.setdefault(b.frame.frame_index if b.frame else None, set()).add(b.object_id)
    sample = None
    if masks:
        m = masks[0]
        raster = decode_mask_to_raster(m.mask)
        sample = {"object_id": m.object_id, "frame": m.frame.frame_index if m.frame else None,
                  "encoding": m.mask.encoding, "raster_wh": [m.mask.width, m.mask.height],
                  "bounds": [m.bounds.left, m.bounds.top, m.bounds.right, m.bounds.bottom],
                  "on_pixels": int(sum(raster)), "raster_len": len(raster)}
    first_box = None
    if boxes:
        b = boxes[0]
        first_box = {"object_id": b.object_id, "frame": b.frame.frame_index if b.frame else None,
                     "ltrb": [b.left, b.top, b.right, b.bottom]}
    return {
        "outcome": res.outcome.status,
        "n_box_records": len(boxes),
        "n_mask_records": len(masks),
        "n_text_records": sum(r.kind == "text" for r in res.records),
        "frames_with_box": len(frames),
        "first_last_frame": [frames[0], frames[-1]] if frames else None,
        "object_ids": sorted({b.object_id for b in boxes}),
        "max_objects_per_frame": max((len(v) for v in per_frame.values()), default=0),
        "diagnostics": [{"code": d.code, "message": d.message} for d in res.diagnostics][:10],
        "first_box": first_box,
        "sample_mask": sample,
    }


def run_sam(client, clip, prompt):
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
    events, t_first, final = [], None, None
    for ev in stream:
        d = ev.model_dump()
        events.append(d)
        if ev.type == "response.output_text.delta" and t_first is None:
            t_first = time.perf_counter() - t1
        if ev.type in ("response.completed", "response.incomplete", "response.failed"):
            final = d.get("response")
    t_total = time.perf_counter() - t1

    tag = prompt.replace(" ", "_")
    (OUT / f"sam_{tag}_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    raw = "".join(e.get("delta", "") for e in events if e["type"] == "response.output_text.delta")
    (OUT / f"sam_{tag}_raw.txt").write_text(raw)

    types = {}
    for e in events:
        types[e["type"]] = types.get(e["type"], 0) + 1
    return {
        "prompt": prompt,
        "clip_frames": N_FRAMES,
        "clip_bytes": clip.stat().st_size,
        "upload_seconds": round(t_upload, 2),
        "first_delta_seconds": round(t_first, 2) if t_first is not None else None,
        "stream_seconds": round(t_total, 2),
        "event_types": types,
        "raw_chars": len(raw),
        "final_status": final.get("status") if final else None,
        "final_usage": final.get("usage") if final else None,
        "cost_usd_by_price_table": round(max(N_FRAMES, SAM_MIN_FRAMES) * SAM_PRICE_FRAME, 6),
        "parsed": parse_sam(events),
    }


def main():
    global SRC, OUT, START, IMAGE_FRAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse", action="store_true")
    ap.add_argument("--sam", action="store_true")
    ap.add_argument("--sam-prompt", default="hand")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--start", type=int, default=START, help="first source frame of the 5 s clip")
    ap.add_argument("--image-frame", type=int, default=IMAGE_FRAME)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    SRC =ROOT / "data" / "episodes_720p" / f"episode_{args.episode:06d}.mp4"
    OUT, START, IMAGE_FRAME = Path(args.out), args.start, args.image_frame

    load_env()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    clip, image = OUT / "clip.mp4", OUT / "frame.jpg"
    if not clip.exists():
        # made once per folder, so every prompt runs on the same file
        clip, image = prepare_inputs()
    path = OUT / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if args.muse:
        report["muse"] = run_muse(client, image)
        print("muse:", json.dumps(report["muse"], indent=1), flush=True)
    if args.sam:
        key = "sam_" + args.sam_prompt.replace(" ", "_")
        report[key] = run_sam(client, clip, args.sam_prompt)
        print("sam:", json.dumps(report[key], indent=1), flush=True)
    path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
