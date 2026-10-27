"""Muse Spark calls for number 2: the checklist call and the checklist-free baseline.

    python scripts/muse_hot3d.py --task checklist --role held_out              # plan and cost only
    python scripts/muse_hot3d.py --task baseline --role score_train --limit 5 --send   # paid

Without --send nothing is sent. Inputs: out/number2/clips_final.csv and the
six 720 px frames per clip from hot3d_number2_inputs.py. Prompts:
scripts/prompts/number2_{checklist,baseline}.txt, with {hand} filled in.
Each answer is cached in out/number2/muse/<task>/<clip_id>.json with the raw
reply, the parsed JSON and the usage, so a rerun only sends what is missing.
Cost estimate before sending: 1,100 tokens per 720 px frame (EXP-01), plus the
prompt, plus 800 output tokens (reasoning effort low, EXP-01).
"""

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
PROMPTS = Path(__file__).resolve().parent / "prompts"
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6
TOKENS_PER_FRAME, TOKENS_OUT = 1100, 800
GRADES = {"high": 1.0, "mid": 0.5, "low": 0.0}
CHECKLIST_KEYS = ["finger_occlusion", "object_size", "object_shape", "object_thickness", "object_width",
                  "object_handle", "free_space"]


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def prompt(task, hand):
    return (PROMPTS / f"number2_{task}.txt").read_text().replace("{hand}", hand)


def parse(task, text):
    """First JSON object in the reply, checked against the expected keys. None if it does not fit."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if task == "checklist":
        if not all(str(d.get(k, "")).lower() in GRADES for k in CHECKLIST_KEYS):
            return None
        return {**{k: GRADES[str(d[k]).lower()] for k in CHECKLIST_KEYS},
                "object_name": d.get("object_name"), "reason": d.get("reason")}
    try:
        p = float(d.get("copyable_probability"))
    except (TypeError, ValueError):
        return None
    return {"copyable_probability": min(max(p, 0.0), 100.0) / 100.0, "reason": d.get("reason")}


def call(client, task, r):
    dst = N2 / "muse" / task / f"{r.clip_id}.json"
    frames = [N2 / "muse_frames" / r.clip_id / f"f{j}.jpg" for j in range(6)]
    content = [{"type": "text", "text": prompt(task, r.hand)}]
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model="muse-spark-1.3", reasoning_effort="low",
                                          messages=[{"role": "user", "content": content}])
    text = resp.choices[0].message.content
    u = resp.usage
    rec = {"clip_id": r.clip_id, "task": task, "seconds": round(time.perf_counter() - t0, 2), "reply": text,
           "parsed": parse(task, text), "usage": u.model_dump() if u else None,
           "cost_usd": round(u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT, 6) if u else None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["checklist", "baseline"])
    ap.add_argument("--role", required=True, choices=["held_out", "score_train"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--send", action="store_true", help="really call the API (costs credit)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    d = pd.read_csv(N2 / "clips_final.csv")
    d = d[d.role == args.role].sort_values("clip_id")
    if args.limit:
        d = d.sample(min(args.limit, len(d)), random_state=20261026)
    todo = [r for r in d.itertuples() if not (N2 / "muse" / args.task / f"{r.clip_id}.json").exists()]
    missing = [r.clip_id for r in todo if not (N2 / "muse_frames" / r.clip_id / "f5.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no frames yet, run hot3d_number2_inputs.py frames")
    tok_in = len(todo) * (6 * TOKENS_PER_FRAME + len(prompt(args.task, "right")) // 4)
    est = tok_in * PRICE_IN + len(todo) * TOKENS_OUT * PRICE_OUT
    print(f"{args.task} / {args.role}: {len(d)} clips, {len(todo)} to send, about {tok_in:,} input tokens, "
          f"about ${est:.2f}", flush=True)
    if not args.send or not todo:
        return
    from openai import OpenAI
    load_env()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    spent, bad, failed = 0.0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(call, client, args.task, r): r.clip_id for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                spent += rec["cost_usd"] or 0
                bad += rec["parsed"] is None
            except Exception as exc:  # keep going, a rerun sends what is missing
                failed += 1
                print(f"{futs[fut]}: FAILED {exc!r}"[:300], flush=True)
    print(f"spent ${spent:.2f}, replies that did not parse {bad}, failed {failed}")


if __name__ == "__main__":
    main()
