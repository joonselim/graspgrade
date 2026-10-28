"""Send the number 2 SAM windows (hot3d_number2_inputs.py) to SAM 3.1.

    python scripts/sam_hot3d.py --role held_out                 # plan and cost only
    python scripts/sam_hot3d.py --role score_train --limit 3 --send   # paid pilot

Without --send nothing is sent. One request per window and prompt, prompts
`hand` and `small object` (the object prompt was chosen on OpenGraph, EXP-02
and EXP-03, and should be checked on a few HOT3D windows first). Output per
window in out/number2/sam/<window_id>/, same files as sam_episodes.py.
Done prompts are skipped. Only a "completed" response counts as done
(see sam_episodes.send).
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam_episodes import PROMPTS, SAM_MIN_FRAMES, SAM_PRICE_FRAME, load_env, send  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["held_out", "score_train"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--send", action="store_true", help="really call the API (costs credit)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    w = pd.read_csv(N2 / "sam_windows.csv")
    w = w[w.role == args.role].sort_values("window_id")
    if args.limit:
        w = w.sample(min(args.limit, len(w)), random_state=20261026)
    jobs, billed = [], 0
    for r in w.itertuples():
        d = N2 / "sam" / r.window_id
        d.mkdir(parents=True, exist_ok=True)
        clip = d / "clip10.mp4"
        if not clip.exists():
            clip.symlink_to(N2 / "sam_windows" / f"{r.window_id}.mp4")
        for p in args.prompts.split(","):
            if not (d / f"sam_{p.replace(' ', '_')}_events.jsonl.gz").exists():
                jobs.append((r.window_id, clip, p))
                billed += max(r.frames_10fps, SAM_MIN_FRAMES)
    print(f"{len(w)} windows, {len(jobs)} requests to send, {billed} frames, about ${billed * SAM_PRICE_FRAME:.2f}",
          flush=True)
    if not args.send or not jobs:
        return
    from openai import OpenAI
    load_env()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    total, failed = 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(send, client, *j): j for j in jobs}
        for fut in as_completed(futs):
            wid, _, p = futs[fut]
            try:
                info = fut.result()
                total += info["frames_billed"] or 0
                failed += info["status"] != "completed"  # not done, a rerun sends it again
                print(f"{wid} {p}: {info['status']}, {info['frames_billed']} frames", flush=True)
            except Exception as exc:  # keep going, a rerun picks up what is missing
                failed += 1
                print(f"{wid} {p}: FAILED {exc!r}"[:300], flush=True)
    print(f"billed {total} frames, ${total * SAM_PRICE_FRAME:.2f}, not done {failed}")


if __name__ == "__main__":
    main()
