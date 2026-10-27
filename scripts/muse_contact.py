"""Number 1, the Muse Spark arms (docs/MEASUREMENT_PLAN.md).

    python scripts/muse_contact.py a --episodes 0,30,53            # arm A plan and cost only
    python scripts/muse_contact.py a --split held_out --send        # paid
    python scripts/muse_contact.py c --segments out/contact/rule_segments.json   # arm C plan and cost only
    python scripts/muse_contact.py score --episodes 0,30,53         # score cached answers
    python scripts/muse_contact.py selftest --episodes 0,30,53      # oracle answers, code check and ceilings

Arm A, Muse Spark alone: frames every 6th frame (5 fps) of the 720p episode,
10 frames per call. Per frame and hand: touching or not. A sample stands for
its 6 frames. Arm C, arm B plus Muse Spark: for each start and end found by
arm B, 7 frames 5 frames apart around it (from -15 to +15); Muse Spark picks
the first (or last) touching frame, or 0 for no change, which keeps arm B.
Both arms then use the arm B rules: gaps up to 0.2 s joined, contacts under
0.2 s dropped. Without --send nothing is sent. Answers are cached under
out/number1/. Ground truth: cleaned glove (main), raw glove and labels.
`selftest` answers from the ground truth itself. That checks the code, and it
shows the best each arm could reach: arm A is limited by 5 fps, arm C by the
boundaries arm B finds.
"""

import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_cut import SIDES, Score, ground_truth, segments  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N1 = ROOT / "out" / "number1"
CUTS = ROOT / "data" / "episodes_720p"
PROMPTS = Path(__file__).resolve().parent / "prompts"
A_STEP, A_BATCH = 6, 10
C_OFFSETS = (-15, -10, -5, 0, 5, 10, 15)
JOIN, MIN_LEN = 6, 6  # 30 fps frames, the 0.2 s rules of arm B
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6
TOKENS_PER_FRAME, TOKENS_OUT = 1100, 800
TARGETS = ("tactile_clean", "tactile", "label")


def episodes(args):
    if args.episodes:
        return [int(x) for x in args.episodes.split(",")]
    return json.loads((ROOT / "splits" / "opengraph.json").read_text())[args.split]


def read_frames(ep, wanted):
    """{frame index: jpg bytes} for the wanted 30 fps frames of the 720p cut."""
    cap = cv2.VideoCapture(str(CUTS / f"episode_{ep:06d}.mp4"))
    out, i, last = {}, 0, max(wanted)
    wanted = set(wanted)
    while i <= last:
        if not cap.grab():
            break
        if i in wanted:
            _, img = cap.retrieve()
            out[i] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        i += 1
    cap.release()
    return out


def clean(mask):
    runs = []
    for a, b in segments(mask):
        if runs and a - runs[-1][1] <= JOIN:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return [(a, b) for a, b in runs if b - a >= MIN_LEN]


def ask(client, text, jpgs):
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
        for j in jpgs]
    r = client.chat.completions.create(model="muse-spark-1.3", reasoning_effort="low",
                                       messages=[{"role": "user", "content": content}])
    u = r.usage
    return r.choices[0].message.content, (u.model_dump() if u else None), (
        u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT if u else 0.0)


def first_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- arm A

def a_batches(n30):
    samples = list(range(0, n30, A_STEP))
    return [samples[i:i + A_BATCH] for i in range(0, len(samples), A_BATCH)]


def cmd_a(args):
    gt = ground_truth(episodes(args))
    jobs = []
    for ep, g in gt.items():
        for b in a_batches(len(g["label"]["left"])):
            if not (N1 / "muse_a" / f"episode_{ep:06d}" / f"{b[0]:05d}.json").exists():
                jobs.append((ep, b))
    frames = sum(len(b) for _, b in jobs)
    text_tok = len((PROMPTS / "number1_arm_a.txt").read_text()) // 4
    est = (frames * TOKENS_PER_FRAME + len(jobs) * text_tok) * PRICE_IN + len(jobs) * TOKENS_OUT * PRICE_OUT
    print(f"arm A: {len(gt)} episodes, {len(jobs)} calls, {frames} frames, about ${est:.2f}", flush=True)
    if not args.send or not jobs:
        return
    client = make_client()

    def run(ep, b):
        jpg = read_frames(ep, b)
        text = (PROMPTS / "number1_arm_a.txt").read_text().replace("{n}", str(len(b)))
        reply, usage, cost = ask(client, text, [jpg[k] for k in b])
        d = first_json(reply)
        ans = {}
        if d and isinstance(d.get("frames"), list):
            for x in d["frames"]:
                try:
                    ans[int(x["i"])] = {s: bool(x.get(s)) for s in SIDES}
                except (KeyError, TypeError, ValueError):
                    pass
        rec = {"episode": ep, "frames": b, "reply": reply, "answers": {str(b[i - 1]): v for i, v in ans.items()
                                                                     if 1 <= i <= len(b)}, "usage": usage, "cost_usd": cost}
        p = N1 / "muse_a" / f"episode_{ep:06d}" / f"{b[0]:05d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec))
        return cost

    spent = 0.0
    with ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(run, *j) for j in jobs]):
            try:
                spent += f.result()
            except Exception as exc:  # a rerun sends what is missing
                print("FAILED", repr(exc)[:200], flush=True)
    print(f"spent ${spent:.2f}")


def a_masks(ep, n30, answers):
    """answers: {sample frame: {left: bool, right: bool}} -> 30 fps mask per hand."""
    out = {}
    for s in SIDES:
        m = np.zeros(n30, bool)
        for k, v in answers.items():
            if v.get(s):
                m[int(k):int(k) + A_STEP] = True
        out[s] = m
    return out


# ---------------------------------------------------------------- arm C

def c_jobs(seg_path):
    segs = json.loads(Path(seg_path).read_text())
    jobs = []
    for ep, d in segs.items():
        n30 = d["n30"]
        for s in SIDES:
            for i, (a, b) in enumerate(d.get(s, [])):
                for kind, f in (("start", a), ("end", b - 1)):
                    frames = [int(np.clip(f + o, 0, n30 - 1)) for o in C_OFFSETS]
                    jobs.append({"episode": int(ep), "hand": s, "segment": i, "kind": kind, "frames": frames})
    return segs, jobs


def c_text(job):
    if job["kind"] == "start":
        return (PROMPTS / "number1_arm_c.txt").read_text().replace("{change}", "starts touching an object").replace(
            "{question}", "Which is the first frame where the {hand} hand touches the object?").replace("{hand}", job["hand"])
    return (PROMPTS / "number1_arm_c.txt").read_text().replace("{change}", "stops touching the object it holds").replace(
        "{question}", "Which is the last frame where the {hand} hand still touches the object?").replace("{hand}", job["hand"])


def c_key(job):
    return N1 / "muse_c" / f"episode_{job['episode']:06d}" / f"{job['hand']}_{job['segment']:03d}_{job['kind']}.json"


def cmd_c(args):
    _, jobs = c_jobs(args.segments)
    todo = [j for j in jobs if not c_key(j).exists()]
    est = len(todo) * (7 * TOKENS_PER_FRAME * PRICE_IN + TOKENS_OUT * PRICE_OUT)
    print(f"arm C: {len(jobs)} boundaries, {len(todo)} to send, about ${est:.2f}", flush=True)
    if not args.send or not todo:
        return
    client = make_client()

    def run(job):
        jpg = read_frames(job["episode"], job["frames"])
        reply, usage, cost = ask(client, c_text(job), [jpg[k] for k in job["frames"]])
        d = first_json(reply) or {}
        try:
            pick = int(d.get("frame", 0))
        except (TypeError, ValueError):
            pick = 0
        rec = {**job, "reply": reply, "pick": pick if 0 <= pick <= 7 else 0, "confidence": d.get("confidence"),
               "usage": usage, "cost_usd": cost}
        c_key(job).parent.mkdir(parents=True, exist_ok=True)
        c_key(job).write_text(json.dumps(rec))
        return cost

    spent = 0.0
    with ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in todo]):
            try:
                spent += f.result()
            except Exception as exc:  # a rerun sends what is missing
                print("FAILED", repr(exc)[:200], flush=True)
    print(f"spent ${spent:.2f}")


def c_segments(segs, picks):
    """Arm B segments with the boundaries Muse Spark picked. picks: {(ep, hand, i, kind): frame number 0..7}."""
    out = {}
    for ep, d in segs.items():
        n30 = d["n30"]
        for s in SIDES:
            res = []
            for i, (a, b) in enumerate(d.get(s, [])):
                pa, pb = picks.get((int(ep), s, i, "start"), 0), picks.get((int(ep), s, i, "end"), 0)
                fa = [int(np.clip(a + o, 0, n30 - 1)) for o in C_OFFSETS]
                fb = [int(np.clip(b - 1 + o, 0, n30 - 1)) for o in C_OFFSETS]
                na = fa[pa - 1] if pa else a
                nb = fb[pb - 1] + 1 if pb else b
                if nb > na:
                    res.append((na, nb))
            out[(int(ep), s)] = res
    return out


# ---------------------------------------------------------------- scoring

def score_arms(eps, a_answers, c_picks, segs):
    gt = ground_truth(eps)
    res = {}
    for t in TARGETS:
        sa, sb, sc = Score(), Score(), Score()
        for ep in eps:
            n30 = len(gt[ep]["label"]["left"])
            if a_answers is not None:
                masks = a_masks(ep, n30, a_answers.get(ep, {}))
            for s in SIDES:
                ref = gt[ep][t][s]
                if a_answers is not None:
                    sa.add(clean(masks[s]), ref)
                if segs is not None and str(ep) in segs:
                    sb.add([tuple(x) for x in segs[str(ep)].get(s, [])], ref)
                    sc.add(c_segments(segs, c_picks).get((ep, s), []), ref)
        res[t] = {"arm_A": sa.result() if a_answers is not None else None,
                  "arm_B": sb.result() if segs is not None else None,
                  "arm_C": sc.result() if segs is not None else None}
    return res


def load_a(eps):
    out = {}
    for ep in eps:
        for f in (N1 / "muse_a" / f"episode_{ep:06d}").glob("*.json"):
            out.setdefault(ep, {}).update(json.loads(f.read_text())["answers"])
    return out


def load_c():
    picks = {}
    for f in (N1 / "muse_c").glob("*/*.json"):
        r = json.loads(f.read_text())
        picks[(r["episode"], r["hand"], r["segment"], r["kind"])] = r["pick"]
    return picks


def cmd_score(args):
    eps = episodes(args)
    segs = json.loads(Path(args.segments).read_text()) if Path(args.segments).exists() else None
    res = score_arms(eps, load_a(eps), load_c(), segs)
    (N1 / "arms_score.json").parent.mkdir(parents=True, exist_ok=True)
    (N1 / "arms_score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({t: {a: (r and {k: r[k] for k in ("boundary_f1_5", "boundary_f1_15", "contact_iou")})
                          for a, r in v.items()} for t, v in res.items()}, indent=1))


def cmd_selftest(args):
    """Oracle answers from the main ground truth: checks the code and gives each arm its ceiling."""
    eps = episodes(args)
    gt = ground_truth(eps)
    a_ans = {ep: {str(k): {s: bool(gt[ep]["tactile_clean"][s][k]) for s in SIDES}
                  for k in range(0, len(gt[ep]["label"]["left"]), A_STEP)} for ep in eps}
    segs = json.loads(Path(args.segments).read_text())
    picks = {}
    for ep, d in segs.items():
        e = int(ep)
        if e not in gt:
            continue
        for s in SIDES:
            ref = gt[e]["tactile_clean"][s]
            starts = np.array([a for a, _ in segments(ref)])
            ends = np.array([b - 1 for _, b in segments(ref)])
            for i, (a, b) in enumerate(d.get(s, [])):
                for kind, f, true in (("start", a, starts), ("end", b - 1, ends)):
                    frames = [int(np.clip(f + o, 0, d["n30"] - 1)) for o in C_OFFSETS]
                    near = true[np.abs(true - f) <= 15] if len(true) else []
                    if len(near):
                        g = near[np.argmin(np.abs(near - f))]
                        picks[(e, s, i, kind)] = int(np.argmin([abs(x - g) for x in frames])) + 1
    res = score_arms(eps, a_ans, picks, segs)
    (N1 / "selftest.json").parent.mkdir(parents=True, exist_ok=True)
    (N1 / "selftest.json").write_text(json.dumps(res, indent=1))
    for t, v in res.items():
        print(t)
        for a, r in v.items():
            print(f"  {a}: F1@5 {r['boundary_f1_5']}  F1@15 {r['boundary_f1_15']}  IoU {r['contact_iou']}  "
                  f"boundaries {r['boundaries_pred']}/{r['boundaries_ref']}")
    plot_ceilings(res)


def plot_ceilings(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res["tactile_clean"]
    arms = [("A: Muse Spark alone,\nperfect answers at 5 fps", r["arm_A"]), ("B: SAM rule\n(as measured)", r["arm_B"]),
            ("C: B plus Muse Spark,\nperfect answers", r["arm_C"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(arms))
    for k, (key, lab, c) in enumerate((("boundary_f1_5", "boundary F1, 5 frames", "#1c7ed6"),
                                        ("boundary_f1_15", "boundary F1, 15 frames", "#74c0fc"),
                                        ("contact_iou", "contact IoU", "#2b8a3e"))):
        ax.bar(x + (k - 1) * 0.27, [a[1][key] for a in arms], 0.27, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("Best possible score per arm, cleaned glove, pilot episodes 0, 30, 53", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp21_arm_ceilings.png", dpi=125)


def make_client():
    from openai import OpenAI
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a", "c", "score", "selftest"])
    ap.add_argument("--episodes")
    ap.add_argument("--split", choices=["tune", "held_out"])
    ap.add_argument("--segments", default=str(ROOT / "out" / "contact" / "rule_segments.json"))
    ap.add_argument("--send", action="store_true", help="really call the API (costs credit)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"a": cmd_a, "c": cmd_c, "score": cmd_score, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
