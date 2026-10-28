"""Soft object check (docs/MEASUREMENT_PLAN.md, "Soft object check"): one Muse Spark call per clip.

    python scripts/muse_soft.py segments                      # OpenGraph glove contact segments and their object
    python scripts/muse_soft.py frames                        # 3 frames per OpenGraph segment that may be sent
    python scripts/muse_soft.py send --set pilot              # plan and cost only
    python scripts/muse_soft.py send --set hot3d --send       # paid
    python scripts/muse_soft.py score                         # caught, held back, pass

Sets:
  pilot      10 HOT3D score_train clips and 10 OpenGraph tuning segments of
             objects labeled rigid or squeezable (seed 20261026). Not evaluated.
  hot3d      all 1,426 HOT3D held-out clips, frames f1 f2 f3 of out/number2/muse_frames
             (contact start, one third and two thirds into the contact).
  opengraph  all OpenGraph held-out segments, plus every tuning segment whose
             object is labeled soft. Frames: segment start, one third and two
             thirds in, from the 720p episode video.
OpenGraph segments are the cleaned glove contacts (the number 1 ground truth),
per hand. The object of a segment is its most common annotation.target_object.
Segments named as a list of objects or "no object" are left out.

Labels: splits/softness_labels.csv, filled by the two team members before any
call. --send stops unless every row has an agreed label (rigid, squeezable, soft).
The prompt is scripts/prompts/soft_object.txt, the text fixed in the plan.

Caching: an answer is out/soft/muse/<item>.json. A reply that does not parse
(no JSON, or softness not one of rigid, squeezable, soft, unsure) is kept in
out/soft/muse/unparsed/ and sent once more right away. If the second reply does
not parse either, the item is cached as unsure (forced_unsure), as the plan
says. A call that raises is not cached, so a rerun sends it again.
Held back ("needs a touch check") means the answer is soft or unsure.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_cut as cc  # noqa: E402
from muse_hot3d import PRICE_IN, PRICE_OUT, load_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "soft"
N2 = ROOT / "out" / "number2"
LABELS = ROOT / "splits" / "softness_labels.csv"
PROMPT = Path(__file__).resolve().parent / "prompts" / "soft_object.txt"
SEED = 20261026
SOFTNESS = ("rigid", "squeezable", "soft", "unsure")
LABEL_VALUES = ("rigid", "squeezable", "soft")
HELD_BACK = ("soft", "unsure")
TOKENS_PER_FRAME = {"hot3d": 628, "opengraph": 1100}  # measured on P7a (HOT3D 720 px), EXP-01 (OpenGraph 720p)
TOKENS_OUT = 1120  # P7a baseline mean
PASS_CAUGHT, PASS_HELD_BACK = 0.70, 0.10
MIN_SOFT_SEGMENTS, MIN_SOFT_OBJECTS = 20, 4


def prompt(hand):
    return PROMPT.read_text().strip().replace("{hand}", hand)


def parse(text):
    """{"softness", "object", "reason"} from the first JSON object in the reply, None if it does not fit."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    s = str(d.get("softness", "")).strip().lower()
    return {"softness": s, "object": d.get("object"), "reason": d.get("reason")} if s in SOFTNESS else None


# ---------------------------------------------------------------- items

def opengraph_segments():
    cols = ["episode_index", "frame_index", "annotation.target_object",
            "observation.tactile.left", "observation.tactile.right"]
    d = pd.read_parquet(cc.DATA, columns=cols)
    held = set(json.loads((ROOT / "splits" / "opengraph.json").read_text())["held_out"])
    rows = []
    for ep, e in d.groupby("episode_index"):
        e = e.sort_values("frame_index")
        obj = e["annotation.target_object"].values
        for side in cc.SIDES:
            mask = cc.clean_glove(np.stack(e[f"observation.tactile.{side}"].values).sum(1) > cc.TACTILE_THR)
            for a, b in cc.segments(mask):
                names = pd.Series(obj[a:b]).dropna().astype(str)
                names = names[names.str.len() > 0].value_counts()
                rows.append({"item": f"og_{int(ep):06d}_{side}_{a:05d}", "dataset": "opengraph",
                             "split": "held_out" if ep in held else "tune", "episode": int(ep), "hand": side,
                             "start": int(a), "frames": int(b - a), "object": names.index[0] if len(names) else "",
                             "object_purity": round(names.iloc[0] / (b - a), 3) if len(names) else 0.0})
    s = pd.DataFrame(rows)
    left_out = s.object.str.startswith("[") | s.object.isin(["", "no object"])
    return s[~left_out].reset_index(drop=True), int(left_out.sum())


def og_frame_index(r):
    return [r.start, r.start + r.frames // 3, r.start + 2 * r.frames // 3]


def frame_paths(r):
    if r.dataset == "hot3d":
        return [N2 / "muse_frames" / r.item / f"f{j}.jpg" for j in (1, 2, 3)]
    return [OUT / "frames" / r.item / f"f{j}.jpg" for j in range(3)]


def load_labels():
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    for c in ("labeler_1", "labeler_2", "agreed"):
        lab[c] = lab[c].str.strip().str.lower()
    return lab


def labels_complete(lab):
    return bool(lab.agreed.isin(LABEL_VALUES).all())


def agreed(lab, dataset):
    return lab[(lab.dataset == dataset) & lab.agreed.isin(LABEL_VALUES)].set_index("object").agreed.to_dict()


def hot3d_clips(role):
    d = pd.read_csv(N2 / "clips_final.csv")
    d = d[d.role == role].rename(columns={"clip_id": "item"})
    return d.assign(dataset="hot3d", split="held_out" if role == "held_out" else "tune")


def items(name, lab):
    """The clips or segments of one set. OpenGraph tuning segments need the labels."""
    if name == "hot3d":
        return hot3d_clips("held_out")
    og = pd.read_csv(OUT / "opengraph_segments.csv")
    og_lab = og.object.map(agreed(lab, "opengraph"))
    if name == "opengraph":
        return og[(og.split == "held_out") | ((og.split == "tune") & (og_lab == "soft"))]
    hot = hot3d_clips("score_train").sample(10, random_state=SEED)
    rigid = og[(og.split == "tune") & og_lab.isin(["rigid", "squeezable"])]
    return pd.concat([hot, rigid.sample(min(10, len(rigid)), random_state=SEED)], ignore_index=True)


# ---------------------------------------------------------------- calls

def answer_path(item):
    return OUT / "muse" / f"{item}.json"


def call(client, r, tries=2):
    """Send one item. After `tries` replies that do not parse, the item is cached as unsure."""
    content = [{"type": "text", "text": prompt(r.hand)}]
    for f in frame_paths(r):
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    cost, replies = 0.0, []
    for _ in range(tries):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(model="muse-spark-1.3", reasoning_effort="low",
                                              messages=[{"role": "user", "content": content}])
        text, u = resp.choices[0].message.content, resp.usage
        c = round(u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT, 6) if u else 0.0
        cost += c
        rec = {"item": r.item, "dataset": r.dataset, "hand": r.hand, "seconds": round(time.perf_counter() - t0, 2),
               "reply": text, "parsed": parse(text), "usage": u.model_dump() if u else None, "cost_usd": c}
        if rec["parsed"] is not None:
            break
        p = OUT / "muse" / "unparsed" / f"{r.item}_{time.time_ns()}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
        replies.append(text)
    else:  # no reply parsed: counts as unsure (plan rule), cached so it is not sent a third time
        rec = {**rec, "parsed": {"softness": "unsure", "object": None, "reason": None, "forced_unsure": True},
               "unparsed_replies": replies}
    rec["cost_usd"] = round(cost, 6)
    p = answer_path(r.item)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    return rec


def cmd_segments(args):
    s, left_out = opengraph_segments()
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "opengraph_segments.csv", index=False)
    print(f"{len(s)} segments ({left_out} left out: list of objects or no object), "
          f"{s.groupby('split').size().to_dict()}, {s.object.nunique()} object names")


def cmd_frames(args):
    """Frames of every OpenGraph segment that any set could send (held-out, and all tuning segments)."""
    from muse_contact import read_frames
    s = pd.read_csv(OUT / "opengraph_segments.csv")
    todo = [r for r in s.itertuples() if not all(p.exists() for p in frame_paths(r))]
    for ep, g in pd.DataFrame(todo).groupby("episode") if todo else []:
        got = read_frames(int(ep), sorted({i for r in g.itertuples() for i in og_frame_index(r)}))
        for r in g.itertuples():
            d = OUT / "frames" / r.item
            d.mkdir(parents=True, exist_ok=True)
            for j, i in enumerate(og_frame_index(r)):
                (d / f"f{j}.jpg").write_bytes(got[i])
    print(f"wrote frames for {len(todo)} segments, {len(s)} segments in all")


def cmd_send(args):
    lab = load_labels()
    if args.set != "hot3d" and not lab.agreed.isin(LABEL_VALUES).any():
        print("no agreed labels yet: OpenGraph tuning segments of soft objects and the pilot segments "
              "cannot be chosen, they are left out of this estimate")
    d = items(args.set, lab)
    todo = [r for r in d.itertuples() if not answer_path(r.item).exists()]
    missing = [r.item for r in todo if not all(p.exists() for p in frame_paths(r))]
    text_tok = len(prompt("right")) // 4
    tok_in = sum(3 * TOKENS_PER_FRAME[r.dataset] + text_tok for r in todo)
    est = tok_in * PRICE_IN + len(todo) * TOKENS_OUT * PRICE_OUT
    print(f"soft check / {args.set}: {len(d)} items {d.dataset.value_counts().to_dict()}, {len(todo)} to send, "
          f"about {tok_in:,} input tokens, about ${est:.2f} (a reply that does not parse is sent once more)")
    if missing:
        print(f"{len(missing)} items have no frames yet (run `frames`)")
    if not args.send or not todo:
        return
    if not labels_complete(lab):
        raise SystemExit(f"{(~lab.agreed.isin(LABEL_VALUES)).sum()} rows of {LABELS.name} have no agreed label. "
                         "The plan needs every object labeled (and committed) before any call")
    if missing:
        raise SystemExit("frames missing, nothing sent")
    from openai import OpenAI
    load_env()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    spent, forced, failed = 0.0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(call, client, r): r.item for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                spent += rec["cost_usd"]
                forced += bool(rec["parsed"].get("forced_unsure"))
            except Exception as exc:  # not cached, a rerun sends it again
                failed += 1
                print(f"{futs[fut]}: FAILED {exc!r}"[:300], flush=True)
    print(f"spent ${spent:.2f}, counted as unsure after two replies that did not parse {forced}, failed {failed}")


# ---------------------------------------------------------------- score

def answers(ids):
    out = {}
    for i in ids:
        p = answer_path(i)
        if p.exists():
            a = json.loads(p.read_text())["parsed"]
            out[i] = {"softness": a["softness"], "forced_unsure": bool(a.get("forced_unsure"))}
    return pd.DataFrame.from_dict(out, orient="index", columns=["softness", "forced_unsure"])


def share(x):
    return round(float(np.mean(x)), 3) if len(x) else None


def caught_interval(t, n=1000):
    """95% interval of the caught share, resampling soft objects (segments of one object are related)."""
    objs = t.object.unique()
    if len(objs) < 2:
        return None
    rng = np.random.default_rng(SEED)
    by = {o: t.held_back.values[t.object.values == o] for o in objs}
    boots = [np.concatenate([by[o] for o in rng.choice(objs, len(objs))]).mean() for _ in range(n)]
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def soft_metrics(hot, og, lab):
    """hot, og: one row per item with object, split, softness. Returns the plan's metrics and verdict."""
    hot = hot.assign(label=hot.object.map(agreed(lab, "hot3d")), held_back=hot.softness.isin(HELD_BACK))
    og = og.assign(label=og.object.map(agreed(lab, "opengraph")), held_back=og.softness.isin(HELD_BACK))
    soft = og[og.label == "soft"]
    out = {"caught": {"all": share(soft.held_back), "held_out": share(soft[soft.split == "held_out"].held_back),
                      "tune": share(soft[soft.split == "tune"].held_back), "segments": len(soft),
                      "objects": int(soft.object.nunique()), "ci95_by_object": caught_interval(soft),
                      "per_object": soft.groupby("object").held_back.agg(["size", "mean"]).round(3)
                      .rename(columns={"size": "segments", "mean": "caught"}).to_dict("index")},
           "hot3d_held_back": {"all": share(hot.held_back), "clips": len(hot),
                               "by_label": hot.groupby("label").held_back.mean().round(3).to_dict(),
                               "per_object": hot.groupby("object").held_back.mean().round(3).to_dict()},
           "opengraph_held_back_by_label": og.groupby("label").held_back.mean().round(3).to_dict(),
           "label_vs_answer": {d: pd.crosstab(t.label, t.softness).to_dict("index")
                               for d, t in (("hot3d", hot), ("opengraph", og)) if len(t)},
           "forced_unsure": int(pd.concat([hot, og]).forced_unsure.sum())}
    if len(soft) < MIN_SOFT_SEGMENTS or soft.object.nunique() < MIN_SOFT_OBJECTS:
        out["verdict"] = (f"too few soft objects to test ({len(soft)} segments, {soft.object.nunique()} objects, "
                          f"need {MIN_SOFT_SEGMENTS} and {MIN_SOFT_OBJECTS})")
    else:
        ok = out["caught"]["all"] >= PASS_CAUGHT and out["hot3d_held_back"]["all"] <= PASS_HELD_BACK
        out["verdict"] = "pass" if ok else "fail"
    return out


def labeler_agreement(lab):
    from sklearn.metrics import cohen_kappa_score
    both = lab[lab.labeler_1.isin(LABEL_VALUES) & lab.labeler_2.isin(LABEL_VALUES)]
    if not len(both):
        return None
    return {"objects": len(both), "same": share(both.labeler_1 == both.labeler_2),
            "cohen_kappa": round(float(cohen_kappa_score(both.labeler_1, both.labeler_2)), 3)}


def kept_auc(hot):
    """Our score AUC on the HOT3D held-out clips the check keeps, next to all clips (needs number2_score --final)."""
    from number2_score import ci_by_sequence
    from sklearn.metrics import roc_auc_score
    f = N2 / "number2_final_scores.csv"
    if not f.exists():
        return "number2_score.py --final has not been run"
    d = hot.merge(pd.read_csv(f), left_on="item", right_on="clip_id")
    res = {}
    for name, t in (("all", d), ("kept", d[~d.softness.isin(HELD_BACK)])):
        res[name] = {"clips": len(t), "auc": round(float(roc_auc_score(t.copyable, t.our_score)), 3),
                     "ci95_by_sequence": ci_by_sequence(t.copyable.values, t.our_score.values, t.sequence.values)}
    return res


def cmd_score(args):
    lab = load_labels()
    if not labels_complete(lab):
        raise SystemExit("labels are not complete, nothing to score")
    hot, og = items("hot3d", lab), items("opengraph", lab)
    hot = hot.join(answers(hot["item"]), on="item")
    og = og.join(answers(og["item"]), on="item")
    missing = int(hot.softness.isna().sum() + og.softness.isna().sum())
    out = {"missing_answers": missing, "labeler_agreement": labeler_agreement(lab)}
    if missing:
        print(f"{missing} items have no answer yet, scoring only the answered ones, no verdict", file=sys.stderr)
    out.update(soft_metrics(hot.dropna(subset=["softness"]), og.dropna(subset=["softness"]), lab))
    if missing:
        out["verdict"] = "incomplete"
    out["our_score_auc_hot3d"] = kept_auc(hot.dropna(subset=["softness"]))
    (OUT / "soft_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("segments")
    sub.add_parser("frames")
    s = sub.add_parser("send")
    s.add_argument("--set", required=True, choices=["pilot", "hot3d", "opengraph"])
    s.add_argument("--send", action="store_true", help="really call the API (costs credit)")
    s.add_argument("--workers", type=int, default=8)
    sub.add_parser("score")
    args = ap.parse_args()
    {"segments": cmd_segments, "frames": cmd_frames, "send": cmd_send, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
