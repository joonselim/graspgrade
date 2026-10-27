"""Score contact segments against the glove ground truth (GraspGrade number 1 benchmark).

    python score_number1.py predictions.json                    # held-out episodes, cleaned glove
    python score_number1.py predictions.json --split tune
    python score_number1.py predictions.json --truth glove_raw --episodes 0,30,53

predictions.json: {"<episode>": {"left": [[start, end], ...], "right": [...]}, ...}
with frame ranges [start, end) at 30 fps in dataset frame_index. An episode or
hand that is missing counts as "no contact".

Metrics, pooled over episodes and hands:
  boundary F1 at 5 and 15 frames: starts and ends are matched one to one,
      each only to its own kind, closest pairs first, within the tolerance,
  contact IoU over all frames,
  median start and end error (prediction minus truth) of pairs within 15 frames,
and the same per hand. Needs numpy only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TOLS = (5, 15)
SIDES = ("left", "right")


def match(pred, ref, tol):
    """One to one matches within tol, closest first. Returns a list of pred - ref."""
    pairs = sorted((abs(p - r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref) if abs(p - r) <= tol)
    used_p, used_r, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            out.append(pred[i] - ref[j])
    return out


class Score:
    def __init__(self):
        self.tp = {t: 0 for t in TOLS}
        self.n_pred = self.n_ref = self.inter = self.union = 0
        self.err = {"start": [], "end": []}

    def add(self, pred, ref, n):
        pm, rm = np.zeros(n, bool), np.zeros(n, bool)
        for a, b in pred:
            pm[max(0, a):min(n, b)] = True
        for a, b in ref:
            rm[a:b] = True
        self.inter += int((pm & rm).sum())
        self.union += int((pm | rm).sum())
        for k, kind in enumerate(("start", "end")):
            p, r = [s[k] for s in pred], [s[k] for s in ref]
            self.n_pred += len(p)
            self.n_ref += len(r)
            for t in TOLS:
                self.tp[t] += len(match(p, r, t))
            self.err[kind] += match(p, r, max(TOLS))

    def result(self):
        out = {}
        for t in TOLS:
            prec = self.tp[t] / self.n_pred if self.n_pred else 0.0
            rec = self.tp[t] / self.n_ref if self.n_ref else 0.0
            out[f"boundary_f1_{t}"] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
            out[f"precision_{t}"], out[f"recall_{t}"] = round(prec, 3), round(rec, 3)
        out["contact_iou"] = round(self.inter / self.union, 3) if self.union else 0.0
        for kind in ("start", "end"):
            e = self.err[kind]
            out[f"{kind}_error_median"] = float(np.median(e)) if e else None
        out["boundaries_pred"], out["boundaries_true"] = self.n_pred, self.n_ref
        return out


def score(truth, pred, episodes, target="glove_clean"):
    total, per_hand = Score(), {s: Score() for s in SIDES}
    for ep in episodes:
        e = truth["episodes"][ep]
        for s in SIDES:
            p = [tuple(x) for x in pred.get(ep, {}).get(s, [])]
            for sc in (total, per_hand[s]):
                sc.add(p, [tuple(x) for x in e[target][s]], e["n_frames"])
    return {"episodes": len(episodes), "truth": target, "pooled": total.result(),
            **{f"{s}_hand": per_hand[s].result() for s in SIDES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--truth", default="glove_clean", choices=["glove_clean", "glove_raw"])
    ap.add_argument("--split", default="held_out", choices=["held_out", "tune", "all"])
    ap.add_argument("--episodes", help="comma list, overrides --split")
    ap.add_argument("--ground-truth", default=str(Path(__file__).resolve().parent / "number1" / "ground_truth.json"))
    args = ap.parse_args()
    truth = json.loads(Path(args.ground_truth).read_text())
    pred = {k: v for k, v in json.loads(Path(args.predictions).read_text()).items() if isinstance(v, dict)}
    if args.episodes:
        eps = args.episodes.split(",")
    else:
        eps = [k for k, e in truth["episodes"].items() if args.split == "all" or e["split"] == args.split]
    missing = [e for e in eps if e not in pred]
    if missing:
        print(f"note: {len(missing)} episodes have no prediction and count as no contact")
    print(json.dumps(score(truth, pred, eps, args.truth), indent=1))


if __name__ == "__main__":
    main()
