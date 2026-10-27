"""Number 1: find contact start and end per hand from SAM 3.1 masks only.

    python scripts/contact_cut.py features --split tune      # cache per-episode features
    python scripts/contact_cut.py cv                         # 5-fold CV on tune episodes
    python scripts/contact_cut.py smoke                      # the two 5 s clips from EXP-05

Input: out/sam10/episode_XXXXXX/sam_{hand,small_object}_events.jsonl.gz from
sam_episodes.py (10 fps, frame k = source frame 3k). Video only. The tactile
glove and the dataset labels are used only for training and scoring.

Per 10 fps frame and hand (left or right) we compute: how far the hand mask is
from the nearest object mask, how much of that object lies next to the hand,
and whether the object moves with the hand (after removing the motion of the
other objects, which is camera motion). A logistic regression on these
features, with a few frames of context, gives a contact probability. Smoothing
and two thresholds turn it into segments. Segment ends go back to 30 fps and
are shifted by the median delay seen on the tuning episodes (visible contact
vs glove pressure).

Scores, pooled over episodes and hands:
  boundary F1 at +-5 frames (0.167 s) and +-15 frames (0.5 s), starts and ends
  matched one to one, only to the same kind;
  contact IoU over all frames.
Ground truth: dataset labels (annotation.hand), and tactile sum > 800 as a check.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_fast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ego-tactile" / "data" / "chunk-000" / "file-000.parquet"
SAM = ROOT / "out" / "sam10"
OUT = ROOT / "out" / "contact"
STEP, W, H = 3, 1280, 720
SIDES = ("left", "right")
TACTILE_THR = 800
CONTEXT = (-6, -3, -1, 0, 1, 3, 6)
TOLS = (5, 15)
SMOKE = {  # unit name: (folder, episode, first source frame, events file suffix)
    "ep1": (ROOT / "out/meta_smoke/fps10/ep1", 1, 270, "_events.jsonl"),
    "ep53": (ROOT / "out/meta_smoke/fps10/ep53", 53, 136, "_events.jsonl"),
}
BASE = ["hand_present", "obj_present", "log_gap", "near_3px", "obj_next_to_hand",
        "obj_move_future", "obj_move_past", "obj_move_future_long", "obj_move_past_long",
        "comove", "hand_move", "obj_area_ratio", "n_objects"]


# ---------------------------------------------------------------- ground truth

def ground_truth(episodes):
    cols = ["episode_index", "frame_index", "annotation.hand",
            "observation.tactile.left", "observation.tactile.right"]
    d = pd.read_parquet(DATA, columns=cols, filters=[("episode_index", "in", list(episodes))])
    gt = {}
    for ep, e in d.groupby("episode_index"):
        e = e.sort_values("frame_index")
        lab = e["annotation.hand"].values
        gt[int(ep)] = {
            "label": {s: np.isin(lab, [s, "both"]) for s in SIDES},
            "tactile": {s: np.stack(e[f"observation.tactile.{s}"].values).sum(1) > TACTILE_THR for s in SIDES},
        }
    return gt


# ---------------------------------------------------------------- features

def _half(obj):
    """Object mask placed in a half resolution frame (360 x 640)."""
    full = np.zeros((H, W), bool)
    m = obj.mask()
    h, w = min(m.shape[0], H - obj.y0), min(m.shape[1], W - obj.x0)
    full[obj.y0:obj.y0 + h, obj.x0:obj.x0 + w] = m[:h, :w]
    return full[::2, ::2]


def hand_sides(boxes):
    """Left or right for each hand id.

    Hands cross the image centre often, so the side comes from frames that show
    two hands: the id that is more often the left one of the pair is the left
    hand. An id never seen with another hand falls back to its median x.
    """
    left, seen, xs = {}, {}, {}
    for hs in boxes.values():
        cx = {i: (b[0] + b[2]) / 2 for i, b in hs.items()}
        for i, x in cx.items():
            xs.setdefault(i, []).append(x)
            others = [v for j, v in cx.items() if j != i]
            if others:
                seen[i] = seen.get(i, 0) + 1
                left[i] = left.get(i, 0) + (x < min(others))
    return {i: ("left" if (left[i] / seen[i] > 0.5 if seen.get(i) else np.median(v) < W / 2) else "right")
            for i, v in xs.items()}


def frame_table(folder, suffix, n10):
    """Per 10 fps frame and side: hand and nearest object measurements."""
    hands = dict(sam_fast.read_frames(folder / f"sam_hand{suffix}"))
    objs = dict(sam_fast.read_frames(folder / f"sam_small_object{suffix}"))
    side_of = hand_sides({f: {o.id: o.box for o in hs} for f, hs in hands.items()})

    rec = {s: {k: np.full(n10, np.nan) for k in
               ("hx", "hy", "harea", "gap", "near", "ox", "oy", "oarea", "oid")} for s in SIDES}
    centres = {}  # object id -> {frame: (x, y, area)}
    for k in range(n10):
        obj_masks = []
        for o in objs.get(k, []):
            m = _half(o)
            a = m.sum()
            if a == 0:
                continue
            ys, xs_ = np.nonzero(m)
            c = (2 * xs_.mean(), 2 * ys.mean(), 4 * float(a))
            centres.setdefault(o.id, {})[k] = c
            obj_masks.append((o.id, m, c))
        best = {}
        for o in hands.get(k, []):
            s = side_of[o.id]
            area = (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])
            if s not in best or area > best[s][0]:
                best[s] = (area, o)
        for s, (_, o) in best.items():
            hm = _half(o)
            if not hm.any():
                continue
            r = rec[s]
            ys, xs_ = np.nonzero(hm)
            r["hx"][k], r["hy"][k], r["harea"][k] = 2 * xs_.mean(), 2 * ys.mean(), 4 * float(hm.sum())
            if not obj_masks:
                continue
            dist = cv2.distanceTransform((~hm).astype(np.uint8), cv2.DIST_L2, 3)
            cand = []
            for oid, m, c in obj_masks:
                dv = dist[m]
                cand.append((2 * float(dv.min()), -float(np.mean(dv <= 3)), oid, c))
            gap, negnear, oid, c = min(cand, key=lambda t: (t[0], t[1]))
            r["gap"][k], r["near"][k] = gap, -negnear
            r["ox"][k], r["oy"][k], r["oarea"][k], r["oid"][k] = c[0], c[1], c[2], float(oid)
    return rec, centres


def _disp(cx, k0, k1, n):
    if 0 <= k0 < n and 0 <= k1 < n and k0 in cx and k1 in cx:
        return np.array([cx[k1][0] - cx[k0][0], cx[k1][1] - cx[k0][1]])
    return None


def side_features(r, centres, n10):
    """Base features per frame for one side. Motion is in object sizes, minus camera motion."""
    feats = np.zeros((n10, len(BASE)))
    med_area = {i: np.median([v[2] for v in c.values()]) for i, c in centres.items()}
    hand_c = {k: (r["hx"][k], r["hy"][k]) for k in range(n10) if not np.isnan(r["hx"][k])}

    def cam(k0, k1, skip):
        ds = [d for i, c in centres.items() if i != skip for d in [_disp(c, k0, k1, n10)] if d is not None]
        return np.median(ds, axis=0) if ds else np.zeros(2)

    for k in range(n10):
        hp = not np.isnan(r["hx"][k])
        op = not np.isnan(r["gap"][k])
        f = dict.fromkeys(BASE, 0.0)
        f["hand_present"], f["obj_present"] = float(hp), float(op)
        f["n_objects"] = float(sum(k in c for c in centres.values()))
        if op:
            oid = str(int(r["oid"][k]))
            c = centres[oid]
            size = np.sqrt(med_area[oid])
            f["log_gap"] = np.log1p(r["gap"][k])
            f["near_3px"] = float(r["gap"][k] <= 3)
            f["obj_next_to_hand"] = r["near"][k]
            f["obj_area_ratio"] = r["oarea"][k] / med_area[oid]
            for name, k0, k1 in (("obj_move_future", k, k + 3), ("obj_move_past", k - 3, k),
                                 ("obj_move_future_long", k, k + 10), ("obj_move_past_long", k - 10, k)):
                d = _disp(c, k0, k1, n10)
                if d is not None:
                    f[name] = float(np.linalg.norm(d - cam(k0, k1, oid))) / size
            do = _disp(c, k - 2, k + 2, n10)
            dh = _disp(hand_c, k - 2, k + 2, n10)
            if do is not None and dh is not None:
                cm = cam(k - 2, k + 2, oid)
                do, dh = do - cm, dh - cm
                f["comove"] = float(np.linalg.norm(do - dh) / (np.linalg.norm(do) + np.linalg.norm(dh) + 5.0))
                f["hand_move"] = float(np.linalg.norm(dh)) / size
        else:
            f["log_gap"] = np.log1p(500.0)
        feats[k] = [f[b] for b in BASE]
    return feats


def with_context(x):
    n = len(x)
    cols = [x[np.clip(np.arange(n) + o, 0, n - 1)] for o in CONTEXT]
    return np.hstack(cols)


def unit_features(folder, suffix, n10):
    rec, centres = frame_table(folder, suffix, n10)
    return {s: side_features(rec[s], centres, n10) for s in SIDES}, rec


def cache_episode(ep):
    path = OUT / "features" / f"episode_{ep:06d}.npz"
    if path.exists():
        return path
    folder = SAM / f"episode_{ep:06d}"
    n30 = len(ground_truth([ep])[ep]["label"]["left"])
    n10 = -(-n30 // STEP)
    feats, rec = unit_features(folder, "_events.jsonl.gz", n10)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n30=n30, **{f"x_{s}": feats[s] for s in SIDES},
                        **{f"gap_{s}": rec[s]["gap"] for s in SIDES})
    return path


# ---------------------------------------------------------------- segments and scores

def segments(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def postprocess(p, hi, lo, min_len=2, merge_gap=2):
    """Smoothed probability -> contact mask at 10 fps with hysteresis."""
    q = np.convolve(p, np.ones(3) / 3, mode="same")
    on, out = False, np.zeros(len(q), bool)
    for k, v in enumerate(q):
        on = v >= hi if not on else v >= lo
        out[k] = on
    segs = segments(out)
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def to_30fps(segs10, n30, start30=0, shift_on=0, shift_off=0):
    out = []
    for a, b in segs10:
        s = int(np.clip(start30 + STEP * a + shift_on, 0, n30))
        e = int(np.clip(start30 + STEP * b + shift_off, 0, n30))
        if e > s:
            out.append((s, e))
    return out


def match(pred, ref, tol):
    """One to one matches within tol, closest first. Returns list of (pred - ref)."""
    pairs = sorted((abs(p - r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref) if abs(p - r) <= tol)
    used_p, used_r, errs = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            errs.append(pred[i] - ref[j])
    return errs


class Score:
    def __init__(self):
        self.tp = {t: 0 for t in TOLS}
        self.n_pred = self.n_ref = 0
        self.inter = self.union = 0
        self.err = {"start": [], "end": []}

    def add(self, pred_segs, ref_mask, lo=0, hi=None):
        hi = len(ref_mask) if hi is None else hi
        ref_segs = [(a, b) for a, b in segments(ref_mask[lo:hi])]
        ref_segs = [(a + lo, b + lo) for a, b in ref_segs]
        pm = np.zeros(len(ref_mask), bool)
        for a, b in pred_segs:
            pm[a:b] = True
        self.inter += int((pm[lo:hi] & ref_mask[lo:hi]).sum())
        self.union += int((pm[lo:hi] | ref_mask[lo:hi]).sum())
        for kind, idx in (("start", 0), ("end", 1)):
            p = [s[idx] for s in pred_segs]
            r = [s[idx] for s in ref_segs]
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
            out[f"{kind}_abs_error_mean"] = round(float(np.mean(np.abs(e))), 2) if e else None
        out["boundaries_pred"], out["boundaries_ref"] = self.n_pred, self.n_ref
        return out


# ---------------------------------------------------------------- training and CV

def load_cached(eps, gt):
    data = {}
    for ep in eps:
        z = np.load(OUT / "features" / f"episode_{ep:06d}.npz")
        n30 = int(z["n30"])
        n10 = z["x_left"].shape[0]
        for s in SIDES:
            y = {g: gt[ep][g][s][::STEP][:n10] for g in ("label", "tactile")}
            data[(ep, s)] = {"x": with_context(z[f"x_{s}"]), "base": z[f"x_{s}"], "y": y, "n30": n30,
                             "gt30": {g: gt[ep][g][s] for g in ("label", "tactile")}, "gap": z[f"gap_{s}"]}
    return data


def fit(units, target="label"):
    x = np.vstack([u["x"] for u in units])
    y = np.concatenate([u["y"][target] for u in units])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return model.fit(x, y)


def evaluate(units, probs, hi, lo, shifts, target="label"):
    sc = Score()
    for u, p in zip(units, probs):
        segs = to_30fps(postprocess(p, hi, lo), u["n30"], 0, *shifts)
        sc.add(segs, u["gt30"][target])
    return sc.result()


def tune_post(units, probs, target="label"):
    """Pick thresholds on out-of-fold probabilities, then the start and end delay."""
    best = None
    for hi in (0.4, 0.5, 0.6, 0.7):
        for lo in (0.2, 0.3, 0.4, 0.5):
            if lo > hi:
                continue
            r = evaluate(units, probs, hi, lo, (0, 0), target)
            key = r["boundary_f1_15"] + r["contact_iou"]
            if best is None or key > best[0]:
                best = (key, hi, lo, r)
    _, hi, lo, r = best
    shifts = (-int(round(r["start_error_median"] or 0)), -int(round(r["end_error_median"] or 0)))
    return hi, lo, shifts


def rule_probs(u):
    """Baseline without learning: in contact when the hand mask is within 3 px of an object."""
    return (np.nan_to_num(u["gap"], nan=1e9) <= 3).astype(float)


def cmd_cv(args):
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    eps = [e for e in split["tune"] if (OUT / "features" / f"episode_{e:06d}.npz").exists()]
    gt = ground_truth(eps)
    data = load_cached(eps, gt)
    folds = [eps[i::5] for i in range(5)]
    report = {"episodes": eps}
    for target in ("label", "tactile"):
        units, probs = [], []
        for fold in folds:
            train = [u for (e, s), u in data.items() if e not in fold]
            test = [(k, u) for k, u in data.items() if k[0] in fold]
            model = fit(train, target)
            for k, u in test:
                units.append(u)
                probs.append(model.predict_proba(u["x"])[:, 1])
        hi, lo, shifts = tune_post(units, probs, target)
        report[target] = {
            "thresholds": [hi, lo], "shift_start_end_frames": shifts,
            "logreg_cv": evaluate(units, probs, hi, lo, shifts, target),
            "logreg_cv_no_shift": evaluate(units, probs, hi, lo, (0, 0), target),
            "rule_gap_3px": evaluate(units, [rule_probs(u) for u in units], 0.5, 0.5, (0, 0), target),
        }
    model = fit(list(data.values()), "label")
    coef = model[-1].coef_[0].reshape(len(CONTEXT), len(BASE))
    report["coef_at_offset_0"] = dict(zip(BASE, np.round(coef[CONTEXT.index(0)], 3).tolist()))
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({t: report[t] for t in ("label", "tactile")}, indent=1))


def cmd_features(args):
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    eps = [int(x) for x in args.episodes.split(",")] if args.episodes else split[args.split]
    for ep in eps:
        folder = SAM / f"episode_{ep:06d}"
        if not all((folder / f"sam_{p}_events.jsonl.gz").exists() for p in ("hand", "small_object")):
            print(f"ep {ep}: SAM output missing, skipped")
            continue
        print(f"ep {ep}: {cache_episode(ep)}", flush=True)


def cmd_smoke(args):
    """Runs the whole path on the two 5 s clips. The numbers are a code check, not a result."""
    gt = ground_truth([v[1] for v in SMOKE.values()])
    units = []
    for name, (folder, ep, start, suffix) in SMOKE.items():
        n10 = 50
        feats, rec = unit_features(folder, suffix, n10)
        for s in SIDES:
            g30 = {g: gt[ep][g][s][start:start + 150] for g in ("label", "tactile")}
            units.append({"name": f"{name}_{s}", "x": with_context(feats[s]), "base": feats[s],
                          "y": {g: g30[g][::STEP] for g in g30}, "n30": 150, "gt30": g30, "gap": rec[s]["gap"]})
    for u in units:
        b = dict(zip(BASE, u["base"].T))
        print(u["name"], "contact frames (label, 10 fps):", int(u["y"]["label"].sum()),
              "| gap px:", np.round(u["gap"][::7], 1).tolist())
        print("   obj_move_future:", np.round(b["obj_move_future"][::7], 2).tolist(),
              "comove:", np.round(b["comove"][::7], 2).tolist())
    model = fit(units)
    probs = [model.predict_proba(u["x"])[:, 1] for u in units]
    print("train-on-test logreg:", evaluate(units, probs, 0.5, 0.3, (0, 0)))
    print("rule gap <= 3 px:", evaluate(units, [rule_probs(u) for u in units], 0.5, 0.5, (0, 0)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--split", default="tune", choices=["tune", "held_out"])
    f.add_argument("--episodes")
    sub.add_parser("cv")
    sub.add_parser("smoke")
    args = ap.parse_args()
    {"features": cmd_features, "cv": cmd_cv, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
