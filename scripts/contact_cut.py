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
Ground truth (fixed 2026-10-27): the cleaned glove is the main one, tactile sum > 800
with dropouts up to 0.1 s joined and blips under 0.1 s dropped (clean_glove). The raw
glove and the dataset labels (annotation.hand) are reported next to it as checks.
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

GLOVE_JOIN, GLOVE_MIN = 3, 3  # frames: join pressure dropouts up to 0.1 s, drop pressure blips under 0.1 s


def clean_glove(mask):
    """Main ground truth: glove contact after removing threshold flicker (EXP-14)."""
    out = np.zeros_like(mask)
    runs = []
    for a, b in segments(mask):
        if runs and a - runs[-1][1] <= GLOVE_JOIN:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    for a, b in runs:
        if b - a >= GLOVE_MIN:
            out[a:b] = True
    return out


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
        gt[int(ep)]["tactile_clean"] = {s: clean_glove(gt[int(ep)]["tactile"][s]) for s in SIDES}
    return gt


# ---------------------------------------------------------------- features

def _half(obj, size=(W, H)):
    """Object mask placed in a half resolution frame (360 x 640 for OpenGraph)."""
    fw, fh = size
    full = np.zeros((fh, fw), bool)
    m = obj.mask()
    h, w = min(m.shape[0], fh - obj.y0), min(m.shape[1], fw - obj.x0)
    full[obj.y0:obj.y0 + h, obj.x0:obj.x0 + w] = m[:h, :w]
    return full[::2, ::2]


def hand_sides(boxes, width=W):
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
    return {i: ("left" if (left[i] / seen[i] > 0.5 if seen.get(i) else np.median(v) < width / 2) else "right")
            for i, v in xs.items()}


def frame_table(folder, suffix, n10, size=(W, H)):
    """Per 10 fps frame and side: hand and nearest object measurements. size is (width, height) in pixels."""
    hands = dict(sam_fast.read_frames(folder / f"sam_hand{suffix}"))
    objs = dict(sam_fast.read_frames(folder / f"sam_small_object{suffix}"))
    side_of = hand_sides({f: {o.id: o.box for o in hs} for f, hs in hands.items()}, size[0])

    rec = {s: {k: np.full(n10, np.nan) for k in
               ("hx", "hy", "harea", "gap", "near", "ox", "oy", "oarea", "oid")} for s in SIDES}
    centres = {}  # object id -> {frame: (x, y, area)}
    for k in range(n10):
        obj_masks = []
        for o in objs.get(k, []):
            m = _half(o, size)
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
            hm = _half(o, size)
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
                        **{f"gap_{s}": rec[s]["gap"] for s in SIDES}, **{f"oid_{s}": rec[s]["oid"] for s in SIDES})
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
            y = {g: gt[ep][g][s][::STEP][:n10] for g in ("label", "tactile", "tactile_clean")}
            data[(ep, s)] = {"x": with_context(z[f"x_{s}"]), "base": z[f"x_{s}"], "y": y, "n30": n30,
                             "gt30": {g: gt[ep][g][s] for g in ("label", "tactile", "tactile_clean")}, "gap": z[f"gap_{s}"],
                             "oid": z[f"oid_{s}"] if f"oid_{s}" in z.files else None}
    return data


def fit(units, target="tactile_clean"):
    x = np.vstack([u["x"] for u in units])
    y = np.concatenate([u["y"][target] for u in units])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return model.fit(x, y)


def evaluate(units, probs, hi, lo, shifts, target="tactile_clean"):
    sc = Score()
    for u, p in zip(units, probs):
        segs = to_30fps(postprocess(p, hi, lo), u["n30"], 0, *shifts)
        sc.add(segs, u["gt30"][target])
    return sc.result()


def tune_post(units, probs, target="tactile_clean"):
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


def plot_cv(eps, keys, units, probs, hi, lo, shifts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = ROOT / "docs" / "experiments" / "img"
    for ep in eps:
        fig, axes = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True)
        for ax, s in zip(axes, SIDES):
            i = keys.index((ep, s))
            u, p = units[i], probs[i]
            t30 = np.arange(u["n30"]) / 30
            pm = np.zeros(u["n30"], bool)
            for a, b in to_30fps(postprocess(p, hi, lo), u["n30"], 0, *shifts):
                pm[a:b] = True
            ax.fill_between(t30, 1.05, 1.25, where=u["gt30"]["label"], color="#2b8a3e", step="mid", lw=0, label="label contact")
            ax.fill_between(t30, 1.3, 1.5, where=u["gt30"]["tactile"], color="#868e96", step="mid", lw=0, label="tactile > 800")
            ax.fill_between(t30, -0.25, -0.05, where=pm, color="#e8590c", step="mid", lw=0, label="predicted contact")
            ax.plot(np.arange(len(p)) / 10, p, color="#1c7ed6", lw=0.8, label="contact probability")
            ax.set_ylim(-0.3, 1.55)
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel(f"{s} hand")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.32))
        axes[1].set_xlabel("seconds")
        fig.suptitle(f"Episode {ep}: contact from SAM masks (model trained on the other episodes)", y=0.99)
        fig.tight_layout()
        fig.savefig(img / f"exp06_ep{ep}_contact.png", dpi=120)
        plt.close(fig)


def cmd_cv(args):
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    eps = [e for e in split["tune"] if (OUT / "features" / f"episode_{e:06d}.npz").exists()]
    gt = ground_truth(eps)
    data = load_cached(eps, gt)
    folds = [eps[i::5] for i in range(5)]
    report = {"episodes": eps}
    for target in ("tactile_clean", "tactile", "label"):
        units, probs, keys = [], [], []
        for fold in folds:
            train = [u for (e, s), u in data.items() if e not in fold]
            test = [(k, u) for k, u in data.items() if k[0] in fold]
            model = fit(train, target)
            for k, u in test:
                units.append(u)
                keys.append(k)
                probs.append(model.predict_proba(u["x"])[:, 1])
        hi, lo, shifts = tune_post(units, probs, target)
        report[target] = {
            "thresholds": [hi, lo], "shift_start_end_frames": shifts,
            "logreg_cv": evaluate(units, probs, hi, lo, shifts, target),
            "logreg_cv_no_shift": evaluate(units, probs, hi, lo, (0, 0), target),
            "rule_gap_3px": evaluate(units, [rule_probs(u) for u in units], 0.5, 0.5, (0, 0), target),
        }
        if target == "tactile_clean" and args.plot:
            plot_cv([int(e) for e in args.plot.split(",")], keys, units, probs, hi, lo, shifts)
    model = fit(list(data.values()), "tactile_clean")
    coef = model[-1].coef_[0].reshape(len(CONTEXT), len(BASE))
    report["coef_at_offset_0"] = dict(zip(BASE, np.round(coef[CONTEXT.index(0)], 3).tolist()))
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({t: report[t] for t in ("tactile_clean", "tactile", "label")}, indent=1))


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
            g30 = {g: gt[ep][g][s][start:start + 150] for g in ("label", "tactile", "tactile_clean")}
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


# ---------------------------------------------------------------- arm B: SAM rule (measurement plan)

RULE_GRID = {"gap_px": (2, 4, 8, 12), "move": (0.05, 0.1, 0.2), "comove": (0.5, 0.7),
             "fill": (0, 1, 2), "split_object": (False, True),  # fill and split_object added in EXP-22
             "persist": (1, 3, 5)}  # added in EXP-24, only matters with split_object
FILL, MIN_LEN = 2, 2  # 10 fps frames: join near-runs split by up to 0.2 s, drop contacts under 0.2 s


def stable_ids(oid, persist):
    """Nearest object id per frame, where a new id counts only after it stays for `persist` frames.

    Shorter switches to another object are read as the object held before. Frames
    with no id stay unknown and do not break a run.
    """
    s = np.array(oid, float)
    if persist <= 1:
        return s
    cur, k, n = np.nan, 0, len(s)
    while k < n:
        if not np.isfinite(oid[k]):
            k += 1
            continue
        e = k
        while e < n and (oid[e] == oid[k] or not np.isfinite(oid[e])):
            e += 1
        if np.isfinite(cur) and oid[k] != cur and np.isfinite(oid[k:e]).sum() < persist:
            s[k:e] = np.where(np.isfinite(oid[k:e]), cur, np.nan)
        else:
            cur = oid[k]
        k = e
    return s


def rule_segments(base, gap, gap_px, move, comove, fill=FILL, split_object=False, oid=None, persist=1):
    """Arm B, per hand at 10 fps.

    near: the hand mask is within gap_px of an object mask.
    co-moving: the nearest object moves at least `move` object sizes over 0.3 s
    (camera motion removed) and moves with the hand (comove at most `comove`).
    A contact is a near-run that contains at least one co-moving frame, from the
    start of the run to its end. So a touch counts once the object moved with
    the hand, and it lasts while the hand stays at the object.
    With split_object, a change of nearest object starts a new contact, once the
    new object stays nearest for `persist` frames.
    """
    if split_object and oid is not None:
        oid = stable_ids(oid, persist)
    b = dict(zip(BASE, base.T))
    near = np.nan_to_num(gap, nan=1e9) <= gap_px
    obj_move = np.maximum(b["obj_move_future"], b["obj_move_past"])
    moving = near & (b["hand_present"] > 0) & (obj_move >= move) & (b["comove"] <= comove)
    runs = []
    pieces = []
    for a, e in segments(near):
        if split_object and oid is not None:
            # a new contact when the nearest object changes inside a near-run (EXP-22)
            cuts = [k for k in range(a + 1, e) if np.isfinite(oid[k]) and np.isfinite(oid[k - 1]) and oid[k] != oid[k - 1]]
            bounds = [a] + cuts + [e]
            pieces += list(zip(bounds[:-1], bounds[1:]))
        else:
            pieces.append((a, e))
    for a, e in pieces:
        same = True
        if split_object and oid is not None and runs:
            prev = oid[runs[-1][1] - 1]
            same = not (np.isfinite(oid[a]) and np.isfinite(prev)) or oid[a] == prev
        if runs and a - runs[-1][1] <= fill and same:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((a, e))
    return [(a, e) for a, e in runs if e - a >= MIN_LEN and moving[a:e].any()]


def score_rule(units, params, target, shifts=(0, 0)):
    sc = Score()
    for u in units:
        sc.add(to_30fps(rule_segments(u["base"], u["gap"], oid=u.get("oid"), **params), u["n30"], 0, *shifts), u["gt30"][target])
    return sc.result()


def pick_rule(units, target):
    """Grid search on tuning units: best F1 at 15 frames plus IoU, then the median start and end delay."""
    from itertools import product
    best = None
    for values in product(*RULE_GRID.values()):
        p = dict(zip(RULE_GRID, values))
        r = score_rule(units, p, target)
        key = r["boundary_f1_15"] + r["contact_iou"]
        if best is None or key > best[0]:
            best = (key, p, r)
    _, p, r = best
    return p, (-int(round(r["start_error_median"] or 0)), -int(round(r["end_error_median"] or 0)))


def reference(units, target, kind):
    sc = Score()
    for u in units:
        sc.add([(0, u["n30"])] if kind == "always" else [], u["gt30"][target])
    return sc.result()


def cmd_rule(args):
    """Arm B and the reference rows, leave one tuning episode out (parameters chosen on the others)."""
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    eps = [e for e in split["tune"] if (OUT / "features" / f"episode_{e:06d}.npz").exists()]
    data = load_cached(eps, ground_truth(eps))
    report = {"episodes": eps, "note": "glove_clean = tactile sum > 800 with flicker under 0.1 s removed (main). glove = raw tactile sum > 800 and label = dataset labels (checks)"}
    segments_out = {}
    for target, name in (("tactile_clean", "glove_clean"), ("tactile", "glove"), ("label", "label")):
        sc, chosen, per_ep = Score(), {}, {}
        for ep in eps:
            train = [u for (e, s), u in data.items() if e != ep]
            test = [(s, u) for (e, s), u in data.items() if e == ep]
            params, shifts = pick_rule(train, target)
            chosen[ep] = {"params": params, "shift_start_end": shifts}
            one = Score()
            for side, u in test:
                segs = to_30fps(rule_segments(u["base"], u["gap"], oid=u.get("oid"), **params), u["n30"], 0, *shifts)
                if target == "tactile_clean":  # arm B output for arm C, rule chosen on the other episodes
                    segments_out.setdefault(str(ep), {"n30": u["n30"]})[side] = [[int(a), int(b)] for a, b in segs]
                sc.add(segs, u["gt30"][target])
                one.add(segs, u["gt30"][target])
            per_ep[ep] = {k: one.result()[k] for k in ("boundary_f1_5", "boundary_f1_15", "contact_iou")}
        units = list(data.values())
        report[name] = {
            "arm_B_sam_rule_leave_one_out": sc.result(),
            "arm_B_per_episode": per_ep,
            "arm_B_chosen_on_other_episodes": chosen,
            "reference_always_in_contact": reference(units, target, "always"),
            "reference_never_in_contact": reference(units, target, "never"),
            "contact_share": round(float(np.mean(np.concatenate([u["gt30"][target] for u in units]))), 3),
        }
    (OUT / "rule_report.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "rule_segments.json").write_text(json.dumps(segments_out))
    for name in ("glove_clean", "glove", "label"):
        r = report[name]
        print(name, "contact share", r["contact_share"])
        for row in ("arm_B_sam_rule_leave_one_out", "reference_always_in_contact", "reference_never_in_contact"):
            x = r[row]
            print(f"  {row}: F1@5 {x['boundary_f1_5']}  F1@15 {x['boundary_f1_15']}  IoU {x['contact_iou']}  "
                  f"start err {x['start_error_median']}  end err {x['end_error_median']}  "
                  f"boundaries {x['boundaries_pred']}/{x['boundaries_ref']}")
        print("  per episode", r["arm_B_per_episode"])
        print("  chosen", r["arm_B_chosen_on_other_episodes"])
    if args.plot:
        plot_rule([int(e) for e in args.plot.split(",")], data, report["glove_clean"]["arm_B_chosen_on_other_episodes"],
                  args.fig)


def plot_rule(eps, data, chosen, name="exp22_sam_rule_split.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(eps), 1, figsize=(12, 1.9 * len(eps)), squeeze=False)
    for ax, ep in zip(axes[:, 0], eps):
        params, shifts = chosen[ep]["params"], chosen[ep]["shift_start_end"]
        y, ticks = 0, []
        for s in SIDES:
            u = data[(ep, s)]
            t = np.arange(u["n30"]) / 30
            pm = np.zeros(u["n30"], bool)
            for a, b in to_30fps(rule_segments(u["base"], u["gap"], oid=u.get("oid"), **params), u["n30"], 0, *shifts):
                pm[a:b] = True
            for mask, c, lab in ((u["gt30"]["tactile_clean"], "#2b8a3e", f"{s}: glove (main truth)"),
                                 (pm, "#e8590c", f"{s}: arm B (SAM rule)")):
                ax.fill_between(t, y, y + 0.8, where=mask, color=c, step="mid", lw=0)
                ticks.append((y + 0.4, lab))
                y += 1
            y += 0.4
        ax.set_yticks([p for p, _ in ticks])
        ax.set_yticklabels([n for _, n in ticks], fontsize=7)
        ax.set_xlim(0, t[-1])
        ax.set_title(f"Episode {ep}, rule chosen on the other pilot episodes: {params}", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / name, dpi=120)  # earlier figures are kept as they were


def cmd_rule_apply(args):
    """Arm B with the rule chosen on the train episodes, applied to other episodes.

    The output feeds arm C (muse_contact.py c --segments ...). With --train tune
    and --apply held_out this is the final arm B. Ground truth is used only to
    choose the rule on the train episodes, never on the applied ones.
    """
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())

    def eps_of(s):
        e = [int(x) for x in s.split(",")] if "," in s or s.isdigit() else split[s]
        return [x for x in e if (OUT / "features" / f"episode_{x:06d}.npz").exists()]

    train, apply = eps_of(args.train), eps_of(args.apply)
    data = load_cached(train, ground_truth(train))
    params, shifts = pick_rule(list(data.values()), "tactile_clean")
    out = {"rule": {"params": params, "shift_start_end": shifts, "train_episodes": train}}
    for ep in apply:
        z = np.load(OUT / "features" / f"episode_{ep:06d}.npz")
        n30 = int(z["n30"])
        out[str(ep)] = {"n30": n30}
        for s in SIDES:
            oid = z[f"oid_{s}"] if f"oid_{s}" in z.files else None
            segs = to_30fps(rule_segments(z[f"x_{s}"], z[f"gap_{s}"], oid=oid, **params), n30, 0, *shifts)
            out[str(ep)][s] = [[int(a), int(b)] for a, b in segs]
    path = Path(args.out)
    path.write_text(json.dumps(out))
    print(f"rule {params}, shift {shifts}, chosen on {len(train)} episodes; segments for {len(apply)} episodes -> {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--split", default="tune", choices=["tune", "held_out"])
    f.add_argument("--episodes")
    c = sub.add_parser("cv")
    c.add_argument("--plot", help="comma list of episodes to draw")
    r = sub.add_parser("rule")
    r.add_argument("--plot", help="comma list of episodes to draw")
    r.add_argument("--fig", default="exp22_sam_rule_split.png", help="figure name in docs/experiments/img")
    a = sub.add_parser("rule-apply")
    a.add_argument("--train", default="tune", help="split name or comma list")
    a.add_argument("--apply", default="held_out", help="split name or comma list")
    a.add_argument("--out", default=str(OUT / "rule_segments_applied.json"))
    sub.add_parser("smoke")
    args = ap.parse_args()
    {"features": cmd_features, "cv": cmd_cv, "rule": cmd_rule, "rule-apply": cmd_rule_apply,
     "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
