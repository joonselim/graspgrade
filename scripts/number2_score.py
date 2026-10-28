"""Number 2: train the score and run the comparisons of docs/MEASUREMENT_PLAN.md.

    python scripts/number2_score.py --check     # code check on tuning only: train item_train, test score_train
    python scripts/number2_score.py --gate      # P7 gate on the 400 score_train clips (after P7a)
    python scripts/number2_score.py --gate --standin   # the same gate on EXP-19 stand-in grades (code check)
    python scripts/number2_score.py --final     # train score_train, test held_out (after all paid runs)

Inputs, per clip (out/number2/clips_final.csv):
  local checks   camera stability, action completeness (hot3d_local_checks.py)
  hand shape fit logistic regression on 10 MediaPipe hand features, trained on
                 item_train clips only, graded by its own tercile cut points
  SAM checks     visibility, contact clarity (number2_sam_checks.py, after SAM)
  Muse checklist finger occlusion and six object items (muse_hot3d.py --task checklist)
  Muse baseline  copyable probability (muse_hot3d.py --task baseline)
With --check, the Muse object items are replaced by the EXP-19 stand-in grades
and missing paid inputs are left out, so only the code path is tested.
With --final, an item with any clip missing stops the run (it is not left out).

Comparisons: our score vs Muse alone; object-only reference (copyable share
per object in every tuning clip outside the test role: all 2,385 tuning clips
in the final run, the item_train clips in check mode); Muse alone vs Muse
plus our checks.
Metrics: AUC with a 95% interval resampling sequences, AUC per person,
copyable share in the top 20% against the base rate, all clips and one-hand clips.

Unseen objects (comparison 4): for each test object, the hand shape fit
model, its cut points and the score weights are fit again without that
object, and the refit score rates the test clips of that object. Muse alone
is unchanged. The object-only reference gives every clip the training base
rate. Written into the output of --check and --final.

Checklist against learning (comparison 5): R1 Muse alone, R2 the mean of the
Muse checklist grades (equal weights, no learning), R3 a logistic regression
on the same grades. Paired AUC differences with a 95% interval resampling
sequences: R3 - R2, our score - R3, and R2 - R1, R3 - R1 when Muse alone exists.

Object rate as an add-on (comparison 6): for k = 5, 20 and all labeled tuning
clips of the object, a logistic regression on logit(our score) and
logit((copyable + 1) / (k + 2)), trained on the training clips (scores left one
person out, rates from other persons), 20 draws for k = 5 and 20. Rows: object
rate alone and combined, paired differences against both with a 95% interval
resampling sequences.

Extra rows: AUC inside each object (pairs from the same object, ties count
half, also inside object and person, and per object with 5 or more clips of
each kind), and AUC of copyable clips against the clips failing for each
reason (grip, reach, collision, from benchmark/number2/clips.csv).

Gate (docs/MEASUREMENT_PLAN.md, "Muse Spark run in two steps"): logistic
regression on the six object item grades of the 400 score_train clips,
leave one person out (mean AUC over people) and leave one object out
(pooled AUC, 95% interval resampling sequences). Pass if both are 0.65 or
more, stop if either is below 0.60, weak in between.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_hand_shape import FEATS as SHAPE, video_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
CHECKS = ROOT / "out" / "hot3d_checks"
KEY = ["sequence", "hand", "clip_start", "object"]
MUSE_ITEMS = ["finger_occlusion", "object_size", "object_shape", "object_thickness", "object_width",
              "object_handle", "free_space"]
STANDIN = {"object_size": "item_size", "object_shape": "item_shape", "object_thickness": "item_thickness",
           "object_width": "item_width", "object_handle": "item_handle", "free_space": "item_free_space"}
OBJECT_ITEMS = list(STANDIN)  # the six object items, finger occlusion is not one of them
HAND_SHAPE_ROLE = "item_train"
GATE_PASS, GATE_STOP = 0.65, 0.60
CHECKLIST_TASK = "checklist"  # the checklist version for held-out, picked by the rule in the plan (EXP-30)
BENCH_CLIPS = ROOT / "benchmark" / "number2" / "clips.csv"  # failed_condition per clip (EXP-28)
K_LABELED, RATE_DRAWS = (5, 20, None), 20  # comparison 6: labeled clips per object (None = all), draws
FAIL_REASONS = ("grip", "reach", "collision")


def terciles(train_scores):
    return np.percentile(train_scores, [33.3, 66.7])


def to_grade(x, cuts):
    return np.where(x >= cuts[1], 1.0, np.where(x >= cuts[0], 0.5, 0.0))


def hand_shape_fit(d, train_role, exclude_object=None):
    """The hand shape check: model and cut points from item_train clips only (and not from exclude_object)."""
    path = N2 / "hand_shape_video.csv"
    if path.exists():
        hs = pd.read_csv(path)
    else:
        parts = []
        for seq, g in d.groupby("sequence"):
            f = video_features(seq, g)
            parts.append(pd.DataFrame(f, columns=[f"v_{c}" for c in SHAPE]).assign(clip_id=g.clip_id.values))
        hs = pd.concat(parts)
        hs.to_csv(path, index=False)
    x = d[["clip_id", "role", "object", "copyable"]].merge(hs, on="clip_id")
    cols = [f"v_{c}" for c in SHAPE]
    is_train = (x.role == train_role) & (x.object != exclude_object)
    x[cols] = x[cols].fillna(x[is_train][cols].median())
    tr = x[is_train]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(tr[cols], tr.copyable)
    p = m.predict_proba(x[cols])[:, 1]
    cuts = terciles(p[is_train.values])
    return pd.Series(to_grade(p, cuts), index=x.clip_id.values)


def standins(d):
    """EXP-19 stand-in grades under the Muse item names."""
    st = pd.read_csv(CHECKS / "object_items.csv")[KEY + list(STANDIN.values())]
    return d.merge(st, on=KEY, how="left").rename(columns={v: k for k, v in STANDIN.items()})


def muse_items(d, task=None):
    rows = []
    for f in (N2 / "muse" / (task or CHECKLIST_TASK)).glob("*.json"):
        r = json.loads(f.read_text())
        if r["parsed"]:
            rows.append({"clip_id": r["clip_id"], **{k: r["parsed"][k] for k in MUSE_ITEMS}})
    cols = ["clip_id"] + MUSE_ITEMS
    return d.merge(pd.DataFrame(rows, columns=cols), on="clip_id", how="left")


def assemble(check):
    d = pd.read_csv(N2 / "clips_final.csv")
    local = pd.read_csv(CHECKS / "clip_checks.csv")[KEY + ["grade_camera_stability", "grade_completeness"]]
    d = d.merge(local, on=KEY, how="left")
    d["grade_hand_shape_fit"] = d.clip_id.map(hand_shape_fit(d, HAND_SHAPE_ROLE))
    items = ["grade_camera_stability", "grade_completeness", "grade_hand_shape_fit"]
    sam = N2 / "sam_checks.csv"
    if sam.exists():
        d = d.merge(pd.read_csv(sam), on="clip_id", how="left")
        items += ["grade_visibility", "grade_contact_clarity"]
    if check:
        d = standins(d)
        items += list(STANDIN)
    elif (N2 / "muse" / "checklist").exists():
        d = muse_items(d)
        items += MUSE_ITEMS
    base = N2 / "muse" / "baseline"
    if base.exists():
        rows = [json.loads(f.read_text()) for f in base.glob("*.json")]
        b = pd.DataFrame([{"clip_id": r["clip_id"], "muse_baseline": r["parsed"]["copyable_probability"]}
                          for r in rows if r["parsed"]])
        d = d.merge(b, on="clip_id", how="left")
    return d, items


def ci_by_sequence(y, s, seqs, seed=20261026):
    """95% interval of the AUC, resampling whole sequences."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx = {q: np.where(seqs == q)[0] for q in uniq}
    boots = []
    for _ in range(1000):
        i = np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def metrics(d, score, seed=20261026):
    y, s = d.copyable.values, d[score].values
    ok = ~np.isnan(s)
    y, s, seqs = y[ok], s[ok], d.sequence.values[ok]
    top = s >= np.percentile(s, 80)
    sub = d[ok]
    one = ~sub.two_hands.values.astype(bool)
    return {"clips": int(ok.sum()), "auc": round(float(roc_auc_score(y, s)), 3),
            "ci95_by_sequence": ci_by_sequence(y, s, seqs, seed),
            "auc_per_person": {p: round(float(roc_auc_score(g.copyable, g[score])), 3)
                               for p, g in sub.groupby("person") if g.copyable.nunique() == 2},
            "top20_copyable_share": round(float(y[top].mean()), 3), "base_rate": round(float(y.mean()), 3),
            "auc_one_hand_clips": round(float(roc_auc_score(y[one], s[one])), 3)}


def paired_auc(d, base, new, n=1000, seed=20261026):
    """AUC of column `new` minus column `base` on the same clips, resampling whole sequences."""
    d = d.dropna(subset=[base, new])
    y, seqs = d.copyable.values, d.sequence.values
    a, b = d[base].values, d[new].values
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx = {q: np.where(seqs == q)[0] for q in uniq}
    diffs = []
    for _ in range(n):
        i = np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"rows": f"{new} minus {base}", "clips": len(d), "sequences": len(uniq),
            "difference": round(float(roc_auc_score(y, b) - roc_auc_score(y, a)), 3),
            "ci95_by_sequence": [round(float(lo), 3), round(float(hi), 3)], "gain": bool(lo > 0)}


def checklist_vs_learning(train, test, items):
    """Comparison 5: R1 Muse alone, R2 checklist grades with equal weights, R3 the same grades learned.

    The checklist grades are the Muse items present in both sets (with --check, the
    six stand-in object items, finger occlusion has no stand-in)."""
    cols = [c for c in MUSE_ITEMS if c in items]
    if not cols:
        return None
    test = test.copy()
    test["R2_equal_weights"] = test[cols].mean(axis=1)
    test["R3_learned"], w = fit_score(train, test, cols)
    out = {"grades": cols, "R3_weights": w, "R2_equal_weights": metrics(test, "R2_equal_weights"),
           "R3_learned": metrics(test, "R3_learned"),
           "paired": [paired_auc(test, "R2_equal_weights", "R3_learned"),
                      paired_auc(test, "R3_learned", "our_score")]}
    if "muse_baseline" in test and test.muse_baseline.notna().any():
        out["R1_muse_alone"] = metrics(test, "muse_baseline")
        out["paired"] += [paired_auc(test, "muse_baseline", "R2_equal_weights"),
                          paired_auc(test, "muse_baseline", "R3_learned")]
    return out


def fit_score(train, test, cols):
    m = LogisticRegression(C=0.5, max_iter=2000).fit(train[cols], train.copyable)
    return m.predict_proba(test[cols])[:, 1], dict(zip(cols, np.round(m.coef_[0], 3).tolist()))


def reference_pool(d, test_role):
    """Labeled tuning clips for the object-only reference: every tuning clip outside the test role.

    In the final run that is all 2,385 tuning clips (item_train and score_train),
    in check mode the item_train clips, so the score_train test clips are never in it."""
    return d[(d.role != test_role) & (d.role != "held_out")]


def object_reference(d, test, test_role):
    """Comparison 2: the copyable share of each object in the reference pool."""
    pool = reference_pool(d, test_role)
    return test.object.map(pool.groupby("object").copyable.mean()).fillna(pool.copyable.mean())


def unseen_objects(d, items, train_role, test_role):
    """Comparison 4: every learned part refit without the object being scored."""
    has_base = "muse_baseline" in d
    parts = []
    for o in sorted(d[d.role == test_role].object.unique()):
        dd = d.copy()
        dd["grade_hand_shape_fit"] = dd.clip_id.map(hand_shape_fit(d, HAND_SHAPE_ROLE, exclude_object=o))
        tr = dd[(dd.role == train_role) & (dd.object != o)]
        te = dd[(dd.role == test_role) & (dd.object == o)].copy()
        te["our_score"], _ = fit_score(tr, te, items)
        te["plug_in"] = np.nan
        if has_base:
            trb, ok = tr.dropna(subset=["muse_baseline"]), te.muse_baseline.notna()
            if len(trb) and ok.any():
                te.loc[ok, "plug_in"], _ = fit_score(trb, te[ok], ["muse_baseline"] + items)
        parts.append(te)
    u = pd.concat(parts)
    u["object_reference"] = reference_pool(d, test_role).copyable.mean()  # an unseen object has no share
    out = {"note": "per test object, hand shape fit (model and cut points) and score weights refit without it",
           "our_score": metrics(u, "our_score"), "object_reference": metrics(u, "object_reference")}
    if has_base and u.plug_in.notna().any():
        out["muse_alone"] = metrics(u, "muse_baseline")
        out["muse_plus_checks"] = metrics(u.dropna(subset=["plug_in"]), "plug_in")
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof_by_person(train, cols):
    """Score of each training clip from a fit without that clip's person."""
    s = pd.Series(np.nan, index=train.index)
    for p in train.person.unique():
        te = train.person == p
        s[te] = fit_score(train[~te], train[te], cols)[0]
    return s.values


def draw_rates(clips, pool, k, rng, other_persons):
    """Object rate (copyable + 1) / (n + 2) from k labeled pool clips of each clip's object.

    k None takes every pool clip of the object. With other_persons, pool clips of the
    clip's own person are not used (training clips)."""
    by_object = {o: (g.copyable.values, g.person.values) for o, g in pool.groupby("object")}
    out = np.empty(len(clips))
    for j, (o, p) in enumerate(zip(clips.object.values, clips.person.values)):
        lab, who = by_object.get(o, (np.zeros(0), np.zeros(0)))
        if other_persons:
            lab = lab[who != p]
        if k is not None and len(lab) > k:
            lab = rng.choice(lab, k, replace=False)
        out[j] = (lab.sum() + 1) / (len(lab) + 2)
    return out


def boot_indices(seqs, n=1000, seed=20261026):
    """Clip indices of n resamples of whole sequences."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx = {q: np.where(seqs == q)[0] for q in uniq}
    return [np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))]) for _ in range(n)]


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def object_rate_add_on(train, test, pool, items, seed=20261026):
    """Comparison 6: our score with the object rate from k labeled tuning clips of the object added.

    Per k, a logistic regression on logit(our score) and logit(object rate), trained on
    the training clips with leave-one-person-out scores and rates from other persons,
    applied to the test clips. 20 draws for k = 5 and 20, one for all. Bootstrap draw b
    uses rate draw b modulo the number of draws."""
    tr_score, te_score = logit(oof_by_person(train, items)), logit(test.our_score.values)
    y = test.copyable.values
    boots = [i for i in boot_indices(test.sequence.values, seed=seed) if y[i].min() != y[i].max()]
    out = {"labeled_pool_clips": len(pool), "k0_our_score_auc": round(float(roc_auc_score(y, te_score)), 3)}
    for k in K_LABELED:
        n_draws = 1 if k is None else RATE_DRAWS
        rates, combined, weights = [], [], []
        for r in range(n_draws):
            rng = np.random.default_rng(seed + r)
            tr_rate, te_rate = draw_rates(train, pool, k, rng, True), draw_rates(test, pool, k, rng, False)
            m = LogisticRegression(C=0.5, max_iter=2000).fit(np.c_[tr_score, logit(tr_rate)], train.copyable)
            rates.append(te_rate)
            combined.append(m.predict_proba(np.c_[te_score, logit(te_rate)])[:, 1])
            weights.append(m.coef_[0])
        a_rate = np.array([roc_auc_score(y, s) for s in rates])
        a_comb = np.array([roc_auc_score(y, s) for s in combined])
        row = {"draws": n_draws, "object_rate_auc": round(float(a_rate.mean()), 3),
               "combined_auc": round(float(a_comb.mean()), 3),
               "combined_weights_mean": dict(zip(["our_score", "object_rate"], np.round(np.mean(weights, 0), 3).tolist()))}
        d_rate, d_score = [], []
        for b, i in enumerate(boots):
            c = roc_auc_score(y[i], combined[b % n_draws][i])
            d_rate.append(c - roc_auc_score(y[i], rates[b % n_draws][i]))
            d_score.append(c - roc_auc_score(y[i], te_score[i]))
        for name, point, diffs in (("combined minus object rate", a_comb - a_rate, d_rate),
                                   ("combined minus our score", a_comb - roc_auc_score(y, te_score), d_score)):
            lo, hi = interval(diffs)
            row[name] = {"difference": round(float(np.mean(point)), 3), "ci95_by_sequence": [lo, hi], "gain": lo > 0}
        out["k=all" if k is None else f"k={k}"] = row
    return out


def pair_auc(y, s, groups):
    """Share of correctly ordered pairs (one copyable, one not) inside each group, ties count half.

    The pooled share is the AUC per group weighted by its number of pairs. Also returns
    the table per group."""
    t = pd.DataFrame({"y": y, "s": s, "g": groups})
    t["r"] = t.groupby("g").s.rank()
    t["ry"] = t.r * t.y
    g = t.groupby("g").agg(pos=("y", "sum"), n=("y", "size"), ry=("ry", "sum"))
    g["neg"] = g.n - g.pos
    g["pairs"] = g.pos * g.neg
    g["u"] = g.ry - g.pos * (g.pos + 1) / 2
    return float(g.u.sum() / g.pairs.sum()), g


def extra_rows(train, test, items, seed=20261026):
    """Extra rows: AUC inside each object, and AUC for each reason a clip fails."""
    test = test.copy()
    cols = ["our_score", "object_reference"]
    grades = [c for c in MUSE_ITEMS if c in items]
    if grades:
        test["R3_learned"], _ = fit_score(train, test, grades)
        cols.append("R3_learned")
    if "muse_baseline" in test and test.muse_baseline.notna().any():
        cols.append("muse_baseline")
    fail = pd.read_csv(BENCH_CLIPS)[["clip_id", "copyable", "failed_condition"]]
    test = test.merge(fail, on="clip_id", how="left", suffixes=("", "_benchmark"))
    if (test.copyable != test.copyable_benchmark).any():
        raise SystemExit("copyable in clips_final.csv and benchmark/number2/clips.csv differ")
    y = test.copyable.values
    n1, n0 = y.sum(), len(y) - y.sum()
    pos_obj = test.groupby("object").copyable.agg(["sum", "size"])
    same = int((pos_obj["sum"] * (pos_obj["size"] - pos_obj["sum"])).sum())
    inside = {"pairs_same_object": same, "share_of_all_pairs": round(same / (n1 * n0), 3)}
    for c in cols:
        t = test.dropna(subset=[c])
        y, s, seqs = t.copyable.values, t[c].values, t.sequence.values
        obj, obj_person = t.object.values, (t.object + "|" + t.person).values
        main, table = pair_auc(y, s, obj)
        boots = [pair_auc(y[i], s[i], obj[i])[0] for i in boot_indices(seqs, seed=seed)]
        both = table[(table.pos >= 5) & (table.neg >= 5)]
        inside[c] = {"clips": len(t), "within_object": round(main, 3), "ci95_by_sequence": interval(boots),
                     "within_object_and_person": round(pair_auc(y, s, obj_person)[0], 3),
                     "per_object_5_each": {o: {"clips": int(r.n), "copyable": int(r.pos), "auc": round(r.u / r.pairs, 3)}
                                           for o, r in both.iterrows()}}
    reasons = {"clips": test.failed_condition.value_counts().to_dict()}
    for reason in FAIL_REASONS:
        sub = test[(test.copyable == 1) | (test.failed_condition == reason)]
        reasons[reason] = {}
        for c in cols:
            t = sub.dropna(subset=[c])
            y, s = t.copyable.values, t[c].values
            reasons[reason][c] = {"clips": len(t), "auc": round(float(roc_auc_score(y, s)), 3),
                                  "ci95_by_sequence": ci_by_sequence(y, s, t.sequence.values, seed)}
    return {"note": "object_reference gives one score per object, so it is 0.5 inside objects by construction",
            "inside_each_object": inside, "per_failure_reason": reasons}


def two_tests(t, cols):
    """EXP-19 tests on one clip table: leave one person out and leave one object out."""
    def fit_predict(tr, te):
        return LogisticRegression(C=0.5, max_iter=2000).fit(tr[cols], tr.copyable).predict_proba(te[cols])[:, 1]

    t = t.reset_index(drop=True)
    lopo, looo = np.zeros(len(t)), np.zeros(len(t))
    for p in t.person.unique():
        lopo[(t.person == p).values] = fit_predict(t[t.person != p], t[t.person == p])
    for o in t.object.unique():
        looo[(t.object == o).values] = fit_predict(t[t.object != o], t[t.object == o])
    per = {p: round(float(roc_auc_score(g.copyable, lopo[g.index])), 3)
           for p, g in t.groupby("person") if g.copyable.nunique() == 2}
    mean, pooled = float(np.mean(list(per.values()))), float(roc_auc_score(t.copyable, looo))
    verdict = "pass" if mean >= GATE_PASS and pooled >= GATE_PASS else (
        "stop" if mean < GATE_STOP or pooled < GATE_STOP else "weak")
    return {"clips": len(t), "leave_one_person_out_mean": round(mean, 3), "leave_one_person_out_per_person": per,
            "leave_one_object_out_pooled": round(pooled, 3),
            "leave_one_object_out_ci95": ci_by_sequence(t.copyable.values, looo, t.sequence.values),
            "verdict": verdict}


def cmd_gate(standin):
    """P7 gate on the 400 score_train clips. With standin, EXP-19 stand-in grades replace Muse (code check)."""
    d = pd.read_csv(N2 / "clips_final.csv")
    d = d[d.role == "score_train"].copy()
    st = standins(d)
    if standin:
        g, source = st, "EXP-19 stand-in grades (code check, not the gate)"
    else:
        g, source = muse_items(d), "Muse Spark checklist"
        missing = int(g[OBJECT_ITEMS].isna().any(axis=1).sum())
        if missing:
            raise SystemExit(f"{missing} of {len(g)} score_train clips have no parsed Muse checklist yet. "
                             "Run muse_hot3d.py --task checklist --role score_train (it sends only what is missing).")
    if g[OBJECT_ITEMS].isna().any(axis=None):
        raise SystemExit("some clips have no grades")
    out = {"grades": source, "rule": f"pass if both >= {GATE_PASS}, stop if either < {GATE_STOP}, weak in between",
           "copyable_share": round(float(g.copyable.mean()), 3), "gate": two_tests(g, OBJECT_ITEMS),
           "weights_all_400": dict(zip(OBJECT_ITEMS, np.round(LogisticRegression(C=0.5, max_iter=2000).fit(
               g[OBJECT_ITEMS], g.copyable).coef_[0], 3).tolist()))}
    if not standin:  # reported with the gate, not part of it
        out["standins_same_clips"] = two_tests(st, OBJECT_ITEMS)
        out["muse_equals_standin"] = {k: round(float((g[k].values == st[k].values).mean()), 3) for k in OBJECT_ITEMS}
        base = N2 / "muse" / "baseline"
        rows = [json.loads(f.read_text()) for f in base.glob("*.json")] if base.exists() else []
        b = pd.DataFrame([{"clip_id": r["clip_id"], "p": r["parsed"]["copyable_probability"]} for r in rows if r["parsed"]],
                         columns=["clip_id", "p"])
        gb = g.merge(b, on="clip_id")
        if len(gb) and gb.copyable.nunique() == 2:
            out["muse_alone"] = {"clips": len(gb), "auc": round(float(roc_auc_score(gb.copyable, gb.p)), 3),
                                 "auc_per_person": {p: round(float(roc_auc_score(x.copyable, x.p)), 3)
                                                    for p, x in gb.groupby("person") if x.copyable.nunique() == 2}}
    name = "gate_standin.json" if standin else "gate.json"
    (N2 / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def complete_items(train, test, items, final):
    """The items with a grade on every train and test clip.

    In the final run nothing is left out without a decision: a missing grade,
    or a missing Muse alone answer (a compared row), stops the run. In check
    mode, items with a missing paid input are left out, as before.
    """
    missing = {c: int(train[c].isna().sum() + test[c].isna().sum()) for c in items}
    if final:
        missing["muse_baseline"] = int(train.muse_baseline.isna().sum() + test.muse_baseline.isna().sum()) \
            if "muse_baseline" in train else len(train) + len(test)
    missing = {c: n for c, n in missing.items() if n}
    if missing and final:
        raise SystemExit(f"clips without a grade, per item: {missing}. Every item must be complete on the train "
                         "and test clips. Send what is missing (rerun the paid step) or decide on a rule first.")
    if missing:
        print(f"check mode, items left out (paid input missing on some clips): {missing}", file=sys.stderr)
    return [c for c in items if c not in missing]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--gate", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--standin", action="store_true", help="with --gate: stand-in grades instead of Muse (code check)")
    args = ap.parse_args()
    if args.gate:
        cmd_gate(args.standin)
        return
    d, items = assemble(args.check)
    train_role, test_role = ("item_train", "score_train") if args.check else ("score_train", "held_out")
    train, test = d[d.role == train_role].copy(), d[d.role == test_role].copy()
    items = complete_items(train, test, items, final=not args.check)
    out = {"mode": "check (tuning only, stand-ins)" if args.check else "final", "train": train_role,
           "test": test_role, "items": items}
    test["our_score"], out["weights"] = fit_score(train, test, items)
    out["our_score"] = metrics(test, "our_score")
    test["object_reference"] = object_reference(d, test, test_role)
    out["object_reference_clips"] = len(reference_pool(d, test_role))
    out["object_reference"] = metrics(test, "object_reference")
    if "muse_baseline" in d and test.muse_baseline.notna().any() and train.muse_baseline.notna().any():
        out["muse_alone"] = metrics(test, "muse_baseline")
        tr, te = train.dropna(subset=["muse_baseline"]), test.dropna(subset=["muse_baseline"]).copy()
        te["plug_in"], out["plug_in_weights"] = fit_score(tr, te, ["muse_baseline"] + items)
        out["muse_plus_checks"] = metrics(te, "plug_in")
    out["unseen_objects"] = unseen_objects(d, items, train_role, test_role)
    out["checklist_vs_learning"] = checklist_vs_learning(train, test, items)
    out["object_rate_add_on"] = object_rate_add_on(train, test, reference_pool(d, test_role), items)
    out["extra"] = extra_rows(train, test, items)
    name ="number2_check.json" if args.check else "number2_final.json"
    (N2 / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
