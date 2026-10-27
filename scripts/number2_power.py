"""Power check for comparison 5 (EXP-27, tuning only, no API call).

    python scripts/number2_power.py

Question: with a held-out sized set (3 persons), how often does the paired
sequence bootstrap find "learned weights (R3) minus equal weights (R2)" above 0?
The six EXP-19 stand-in object grades play the part of the Muse Spark grades
(finger occlusion has no stand-in). For each of the 20 ways to pick 3 of the 6
tuning persons as a pseudo held-out set: R3 is trained on 400 clips drawn from
the other 3 persons, R2 needs no training, both are scored on all tuning clips
of the 3 chosen persons. Done with the grades as they are and with 15% of
grades replaced at random (about the P1 agreement, EXP-23), 3 draws each.
Writes out/number2/power.json and docs/experiments/img/exp27_power.png.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from number2_score import N2, OBJECT_ITEMS, fit_score, paired_auc, standins  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "experiments" / "img"
N_TRAIN, SEED = 400, 20261026
NOISE = {"stand-in grades": (0.0, 1), "15% replaced": (0.15, 3)}


def noisy(t, share, rng):
    t = t.copy()
    for c in OBJECT_ITEMS:
        m = rng.random(len(t)) < share
        t.loc[m, c] = rng.choice([0.0, 0.5, 1.0], m.sum())
    return t


def main():
    d = pd.read_csv(N2 / "clips_final.csv")
    t = standins(d[d.role != "held_out"].copy()).dropna(subset=OBJECT_ITEMS).reset_index(drop=True)
    persons = sorted(t.person.unique())
    rows = []
    for name, (share, draws) in NOISE.items():
        for k in range(draws):
            g = noisy(t, share, np.random.default_rng(SEED + k)) if share else t
            for test_p in combinations(persons, 3):
                te = g[g.person.isin(test_p)].copy()
                pool = g[~g.person.isin(test_p)]
                tr = pool.sample(min(N_TRAIN, len(pool)), random_state=SEED + k)
                te["R2"] = te[OBJECT_ITEMS].mean(axis=1)
                te["R3"], _ = fit_score(tr, te, OBJECT_ITEMS)
                p = paired_auc(te, "R2", "R3")
                rows.append({"grades": name, "draw": k, "test_persons": ",".join(test_p), "clips": p["clips"],
                             "sequences": p["sequences"], "difference": p["difference"],
                             "ci_low": p["ci95_by_sequence"][0], "ci_high": p["ci95_by_sequence"][1],
                             "gain": p["gain"]})
    r = pd.DataFrame(rows)
    out = {name: {"sets": int(len(g)), "gain_found": round(float(g.gain.mean()), 3),
                  "difference_median": round(float(g.difference.median()), 3),
                  "difference_range": [round(float(g.difference.min()), 3), round(float(g.difference.max()), 3)],
                  "clips_median": int(g.clips.median()), "sequences_median": int(g.sequences.median())}
           for name, g in r.groupby("grades")}
    out["per_set"] = r.to_dict("records")
    (N2 / "power.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_set"}, indent=1))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, (name, g) in enumerate(r.groupby("grades", sort=False)):
        g = g.sort_values("difference").reset_index(drop=True)
        x = np.arange(len(g)) + 0.3 * j
        col = "#1971c2" if j == 0 else "#e8590c"
        ax.errorbar(x, g.difference, yerr=[g.difference - g.ci_low, g.ci_high - g.difference], fmt="o", ms=3,
                    color=col, elinewidth=0.8, label=f"{name}: interval above 0 in {g.gain.mean():.0%} of sets")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("pseudo held-out sets (3 of 6 tuning persons), sorted by the difference")
    ax.set_ylabel("AUC, learned minus equal weights")
    ax.set_title("Can a 3-person held-out set show that learning the weights helps?\n"
                 "Point: difference. Bar: 95% interval resampling sequences", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "exp27_power.png", dpi=120)


if __name__ == "__main__":
    main()
