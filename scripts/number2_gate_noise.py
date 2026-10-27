"""How much grading error can the P7 gate take? (EXP-26, tuning only, no API call)

    python scripts/number2_gate_noise.py

The gate (number2_score.py --gate) runs on the 400 score_train clips. Here the
EXP-19 stand-in grades play the part of a perfect Muse Spark, and a share of
the grades (0% to 40%) is replaced by a random grade, 10 draws per share.
The plot shows both gate tests against the share, with the 0.65 and 0.60 lines.
Writes out/number2/gate_noise.json and docs/experiments/img/exp26_gate_noise.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from number2_score import GATE_PASS, GATE_STOP, N2, OBJECT_ITEMS, standins, two_tests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "experiments" / "img"
SHARES, DRAWS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4), 10


def main():
    d = pd.read_csv(N2 / "clips_final.csv")
    st = standins(d[d.role == "score_train"].copy()).reset_index(drop=True)
    rows = []
    for share in SHARES:
        for k in range(DRAWS if share else 1):
            g = st.copy()
            rng = np.random.default_rng(1000 * k + int(share * 100))
            for c in OBJECT_ITEMS:
                m = rng.random(len(g)) < share
                g.loc[m, c] = rng.choice([0.0, 0.5, 1.0], m.sum())
            r = two_tests(g, OBJECT_ITEMS)
            rows.append({"share": share, "draw": k, "person_out": r["leave_one_person_out_mean"],
                         "object_out": r["leave_one_object_out_pooled"], "verdict": r["verdict"]})
    t = pd.DataFrame(rows)
    summary = {str(s): {"person_out_mean": round(float(g.person_out.mean()), 3),
                        "object_out_mean": round(float(g.object_out.mean()), 3),
                        "verdicts": g.verdict.value_counts().to_dict()} for s, g in t.groupby("share")}
    (N2 / "gate_noise.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for col, c, lab in (("person_out", "#1971c2", "leave one person out, mean over people"),
                        ("object_out", "#e8590c", "leave one object out, pooled")):
        m = t.groupby("share")[col]
        x = 100 * np.array(sorted(t.share.unique()))
        ax.plot(x, m.mean().values, "-o", color=c, label=lab)
        ax.fill_between(x, m.min().values, m.max().values, color=c, alpha=0.15, lw=0)
    ax.axhline(GATE_PASS, color="#2b8a3e", ls="--", lw=1)
    ax.axhline(GATE_STOP, color="#c92a2a", ls="--", lw=1)
    ax.text(43.5, GATE_PASS + 0.003, "pass at 0.65", color="#2b8a3e", ha="right", va="bottom", fontsize=8)
    ax.text(43.5, GATE_STOP + 0.003, "stop under 0.60", color="#c92a2a", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("share of object item grades replaced by a random grade (%)")
    ax.set_ylabel("AUC")
    ax.set_title("P7 gate on the 400 score_train clips, stand-in grades with added errors\n"
                 "(line: mean of 10 draws, band: lowest to highest)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-1, 44)
    fig.tight_layout()
    fig.savefig(IMG / "exp26_gate_noise.png", dpi=120)


if __name__ == "__main__":
    main()
