"""Held-out v2 labels next to tuning (labels only, no model is scored). No API call.

    python scripts/hot3d_v2_heldout_summary.py

Compares the copyable share per person and per object between the tuning
and held-out splits, with the thresholds fixed on tuning. Writes
out/hot3d_copyable/v2_split_summary.json and docs/experiments/img/exp20_v2_splits.png.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "hot3d_copyable"


def main():
    t = pd.read_csv(OUT / "tune_v2_clips_labeled.csv")
    h = pd.read_csv(OUT / "held_out_v2_clips_labeled.csv")
    ot = t.groupby("object").copyable.agg(["mean", "size"])
    oh = h.groupby("object").copyable.agg(["mean", "size"])
    both = ot.join(oh, lsuffix="_tune", rsuffix="_held", how="inner")
    both = both[(both.size_tune >= 20) & (both.size_held >= 20)]
    rho = float(spearmanr(both.mean_tune, both.mean_held)[0])
    out = {"tune": {"clips": len(t), "copyable": round(float(t.copyable.mean()), 3),
                    "by_person": t.groupby("person").copyable.mean().round(3).to_dict()},
           "held_out": {"clips": len(h), "copyable": round(float(h.copyable.mean()), 3),
                        "by_person": h.groupby("person").copyable.mean().round(3).to_dict()},
           "objects_compared": len(both), "object_share_spearman_tune_vs_held_out": round(rho, 3)}
    (OUT / "v2_split_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    people = list(out["tune"]["by_person"].items()) + list(out["held_out"]["by_person"].items())
    axes[0].bar(range(len(people)), [100 * v for _, v in people],
                color=["#1c7ed6"] * len(out["tune"]["by_person"]) + ["#e8590c"] * len(out["held_out"]["by_person"]))
    axes[0].set_xticks(range(len(people)))
    axes[0].set_xticklabels([p for p, _ in people], fontsize=8)
    axes[0].axhspan(20, 80, color="#fff3bf", zorder=0)
    axes[0].set_ylabel("copyable v2 (%)")
    axes[0].set_title("Per person: tuning (blue) and held-out (orange)")
    axes[1].scatter(100 * both.mean_tune, 100 * both.mean_held, s=10 + both.size_held / 3, color="#1c7ed6")
    for name, r in both.iterrows():
        axes[1].annotate(name, (100 * r.mean_tune, 100 * r.mean_held), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].plot([0, 100], [0, 100], color="grey", lw=1, ls="--")
    axes[1].set_xlabel("copyable share in tuning clips (%)")
    axes[1].set_ylabel("copyable share in held-out clips (%)")
    axes[1].set_title(f"Per object (at least 20 clips in each split), rank correlation {rho:.2f}")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp20_v2_splits.png", dpi=125)


if __name__ == "__main__":
    main()
