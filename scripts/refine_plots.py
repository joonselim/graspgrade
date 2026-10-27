"""Figures for the local data checks (docs/experiments/img/).

    python scripts/refine_plots.py

Left: agreement between the dataset contact labels and a raw tactile
threshold, for episodes 0 to 62 and for episodes 63 to 71.
Right: head IMU gyro p95 per episode, from out/refine_check/per_episode.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ego-tactile" / "data" / "chunk-000" / "file-000.parquet"
IMG = ROOT / "docs" / "experiments" / "img"
THRESHOLDS = [0, 50, 100, 200, 400, 800, 1200, 1600]


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DATA, columns=["episode_index", "annotation.hand",
                                       "observation.tactile.left", "observation.tactile.right"])
    late = d.episode_index.isin(range(63, 72)).values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for h, style in (("left", "-"), ("right", "--")):
        s = np.stack(d[f"observation.tactile.{h}"].values).sum(1)
        lab = d["annotation.hand"].isin([h, "both"]).values
        for group, mask, colour in (("ep 0-62", ~late, "#2a6fdb"), ("ep 63-71", late, "#d9534f")):
            agree = [100 * float(((s > t) == lab)[mask].mean()) for t in THRESHOLDS]
            ax[0].plot(THRESHOLDS, agree, style, color=colour, marker="o", label=f"{h} hand, {group}")
    ax[0].set_xlabel("tactile threshold (sum of 80 taxels)")
    ax[0].set_ylabel("frames where label and tactile agree (%)")
    ax[0].set_title("Contact labels vs raw tactile")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    p = pd.read_csv(ROOT / "out" / "refine_check" / "per_episode.csv").sort_values("episode")
    colours = ["#d9534f" if 48 <= e <= 62 else "#2a6fdb" for e in p.episode]
    ax[1].bar(p.episode, p.head_gyro_p95, color=colours)
    ax[1].set_xlabel("episode")
    ax[1].set_ylabel("head gyro p95 (rad/s)")
    ax[1].set_title("Camera motion per episode (red: spray bottle tasks)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "exp04_tactile_and_head_motion.png", dpi=110)


if __name__ == "__main__":
    main()
