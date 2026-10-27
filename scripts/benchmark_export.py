"""Write the ground truth as a reusable benchmark (docs/MEASUREMENT_PLAN.md, "Release").

    python scripts/benchmark_export.py

benchmark/number1/ground_truth.json  OpenGraph Ego-Tactile, per episode and hand:
    contact segments from the cleaned glove (main) and the raw glove (check),
    frame ranges [start, end) at 30 fps, frame = dataset frame_index.
benchmark/number2/clips.csv          HOT3D Aria, per clip: where it is, the
    copyable v2 label, the condition that failed, and the label at a 2 mm and a
    10 mm depth limit (the main label uses 5 mm).
Only labels we made. No video, sensor or mocap values are copied.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_cut import SIDES, ground_truth, segments  # noqa: E402
from hot3d_copyable_v2 import DEPTH, NEAR, REACH  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
COPY = ROOT / "out" / "hot3d_copyable"
KEY = ["sequence", "hand", "clip_start", "object"]


def number1():
    split = json.loads((ROOT / "splits" / "opengraph.json").read_text())
    of = {e: s for s in ("tune", "held_out") for e in split[s]}
    gt = ground_truth(sorted(of))
    eps = {}
    for ep, g in sorted(gt.items()):
        eps[str(ep)] = {"split": of[ep], "n_frames": int(len(g["label"]["left"])),
                        **{name: {s: [[int(a), int(b)] for a, b in segments(g[key][s])] for s in SIDES}
                           for name, key in (("glove_clean", "tactile_clean"), ("glove_raw", "tactile"))}}
    out = {"dataset": "OpenGraph Ego-Tactile Manipulation (huggingface.co/datasets/OpenGraphLabs-Research/"
                      "ego-tactile-manipulation), CC BY 4.0",
           "license": "CC BY 4.0. Derived from OpenGraph Ego-Tactile by OpenGraph Labs.",
           "fps": 30, "frame": "dataset frame_index, 0 based. Segments are [start, end).",
           "glove_raw": "sum of the 80 taxels of the glove over 800",
           "glove_clean": "glove_raw with gaps of 3 frames or less joined, then pieces under 3 frames dropped",
           "main_ground_truth": "glove_clean", "episodes": eps}
    p = BENCH / "number1" / "ground_truth.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    n = sum(len(e["glove_clean"][s]) for e in eps.values() for s in SIDES)
    print(f"number 1: {len(eps)} episodes, {n} clean contacts -> {p}")


def conditions(g, depth):
    """Per clip: copyable, and the first condition that failed (grip, reach, collision)."""
    grip = (g["tip_sd"][:, :, 0] <= NEAR) & (g["tip_sd"][:, :, 1:] <= NEAR).any(2)
    reach = ((g["reach"] <= REACH) | ~g["human_touch"][:, None, :]).all(2)
    free = g["mesh_min"] >= -depth
    ok = (grip & reach & free).any(1)
    fail = np.where(ok, "none", np.where(~grip.any(1), "grip", np.where(~(grip & reach).any(1), "reach", "collision")))
    return ok.astype(int), fail


def number2():
    c = pd.read_csv(ROOT / "out" / "number2" / "clips_final.csv")
    parts = []
    for name in ("tune", "held_out"):
        clips = pd.read_csv(COPY / f"{name}_v2_clips.csv")
        g = dict(np.load(COPY / f"{name}_v2_geometry.npz"))
        assert len(clips) == len(g["mesh_min"])
        clips["copyable_check"], clips["failed_condition"] = conditions(g, DEPTH)
        clips["copyable_depth_2mm"], _ = conditions(g, 0.002)
        clips["copyable_depth_10mm"], _ = conditions(g, 0.010)
        parts.append(clips[KEY + ["copyable_check", "failed_condition", "copyable_depth_2mm", "copyable_depth_10mm"]])
    lab = pd.concat(parts)
    d = c.merge(lab, on=KEY, how="left")
    assert d.copyable_check.notna().all() and (d.copyable_check == d.copyable).all(), "labels do not match clips_final"
    d["split"] = np.where(d.role == "held_out", "held_out", "tune")
    cols = ["clip_id", "split", "role", "sequence", "person", "object", "hand", "clip_start", "clip_frames",
            "contact_start", "contact_frames", "two_hands", "copyable", "failed_condition",
            "copyable_depth_2mm", "copyable_depth_10mm"]
    p = BENCH / "number2" / "clips.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    d[cols].sort_values("clip_id").to_csv(p, index=False)
    print(f"number 2: {len(d)} clips, copyable {d.copyable.mean():.3f} -> {p}")
    print(d.groupby("split").failed_condition.value_counts(normalize=True).round(3).to_string())


def figure():
    """What is in the two files (EXP-28)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = json.loads((BENCH / "number1" / "ground_truth.json").read_text())["episodes"]
    c = pd.read_csv(BENCH / "number2" / "clips.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split, col in (("tune", "#1971c2"), ("held_out", "#e8590c")):
        lens = [(b - a) / 30 for e in g.values() if e["split"] == split for s in SIDES for a, b in e["glove_clean"][s]]
        axes[0].hist(lens, bins=np.logspace(-1, 2, 30), alpha=0.6, color=col,
                     label=f"{split}: {len(lens)} contacts in {sum(e['split'] == split for e in g.values())} episodes")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("contact length (s), cleaned glove")
    axes[0].set_ylabel("contacts")
    axes[0].set_title("Number 1: contact segments", fontsize=9)
    axes[0].legend(fontsize=8)
    order = ["none", "collision", "reach", "grip"]
    colors = {"none": "#2b8a3e", "collision": "#c92a2a", "reach": "#f08c00", "grip": "#868e96"}
    shares = c.groupby("split").failed_condition.value_counts(normalize=True).unstack().reindex(["tune", "held_out"])
    left = np.zeros(2)
    for k in order:
        v = shares[k].values
        axes[1].barh([0, 1], v, left=left, color=colors[k], label="copyable" if k == "none" else f"fails {k}")
        for i, x in enumerate(v):
            if x > 0.04:
                axes[1].text(left[i] + x / 2, i, f"{x:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += v
    n = c.split.value_counts()
    axes[1].set_yticks([0, 1], [f"tune ({n['tune']} clips)", f"held-out ({n['held_out']} clips)"], fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("share of clips")
    axes[1].set_title("Number 2: copyable label and the first condition that fails", fontsize=9)
    axes[1].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp28_benchmark.png", dpi=120)


if __name__ == "__main__":
    number1()
    number2()
    figure()
