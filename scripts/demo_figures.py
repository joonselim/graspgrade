"""Demo video figures at 1920x1080 with large text, from the final result files.

Reads out/number1/final/number1_final_report.json (EXP-37),
out/number2/final/number2_final.json (EXP-40) and out/soft/soft_check.json (EXP-42).
Also draws the three text cards for scenes 1, 2 and 6 of docs/DEMO_VIDEO.md.
Writes out/demo/*.png. No API call.

Usage:
    .venv/bin/python scripts/demo_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "demo"

HI = "#1c64d6"      # highlighted row
HI2 = "#e8590c"     # second highlighted row
BASE = "#b9b9b2"    # other rows
TEXT = "#1d1d1b"
MUTED = "#6b6b66"

plt.rcParams.update({
    "font.size": 26, "axes.titlesize": 30, "axes.labelsize": 24,
    "xtick.labelsize": 22, "ytick.labelsize": 26,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(path):
    return json.loads((ROOT / path).read_text())


def canvas():
    return plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")


def save(fig, name, transparent=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if transparent:
        fig.savefig(OUT / name, dpi=100, transparent=True)
    else:
        fig.savefig(OUT / name, dpi=100, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def hbars(ax, labels, values, colors, ci=None, fmt="{:.3f}", xmax=1.0):
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    if ci:
        for yi, (lo, hi) in zip(y, ci):
            if lo is not None:
                ax.plot([lo, hi], [yi, yi], color=TEXT, lw=2.5)
    for yi, v, (lo, hi) in zip(y, values, ci or [(None, None)] * len(values)):
        x = (hi if hi is not None else v) + 0.012 * xmax
        ax.text(x, yi, fmt.format(v), va="center", fontsize=26, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, xmax)


def number1():
    rows = load("out/number1/final/number1_final_report.json")["pooled"]["tactile_clean"]
    order = [("arm_A", "Muse Spark alone (A)"), ("arm_A_shift", "A + timing shift"),
             ("arm_B", "SAM rule (B)"), ("arm_C", "SAM rule + Muse (C)"),
             ("arm_C_shift", "C + timing shift"), ("always_in_contact", "Always in contact")]
    colors = [HI if k == "arm_A_shift" else HI2 if k == "arm_B" else BASE for k, _ in order]
    fig = canvas()
    fig.suptitle("When does a touch start and end?  22 held-out episodes, 574 glove boundaries",
                 fontsize=32, x=0.04, ha="left", y=0.95)
    ax1 = fig.add_axes([0.26, 0.12, 0.30, 0.70])
    hbars(ax1, [l for _, l in order], [rows[k]["boundary_f1_5"] for k, _ in order], colors, xmax=0.3)
    ax1.set_title("Boundary F1, within 5 frames", loc="left", pad=16)
    ax2 = fig.add_axes([0.64, 0.12, 0.30, 0.70])
    hbars(ax2, [""] * len(order), [rows[k]["contact_iou"] for k, _ in order], colors, xmax=1.05)
    ax2.set_title("Contact overlap (IoU)", loc="left", pad=16)
    fig.text(0.04, 0.035, "Ground truth: touch-sensing glove. The timing shift is learned on tuning episodes only.",
             fontsize=22, color=MUTED)
    save(fig, "demo_number1.png")


def number2():
    d = load("out/number2/final/number2_final.json")
    rows = [("Our score", d["our_score"]),
            ("Muse Spark alone, same question", d["muse_alone"]),
            ("Object name, seen objects", d["object_reference"]),
            ("Our score, unseen objects", d["unseen_objects"]["our_score"])]
    for name, hi in (("demo_number2_ours.png", {0: HI, 1: HI2}), ("demo_number2_limits.png", {2: HI2, 3: HI})):
        fig = canvas()
        fig.suptitle("Can a robot hand copy this grasp?  1,426 held-out clips from 3 people",
                     fontsize=32, x=0.04, ha="left", y=0.95)
        ax = fig.add_axes([0.36, 0.14, 0.58, 0.70])
        hbars(ax, [r[0] for r in rows], [r[1]["auc"] for r in rows],
              [hi.get(i, BASE) for i in range(len(rows))],
              ci=[tuple(r[1]["ci95_by_sequence"]) for r in rows], xmax=1.0)
        ax.axvline(0.5, color=MUTED, lw=2, ls="--")
        ax.text(0.505, len(rows) - 0.45, "chance", color=MUTED, fontsize=22)
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("AUC against the robot copy label (bars: 95% interval over sequences)", labelpad=14)
        save(fig, name)


def soft():
    d = load("out/soft/soft_check.json")
    caught, held = d["caught"], d["hot3d_held_back"]
    fig = canvas()
    fig.suptitle("Soft object check: it fails its pass rule", fontsize=34, x=0.04, ha="left", y=0.93)
    items = [("Soft objects held back\n(%d OpenGraph segments)" % caught["segments"], caught["all"], 0.70, "at least 70%",
              caught["all"] >= 0.70),
             ("Rigid objects held back\n(%s HOT3D clips)" % f"{held['clips']:,}", held["all"], 0.10, "at most 10%",
              held["all"] <= 0.10)]
    for i, (label, value, line, rule, ok) in enumerate(items):
        ax = fig.add_axes([0.07 + 0.48 * i, 0.12, 0.40, 0.58])
        ax.bar([0], [value], width=0.5, color="#2b8a3e" if ok else "#c92a2a")
        ax.axhline(line, color=TEXT, lw=3, ls="--")
        ax.text(0.33, line + 0.02, "pass rule: " + rule, fontsize=24)
        ty = line + 0.03 if 0 < line - value < 0.12 else value + 0.02
        ax.text(0, ty, f"{100 * value:.1f}%", ha="center", fontsize=40, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.6, 1.2)
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_title(label + ("   met" if ok else "   not met"), loc="left", pad=18)
    save(fig, "demo_soft.png")


def box(fig, x, y, w, h, color, alpha=1.0):
    fig.patches.append(plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=color,
                                     edgecolor="none", alpha=alpha, zorder=0))


def cards():
    # Scene 1: text over the opening clip, so the PNG is transparent with a dark band.
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    box(fig, 0, 0.10, 1, 0.20, "#000000", alpha=0.55)
    fig.text(0.5, 0.20, "Not every hand video can teach a robot hand.", ha="center", va="center",
             fontsize=46, fontweight="bold", color="white")
    save(fig, "demo_card_scene1.png", transparent=True)

    # Scene 2: what goes in and what comes out.
    fig = canvas()
    fig.text(0.06, 0.84, "GraspGrade", fontsize=80, fontweight="bold")
    box(fig, 0.06, 0.30, 0.26, 0.38, "#eef3fc")
    fig.text(0.08, 0.62, "INPUT", fontsize=24, color=HI, fontweight="bold")
    fig.text(0.08, 0.49, "Human\negocentric\nvideo", fontsize=40, fontweight="bold", va="center", linespacing=1.2)
    fig.text(0.08, 0.35, "any length", fontsize=28, color=MUTED)
    fig.text(0.37, 0.49, "→", fontsize=110, color=HI, ha="center", va="center")
    fig.text(0.42, 0.62, "OUTPUT", fontsize=24, color=HI, fontweight="bold")
    for i, line in enumerate(["Clips cut at each touch",
                              "A 0 to 100 score: can a robot hand copy it?",
                              "A one-line reason"]):
        y = 0.52 - 0.10 * i
        fig.text(0.42, y, str(i + 1), fontsize=36, fontweight="bold", color=HI, va="center")
        fig.text(0.455, y, line, fontsize=32, va="center")
    fig.text(0.06, 0.14, "SAM 3.1  ·  Muse Spark", fontsize=34, color=MUTED)
    save(fig, "demo_card_scene2.png")

    # Scene 6: what is open, and the data sources.
    fig = canvas()
    fig.text(0.5, 0.64, "GraspGrade", fontsize=110, fontweight="bold", ha="center")
    fig.text(0.5, 0.54, "Find the hand videos a robot can learn from.", fontsize=40, ha="center", color=MUTED)
    for x, title, sub in [(0.22, "Code", "every step,\none script each"),
                          (0.50, "Viewer", "held-out clips\nand episodes"),
                          (0.78, "Benchmark", "both label sets,\nstandalone scorers")]:
        fig.text(x, 0.36, title, fontsize=46, fontweight="bold", ha="center", color=HI)
        fig.text(x, 0.33, sub, fontsize=28, ha="center", va="top", color=MUTED, linespacing=1.4)
    fig.text(0.5, 0.10, "Data: OpenGraph Ego-Tactile (CC BY 4.0)  ·  HOT3D, Meta (CC BY-SA 4.0)  ·  Voice: ElevenLabs", fontsize=24,
             ha="center", color=MUTED)
    save(fig, "demo_card_scene6.png")


if __name__ == "__main__":
    number1()
    number2()
    soft()
    cards()
