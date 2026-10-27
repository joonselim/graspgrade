"""Check SAM 3.1 tracks on whole OpenGraph episodes (no API call).

    python scripts/sam_track_check.py 0 30 53

For each episode in out/sam10/ it reports, per prompt: frames with a
detection, objects per frame, how many ids and how long they live, and for
hands whether each hand id stays on one side of the image. Against the
dataset labels it reports, per hand, the share of contact frames where SAM
has that hand. Writes out/sam10/track_check.json and one timeline image per
episode to docs/experiments/img/.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_fast  # noqa: E402
from contact_cut import SAM, SIDES, STEP, W, ground_truth, hand_sides  # noqa: E402

IMG = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "img"


def boxes(ep, prompt):
    path = SAM / f"episode_{ep:06d}" / f"sam_{prompt}_events.jsonl.gz"
    return {f: {o.id: o.box for o in objs} for f, objs in sam_fast.read_frames(path)}


def id_stats(b, n10):
    ids = sorted({i for f in b for i in b[f]}, key=int)
    life = {i: [f for f in range(n10) if i in b.get(f, {})] for i in ids}
    per_frame = [len(b.get(f, {})) for f in range(n10)]
    return ids, life, {
        "frames_with_detection_pct": round(100 * np.mean([c > 0 for c in per_frame]), 1),
        "objects_per_frame_mean": round(float(np.mean(per_frame)), 2),
        "objects_per_frame_max": int(max(per_frame)),
        "n_ids": len(ids),
        "ids_seen_in_90pct_of_frames": sum(len(v) >= 0.9 * n10 for v in life.values()),
        "ids_shorter_than_1s": sum(len(v) < 10 for v in life.values()),
        "id_frames_top5": sorted((len(v) for v in life.values()), reverse=True)[:5],
    }


def check(ep, gt):
    n30 = len(gt["label"]["left"])
    n10 = -(-n30 // STEP)
    hb, ob = boxes(ep, "hand"), boxes(ep, "small_object")
    hids, hlife, hstat = id_stats(hb, n10)
    oids, olife, ostat = id_stats(ob, n10)
    cx = {i: [(hb[f][i][0] + hb[f][i][2]) / 2 for f in hlife[i]] for i in hids}
    side_of = hand_sides(hb)
    hstat["hand_id_side"] = {i: side_of[i] for i in hids}
    hstat["frames_past_image_centre_pct"] = {
        i: round(100 * float(np.mean([(c < W / 2) != (side_of[i] == "left") for c in v])), 1) for i, v in cx.items()}
    have = {s: np.zeros(n10, bool) for s in SIDES}
    for f in range(n10):
        for i in hb.get(f, {}):
            have[side_of[i]][f] = True
    pairs = [(hb[f], [i for i in hb[f] if side_of[i] == "left"], [i for i in hb[f] if side_of[i] == "right"])
             for f in range(n10) if f in hb]
    order = [min((b[i][0] + b[i][2]) for i in l) < min((b[i][0] + b[i][2]) for i in r) for b, l, r in pairs if l and r]
    hstat["left_id_left_of_right_id_pct"] = round(100 * float(np.mean(order)), 1) if order else None
    contact = {}
    for s in SIDES:
        lab = gt["label"][s][::STEP][:n10]
        contact[s] = {"contact_frames_10fps": int(lab.sum()),
                      "hand_found_in_contact_pct": round(100 * float(have[s][lab].mean()), 1) if lab.any() else None,
                      "hand_found_outside_contact_pct": round(100 * float(have[s][~lab].mean()), 1)}
    plot(ep, n10, gt, have, hb, hids, side_of, ob, oids)
    return {"frames_10fps": n10, "hand": hstat, "small_object": ostat, "per_hand_vs_label": contact}


def plot(ep, n10, gt, have, hb, hids, side_of, ob, oids):
    t = np.arange(n10) / 10
    fig, ax = plt.subplots(figsize=(12, 2.2 + 0.18 * len(oids)))
    y = 0
    labels = []
    for s in SIDES:
        lab = gt["label"][s][::STEP][:n10]
        ax.fill_between(t, y, y + 0.8, where=lab, color="#2b8a3e", step="mid", lw=0)
        labels.append((y + 0.4, f"{s} contact (label)"))
        y += 1
        ax.fill_between(t, y, y + 0.8, where=have[s], color="#1c7ed6", step="mid", lw=0)
        labels.append((y + 0.4, f"{s} hand found"))
        y += 1
    for i in oids:
        on = np.array([i in ob.get(f, {}) for f in range(n10)])
        ax.fill_between(t, y, y + 0.6, where=on, color="#e8590c", step="mid", lw=0)
        labels.append((y + 0.3, f"object id {i}"))
        y += 0.8
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([n for _, n in labels], fontsize=7)
    ax.set_xlabel("seconds")
    ax.set_xlim(0, t[-1])
    ax.set_title(f"Episode {ep}: SAM 3.1 at 10 fps over the whole episode")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / f"exp06_ep{ep}_timeline.png", dpi=120)
    plt.close(fig)


def main():
    eps = [int(a) for a in sys.argv[1:]]
    gt = ground_truth(eps)
    report = {ep: check(ep, gt[ep]) for ep in eps}
    out = SAM / "track_check.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
