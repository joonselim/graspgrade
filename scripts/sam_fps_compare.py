"""Compare SAM 3.1 results on a 10 fps clip with the 30 fps results on the same 5 s.

    python scripts/sam_fps_compare.py

Reads the events written by meta_smoke.py: the 30 fps runs in out/meta_smoke/
and out/meta_smoke/ep53/, the 10 fps runs (made with --step 3) in
out/meta_smoke/fps10/ep1/ and out/meta_smoke/fps10/ep53/. Clip frame k at
10 fps is clip frame 3k at 30 fps, so every 10 fps frame has a 30 fps twin.

For hand and small object it reports detection, ids, the mask IoU between each
10 fps id and its 30 fps twin, and the right hand to object gap before and
after the tactile contact. Writes out/meta_smoke/fps10/fps_compare.json, one
overlay grid per clip and a per-frame IoU plot to docs/experiments/img/.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_prompt_compare as spc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "experiments" / "img"
STEP, N10 = 3, 50
CLIPS = {  # name: (30 fps folder, 10 fps folder, contact frame at 30 fps)
    "ep1": (ROOT / "out/meta_smoke", ROOT / "out/meta_smoke/fps10/ep1", 51),
    "ep53": (ROOT / "out/meta_smoke/ep53", ROOT / "out/meta_smoke/fps10/ep53", 50),
}
PROMPTS = ["hand", "small object"]
GRID_K = [0, 17, 33, 49]


def load(folder, prompt):
    spc.OUT = folder
    return spc.load(prompt)


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def match_ids(b30, m30, b10, m10):
    """Pair each 10 fps id with the 30 fps id it overlaps most, by mean mask IoU."""
    ids10 = sorted({i for f in b10 for i in b10[f]})
    ids30 = sorted({i for f in b30 for i in b30[f]})
    per_frame = {}  # (id10, id30) -> {k: iou}
    for k in range(N10):
        for i10, r10 in m10.get(k, {}).items():
            a = spc.full_mask(r10) > 0
            for i30, r30 in m30.get(STEP * k, {}).items():
                per_frame.setdefault((i10, i30), {})[k] = mask_iou(a, spc.full_mask(r30) > 0)
    scores = sorted(((np.mean(list(v.values())), p) for p, v in per_frame.items()), reverse=True)
    pairs, used30 = {}, set()
    for s, (i10, i30) in scores:
        if s > 0.3 and i10 not in pairs and i30 not in used30:
            pairs[i10] = i30
            used30.add(i30)
    rows = []
    for i10 in ids10:
        f10 = {k for k in range(N10) if i10 in b10.get(k, {})}
        row = {"id_10fps": i10, "frames_10fps": len(f10)}
        if i10 in pairs:
            i30 = pairs[i10]
            f30 = {k for k in range(N10) if i30 in b30.get(STEP * k, {})}
            vals = per_frame[(i10, i30)]
            # a frame where only one of the two runs has the object counts as IoU 0
            ious = [vals.get(k, 0.0) for k in sorted(f10 | f30)]
            row.update({"id_30fps": i30, "frames_30fps_at_same_times": len(f30),
                        "mask_iou_mean": round(float(np.mean(ious)), 3),
                        "mask_iou_median": round(float(np.median(ious)), 3),
                        "mask_iou_min": round(float(np.min(ious)), 3),
                        "frames_iou_below_0_5": int(sum(v < 0.5 for v in ious)),
                        "iou_by_frame": {k: round(vals.get(k, 0.0), 3) for k in sorted(f10 | f30)}})
        rows.append(row)
    unmatched30 = [{"id_30fps": i, "frames_30fps_at_same_times": sum(i in b30.get(STEP * k, {}) for k in range(N10))}
                   for i in ids30 if i not in used30]
    return rows, unmatched30


def gaps(hands, hand_masks, masks, frames):
    out = []
    for f in frames:
        rb = spc.right_hand(hands.get(f, {}))
        objs = [o for o in (spc.full_mask(r) for r in masks.get(f, {}).values()) if o.any()]
        if rb is None or not objs:
            out.append(np.nan)
            continue
        rid = next(k for k, v in hands[f].items() if v == rb)
        dist = cv2.distanceTransform((1 - spc.full_mask(hand_masks[f][rid])).astype(np.uint8), cv2.DIST_L2, 3)
        out.append(min(float(dist[o > 0].min()) for o in objs))
    return np.array(out)


def near(g):
    g = g[~np.isnan(g)]
    return round(100 * float(np.mean(g <= 3)), 1) if len(g) else None


def read_frames(path, keep):
    cap, got, i = cv2.VideoCapture(str(path)), {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in keep:
            got[i] = img
        i += 1
    cap.release()
    return got


def draw(img, hand_masks, obj_boxes, obj_masks, label):
    img = img.copy()
    colours = [(0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
    layers = [(rec, (255, 120, 0), None) for rec in hand_masks.values()]
    layers += [(rec, colours[int(oid) % len(colours)], oid) for oid, rec in sorted(obj_masks.items())]
    for rec, c, oid in layers:
        m = spc.full_mask(rec) > 0
        img[m] = (0.45 * img[m] + 0.55 * np.array(c)).astype(np.uint8)
        if oid is not None:
            b = obj_boxes[oid]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
            cv2.putText(img, f"id{oid}", (int(b[0]), int(b[1]) - 8), 0, 1.1, c, 3)
    cv2.putText(img, label, (20, 60), 0, 1.6, (0, 0, 0), 6)
    cv2.putText(img, label, (20, 60), 0, 1.6, (255, 255, 255), 2)
    return cv2.resize(img, (480, 270))


def main():
    report, curves = {}, {}
    for name, (d30, d10, contact) in CLIPS.items():
        data = {p: (load(d30, p), load(d10, p)) for p in PROMPTS}
        (h30, hm30), (h10, hm10) = data["hand"]
        (o30, om30), (o10, om10) = data["small object"]
        rep = {}
        for p in PROMPTS:
            (b30, m30), (b10, m10) = data[p]
            rows, un30 = match_ids(b30, m30, b10, m10)
            rep[p] = {
                "frames_with_detection_10fps_pct": round(100 * sum(bool(b10.get(k)) for k in range(N10)) / N10, 1),
                "frames_with_detection_30fps_same_times_pct":
                    round(100 * sum(bool(b30.get(STEP * k)) for k in range(N10)) / N10, 1),
                "ids_10fps": rows, "unmatched_ids_30fps": un30,
            }
            curves[(name, p)] = rows
        k_contact = -(-contact // STEP)  # first 10 fps frame at or after the contact
        g10 = gaps(h10, hm10, om10, range(N10))
        g30s = gaps(h30, hm30, om30, [STEP * k for k in range(N10)])
        g30 = gaps(h30, hm30, om30, range(150))
        rep["right_hand_to_object_within_3px"] = {
            "contact_frame_10fps": k_contact,
            "10fps_before": near(g10[:k_contact]), "10fps_after": near(g10[k_contact:]),
            "30fps_same_times_before": near(g30s[:k_contact]), "30fps_same_times_after": near(g30s[k_contact:]),
            "30fps_all_frames_before": near(g30[:contact]), "30fps_all_frames_after": near(g30[contact:]),
            "gap_px_10fps": [None if np.isnan(v) else round(float(v), 1) for v in g10],
            "gap_px_30fps_same_times": [None if np.isnan(v) else round(float(v), 1) for v in g30s],
        }
        report[name] = rep

        f30 = read_frames(d30 / "clip.mp4", {STEP * k for k in GRID_K})
        f10 = read_frames(d10 / "clip.mp4", set(GRID_K))
        top = [draw(f30[STEP * k], hm30.get(STEP * k, {}), o30.get(STEP * k, {}), om30.get(STEP * k, {}),
                    f"30 fps  f{STEP * k}") for k in GRID_K]
        bot = [draw(f10[k], hm10.get(k, {}), o10.get(k, {}), om10.get(k, {}), f"10 fps  f{STEP * k}")
               for k in GRID_K]
        cv2.imwrite(str(IMG / f"exp05_{name}_30fps_vs_10fps.jpg"), np.vstack([np.hstack(top), np.hstack(bot)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, name in zip(axes, CLIPS):
        for p, ls in [("hand", "-"), ("small object", "--")]:
            for row in curves[(name, p)]:
                if "iou_by_frame" not in row:
                    continue
                ks = sorted(row["iou_by_frame"])
                ax.plot([STEP * k for k in ks], [row["iou_by_frame"][k] for k in ks], ls, marker=".",
                        label=f"{p} id {row['id_10fps']}")
        ax.axvline(CLIPS[name][2], color="grey", lw=1, ls=":")
        ax.set_title(f"{name}: 10 fps mask vs 30 fps mask, same frame")
        ax.set_xlabel("frame (30 fps index), dotted line = tactile contact")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("mask IoU")
    fig.tight_layout()
    fig.savefig(IMG / "exp05_mask_iou_by_frame.png", dpi=130)

    out = ROOT / "out/meta_smoke/fps10/fps_compare.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    for name, rep in report.items():
        for p in PROMPTS:
            r = rep[p]
            print(name, p, "det10", r["frames_with_detection_10fps_pct"], "det30", r["frames_with_detection_30fps_same_times_pct"])
            for row in r["ids_10fps"]:
                print("   ", {k: v for k, v in row.items() if k != "iou_by_frame"})
            print("    unmatched 30fps:", r["unmatched_ids_30fps"])
        print(name, {k: v for k, v in rep["right_hand_to_object_within_3px"].items() if not k.startswith("gap")})


if __name__ == "__main__":
    main()
