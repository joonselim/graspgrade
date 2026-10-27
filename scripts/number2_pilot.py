"""P1 pilot checks for number 2 (EXP-23). No API call, reads what the pilot sent.

    python scripts/number2_pilot.py

1. SAM on HOT3D windows: per contact frame of each clip, the object that the
   number 2 code takes as nearest to the clip's hand, against the dataset 2D
   box of the grasped object (box2d_objects.csv, raw orientation, turned to
   the preview with (1407 - v, u)). Also whether any SAM object mask matches
   that box at all.
2. Muse checklist object items against the 3D model stand-ins of EXP-19
   (out/hot3d_checks/object_items.csv), and the baseline probability against
   the v2 label.
3. Real token use and cost per call, and the paid steps P6 and P7 rescaled.
Writes out/number2/pilot.json and docs/experiments/img/exp23_*.png.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_fast  # noqa: E402
from contact_cut import frame_table  # noqa: E402
from hot3d_calib_check import ARIA, iou  # noqa: E402
from number2_sam_checks import SIZE, STEP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
IMG = ROOT / "docs" / "experiments" / "img"
ITEMS = ["size", "shape", "thickness", "width", "handle", "free_space"]
MATCH_IOU = 0.5


def gt_boxes(seq, uid):
    """Sequence frame index -> preview box of object uid, from the dataset 2D boxes."""
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
    b = pd.read_csv(gt / "box2d_objects.csv")
    b = b[(b["stream_id"] == "214-1") & (b["object_uid"] == uid)].dropna().set_index("timestamp[ns]")
    out = {}
    for f, t in enumerate(ts):
        if t in b.index:
            r = b.loc[t]
            x0, y0, x1, y1 = r["x_min[pixel]"], r["y_min[pixel]"], r["x_max[pixel]"], r["y_max[pixel]"]
            out[f] = (SIZE - 1 - y1, x0, SIZE - 1 - y0, x1)
    return out


def object_uid(seq, name):
    meta = json.loads((ARIA / seq / "ground_truth" / "metadata.json").read_text())
    return int(dict(zip(meta["object_names"], meta["object_uids"]))[name])


def sam_check(w, clips):
    rows, shots = [], []
    for win in w.itertuples():
        d = N2 / "sam" / win.window_id
        if not (d / "sam_small_object_events.jsonl.gz").exists():
            continue
        n10 = int(win.frames_10fps)
        rec, _ = frame_table(d, "_events.jsonl.gz", n10, (SIZE, SIZE))
        objs = dict(sam_fast.read_frames(d / "sam_small_object_events.jsonl.gz"))
        for cid in json.loads(win.clips):
            c = clips.loc[cid]
            gtb = gt_boxes(c.sequence, object_uid(c.sequence, c.object))
            for k in range(n10):
                f = int(win.start) + STEP * k
                if not (c.contact_start <= f < c.contact_start + c.contact_frames) or f not in gtb:
                    continue
                by_id = {o.id: o.box for o in objs.get(k, [])}
                best_any = max((iou(b, gtb[f]) for b in by_id.values()), default=0.0)
                oid = rec[c.hand]["oid"][k]
                chosen = iou(by_id[str(int(oid))], gtb[f]) if not np.isnan(oid) and str(int(oid)) in by_id else np.nan
                rows.append({"clip_id": cid, "object": c.object, "frame": f, "k": k, "n_objects": len(by_id),
                             "best_any_iou": best_any, "chosen_iou": chosen, "gap_px": rec[c.hand]["gap"][k]})
            mid = [r for r in rows if r["clip_id"] == cid]
            if mid:
                shots.append((win.window_id, mid[len(mid) // 2], c))
    return pd.DataFrame(rows), shots


def draw_shots(shots, path):
    fig, axes = plt.subplots(1, len(shots), figsize=(4.2 * len(shots), 4.6), squeeze=False)
    for ax, (wid, r, c) in zip(axes[0], shots):
        cap = cv2.VideoCapture(str(N2 / "sam_windows" / f"{wid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["k"])
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        d = N2 / "sam" / wid
        rec, _ = frame_table(d, "_events.jsonl.gz", r["k"] + 1, (SIZE, SIZE))
        objs = dict(sam_fast.read_frames(d / "sam_small_object_events.jsonl.gz")).get(r["k"], [])
        oid = rec[c.hand]["oid"][r["k"]]
        for o in objs:
            m = o.mask()
            col = (255, 140, 0) if not np.isnan(oid) and o.id == str(int(oid)) else (120, 120, 255)
            sub = img[o.y0:o.y0 + m.shape[0], o.x0:o.x0 + m.shape[1]]
            mm = m[:sub.shape[0], :sub.shape[1]]
            sub[mm] = (0.5 * sub[mm] + 0.5 * np.array(col)).astype(np.uint8)
        g = gt_boxes(c.sequence, object_uid(c.sequence, c.object))[r["frame"]]
        cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 220, 0), 6)
        ax.imshow(img)
        ax.set_title(f"{c.object}, {c.hand} hand\nchosen IoU {r['chosen_iou']:.2f}, "
                     f"{r['n_objects']} SAM objects", fontsize=9)
        ax.axis("off")
    fig.suptitle("Green: dataset box of the grasped object. Orange: object SAM takes as nearest. "
                 "Blue: other SAM objects.", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def muse_check(clips, items):
    rows = []
    for f in sorted((N2 / "muse" / "checklist").glob("*.json")):
        r = json.loads(f.read_text())
        b = json.loads((N2 / "muse" / "baseline" / f.name).read_text())
        c = clips.loc[r["clip_id"]]
        m = items[(items.sequence == c.sequence) & (items.object == c.object) & (items.hand == c.hand)
                  & (items.clip_start == c.clip_start)]
        row = {"clip_id": r["clip_id"], "object": c.object, "muse_name": r["parsed"]["object_name"],
               "copyable": int(c.copyable), "baseline_p": b["parsed"]["copyable_probability"],
               "finger_occlusion": r["parsed"]["finger_occlusion"]}
        for k in ITEMS:
            row[f"muse_{k}"] = r["parsed"][f"object_{k}" if k != "free_space" else "free_space"]
            row[f"stand_{k}"] = float(m[f"item_{k}"].iloc[0]) if len(m) else np.nan
        for rr, task in ((r, "checklist"), (b, "baseline")):
            row[f"{task}_in"] = rr["usage"]["prompt_tokens"]
            row[f"{task}_out"] = rr["usage"]["completion_tokens"]
            row[f"{task}_usd"] = rr["cost_usd"]
        rows.append(row)
    return pd.DataFrame(rows)


def draw_items(m, path):
    lab = {1.0: "high", 0.5: "mid", 0.0: "low"}
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(m) + 1.4))
    agree = np.array([[m[f"muse_{k}"].iloc[i] == m[f"stand_{k}"].iloc[i] for k in ITEMS] for i in range(len(m))])
    ax.imshow(agree, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(m)):
        for j, k in enumerate(ITEMS):
            ax.text(j, i, f"{lab[m[f'muse_{k}'].iloc[i]]} / {lab.get(m[f'stand_{k}'].iloc[i], '?')}",
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(ITEMS)), ITEMS, fontsize=8)
    ax.set_yticks(range(len(m)), [f"{o} (label {y}, base {p:.2f})" for o, y, p in
                                  zip(m.object, m.copyable, m.baseline_p)], fontsize=8)
    ax.set_title("Muse grade / 3D model stand-in. Green: same, red: different", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    clips = pd.read_csv(N2 / "clips_final.csv").set_index("clip_id")
    w = pd.read_csv(N2 / "sam_windows.csv")
    items = pd.read_csv(ROOT / "out" / "hot3d_checks" / "object_items.csv")

    s, shots = sam_check(w, clips)
    per_clip = s.groupby(["clip_id", "object"]).agg(
        frames=("frame", "size"), n_objects=("n_objects", "mean"),
        detected=("best_any_iou", lambda x: float((x >= MATCH_IOU).mean())),
        chosen_right=("chosen_iou", lambda x: float((x >= MATCH_IOU).mean())),
        chosen_iou=("chosen_iou", "median")).reset_index()
    print(per_clip.round(2).to_string(index=False))
    draw_shots(shots, IMG / "exp23_sam_chosen_object.png")

    m = muse_check(clips, items)
    agree = {k: int((m[f"muse_{k}"] == m[f"stand_{k}"]).sum()) for k in ITEMS}
    print(m[["clip_id", "object", "muse_name", "copyable", "baseline_p", "finger_occlusion"]].to_string(index=False))
    print("item agreement out of", len(m), agree)
    draw_items(m, IMG / "exp23_muse_items.png")

    w_sent = w[[(N2 / "sam" / i / "sam_small_object_events.jsonl.gz").exists() for i in w.window_id]]
    est_ck = 6 * 1100 + len((ROOT / "scripts" / "prompts" / "number2_checklist.txt").read_text()) // 4
    est_bl = 6 * 1100 + len((ROOT / "scripts" / "prompts" / "number2_baseline.txt").read_text()) // 4
    cost = {
        "checklist_usd_per_call": float(m.checklist_usd.mean()), "baseline_usd_per_call": float(m.baseline_usd.mean()),
        "checklist_in": float(m.checklist_in.mean()), "checklist_out": float(m.checklist_out.mean()),
        "baseline_in": float(m.baseline_in.mean()), "baseline_out": float(m.baseline_out.mean()),
        "checklist_usd_estimated": est_ck * 1.25e-6 + 800 * 4.25e-6,
        "baseline_usd_estimated": est_bl * 1.25e-6 + 800 * 4.25e-6,
        "sam_windows_sent": int(len(w_sent)),
    }
    print(json.dumps(cost, indent=1))
    out = {"sam_per_clip": per_clip.to_dict("records"),
           "sam_frames": {"detected": float((s.best_any_iou >= MATCH_IOU).mean()),
                          "chosen_right": float((s.chosen_iou >= MATCH_IOU).mean()), "n": int(len(s))},
           "muse": m.to_dict("records"), "item_agreement": agree, "cost": cost}
    (N2 / "pilot.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
