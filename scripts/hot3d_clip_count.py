"""Count grasp clips in HOT3D from motion capture only, to price number 2 (no API call).

    python scripts/hot3d_clip_count.py

HOT3D objects only move when a hand moves them. So for each object we find
the stretches where its mocap position moves, and call each stretch one grasp
event. A clip is the event plus 1 s before (reach) and 1 s after (release).
This is only for counting and pricing. The real clips for number 2 will be
cut on Day 5 from the hand to object distance.

Uses the RGB frame times (stream 214-1 in masks/mask_qa_pass.csv), 30 fps.
Checks per event: the nearest MANO wrist comes within 0.25 m of the object,
and the share of clip frames that pass the dataset QA mask.
Writes out/hot3d_clips/events.csv and summary.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARIA = ROOT / "data" / "hot3d" / "aria"
OUT = ROOT / "out" / "hot3d_clips"
FPS = 30
SPEED = 0.03          # m/s over a 0.2 s window, static noise is about 0.0003
MERGE_GAP = 15        # frames, join moves less than 0.5 s apart
MIN_LEN = 9           # frames, drop moves shorter than 0.3 s
MIN_DISP = 0.03       # m, drop moves that end less than 3 cm from where they began
PAD = 30              # frames added before and after, 1 s each
WRIST_NEAR = 0.25     # m

SAM_PRICE_FRAME, SAM_MIN_FRAMES = 0.20 / 1000, 50


def segments(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def wrists(seq, ts):
    """Wrist positions of hands 0 and 1 at each RGB time, nan when missing."""
    idx = {t: i for i, t in enumerate(ts)}
    w = np.full((len(ts), 2, 3), np.nan)
    with open(ARIA / seq / "hand_data" / "mano_hand_pose_trajectory.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            i = idx.get(r["timestamp_ns"])
            if i is None:
                continue
            for h, pose in r.get("hand_poses", {}).items():
                if h in ("0", "1"):
                    w[i, int(h)] = pose["wrist_xform"]["t_xyz"]
    return w


def sequence_events(seq):
    gt = ARIA / seq / "ground_truth"
    qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
    rgb = qa[qa["stream_id"] == "214-1"].sort_values("timestamp[ns]")
    ts = rgb["timestamp[ns]"].values
    qa_ok = rgb["mask"].astype(bool).values
    n = len(ts)
    names = dict(zip(*(json.loads((gt / "metadata.json").read_text())[k] for k in ("object_uids", "object_names"))))
    d = pd.read_csv(gt / "dynamic_objects.csv")
    d = d[d["timestamp[ns]"].isin(set(ts))]
    w = wrists(seq, ts)
    rows = []
    for uid, g in d.groupby("object_uid"):
        g = g.set_index("timestamp[ns]").reindex(ts)
        p = g[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values
        v = np.full(n, np.nan)
        v[3:-3] = np.linalg.norm(p[6:] - p[:-6], axis=1) / 0.2
        moving = np.nan_to_num(v) > SPEED
        segs = []
        for a, b in segments(moving):
            if segs and a - segs[-1][1] <= MERGE_GAP:
                segs[-1] = (segs[-1][0], b)
            else:
                segs.append((a, b))
        for a, b in segs:
            pa, pb = p[a], p[b - 1]
            disp = float(np.linalg.norm(pb - pa)) if not (np.isnan(pa).any() or np.isnan(pb).any()) else np.nan
            if b - a < MIN_LEN or not disp >= MIN_DISP:
                continue
            dist = np.linalg.norm(w[a:b] - p[a:b, None, :], axis=2)
            s, e = max(0, a - PAD), min(n, b + PAD)
            rows.append({"sequence": seq, "object": names.get(str(uid), str(uid)), "move_start": int(a),
                         "move_frames": int(b - a), "clip_start": int(s), "clip_frames": int(e - s),
                         "displacement_m": round(disp, 3),
                         "wrist_min_dist_m": round(float(np.nanmin(dist)), 3) if np.isfinite(dist).any() else None,
                         "clip_qa_pass_pct": round(100 * float(qa_ok[s:e].mean()), 1)})
    return rows, n, float(qa_ok.mean())


def merged_windows(ev):
    """Overlapping clips in one sequence joined into one window (one SAM request)."""
    out = []
    for seq, g in ev.groupby("sequence"):
        spans = sorted(zip(g.clip_start, g.clip_start + g.clip_frames))
        cur = None
        for s, e in spans:
            if cur and s <= cur[1]:
                cur[1] = max(cur[1], e)
            else:
                if cur:
                    out.append(cur[1] - cur[0])
                cur = [s, e]
        if cur:
            out.append(cur[1] - cur[0])
    return np.array(out)


def sam_cost(frames30, prompts=2, step=3):
    n10 = np.ceil(np.asarray(frames30) / step)
    billed = np.maximum(n10, SAM_MIN_FRAMES).sum() * prompts
    return int(billed), round(float(billed * SAM_PRICE_FRAME), 2)


def plot(ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = ev[(ev.wrist_min_dist_m <= WRIST_NEAR) & (ev.clip_qa_pass_pct >= 90)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(good.clip_frames / FPS, bins=np.arange(0, 30.5, 0.5), color="#1c7ed6")
    axes[0].axvline(SAM_MIN_FRAMES / 10, color="#e8590c", ls="--", lw=1, label="5 s = SAM minimum (50 frames at 10 fps)")
    axes[0].set_xlabel("clip length in seconds (move plus 1 s each side)")
    axes[0].set_ylabel("clips")
    axes[0].legend(fontsize=8)
    per_seq = good.groupby(["split", "sequence"]).size()
    for part, c in (("tune", "#2b8a3e"), ("held_out", "#868e96")):
        axes[1].hist(per_seq[part], bins=np.arange(0, 45, 2), alpha=0.7, color=c, label=part.replace("_", "-"))
    axes[1].set_xlabel("clips per sequence")
    axes[1].set_ylabel("sequences")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"HOT3D: {len(good)} grasp clips from motion capture in {good.sequence.nunique()} sequences")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "experiments" / "img" / "exp07_hot3d_clips.png", dpi=120)


def main():
    split = json.loads((ROOT / "splits" / "hot3d_aria.json").read_text())
    rows, seq_info = [], {}
    for part in ("tune", "held_out"):
        for seq in split[part]:
            r, n, qa = sequence_events(seq)
            for x in r:
                x["split"] = part
            rows += r
            seq_info[seq] = {"split": part, "rgb_frames": n, "qa_pass": qa}
            print(f"{seq}: {len(r)} events", flush=True)
    ev = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "events.csv", index=False)

    summary = {}
    for part, g in [("tune", ev[ev.split == "tune"]), ("held_out", ev[ev.split == "held_out"]), ("all", ev)]:
        near = g[g.wrist_min_dist_m <= WRIST_NEAR]
        good = near[near.clip_qa_pass_pct >= 90]
        win = merged_windows(good)
        summary[part] = {
            "sequences": int(sum(v["split"] == part or part == "all" for v in seq_info.values())),
            "rgb_frames": int(sum(v["rgb_frames"] for v in seq_info.values() if v["split"] == part or part == "all")),
            "events": len(g),
            "events_wrist_within_25cm": len(near),
            "events_wrist_near_and_qa_90pct": len(good),
            "move_seconds_median": round(float(good.move_frames.median()) / FPS, 2) if len(good) else None,
            "clip_seconds_median": round(float(good.clip_frames.median()) / FPS, 2) if len(good) else None,
            "clip_seconds_p90": round(float(good.clip_frames.quantile(0.9)) / FPS, 2) if len(good) else None,
            "sam_per_event_10fps_2_prompts": sam_cost(good.clip_frames),
            "sam_per_event_30fps_2_prompts": sam_cost(good.clip_frames, step=1),
            "merged_windows": len(win),
            "sam_merged_windows_10fps_2_prompts": sam_cost(win),
        }
    summary["objects_top"] = ev.object.value_counts().head(10).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(ev)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
