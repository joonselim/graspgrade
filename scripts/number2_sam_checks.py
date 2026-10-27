"""Two number 2 checks from SAM 3.1 masks: visibility and contact clarity. No API call.

    python scripts/number2_sam_checks.py           # after sam_hot3d.py has run
    python scripts/number2_sam_checks.py --selftest

Per SAM window (out/number2/sam/<window_id>/), the same frame table as the
number 1 code (contact_cut.frame_table): per 10 fps frame and hand, the gap
between the hand mask and the nearest object mask. Then per clip, for the
clip's hand:
  visibility       share of clip frames where SAM sees that hand and an object
  contact clarity  share of contact frames where the hand is within 3 px of an
                   object, minus that share in the reach and release frames
Grades high, mid, low from the terciles of the score_train clips.
Writes out/number2/sam_checks.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_cut import frame_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
STEP, NEAR_PX = 3, 3
SIZE = 1408  # HOT3D preview frames are 1408 x 1408


def clip_values(rec, start, clips):
    rows = []
    n10 = len(rec["left"]["gap"])
    for c in clips.itertuples():
        side = rec[c.hand]
        k = np.arange(n10)
        f = start + STEP * k
        clip = (f >= c.clip_start) & (f < c.clip_start + c.clip_frames)
        con = (f >= c.contact_start) & (f < c.contact_start + c.contact_frames)
        ends = clip & ~con
        seen = ~np.isnan(side["hx"]) & ~np.isnan(side["gap"])
        near = np.nan_to_num(side["gap"], nan=1e9) <= NEAR_PX
        vis = float(seen[clip].mean()) if clip.any() else 0.0
        clarity = (float(near[con].mean()) if con.any() else 0.0) - (float(near[ends].mean()) if ends.any() else 0.0)
        rows.append({"clip_id": c.clip_id, "visibility": vis, "contact_clarity": clarity})
    return rows


def grade(x, cuts):
    return np.where(x >= cuts[1], 1.0, np.where(x >= cuts[0], 0.5, 0.0))


def main():
    if sys.argv[1:2] == ["--selftest"]:
        # code path only: an OpenGraph pilot episode read as one window starting at frame 0
        rec, _ = frame_table(ROOT / "out" / "sam10" / "episode_000053", "_events.jsonl.gz", 741)
        fake = pd.DataFrame([{"clip_id": "selftest_right", "hand": "right", "clip_start": 150, "clip_frames": 300,
                              "contact_start": 180, "contact_frames": 240}])
        print(clip_values(rec, 0, fake))
        return
    w = pd.read_csv(N2 / "sam_windows.csv")
    clips = pd.read_csv(N2 / "clips_final.csv").set_index("clip_id")
    rows = []
    for r in w.itertuples():
        d = N2 / "sam" / r.window_id
        if not all((d / f"sam_{p}_events.jsonl.gz").exists() for p in ("hand", "small_object")):
            continue
        rec, _ = frame_table(d, "_events.jsonl.gz", int(r.frames_10fps), (SIZE, SIZE))
        rows += clip_values(rec, int(r.start), clips.loc[json.loads(r.clips)].reset_index())
    v = pd.DataFrame(rows).merge(clips[["role"]], left_on="clip_id", right_index=True)
    tr = v[v.role == "score_train"]
    for col in ("visibility", "contact_clarity"):
        v[f"grade_{col}"] = grade(v[col].values, np.percentile(tr[col], [33.3, 66.7]))
    v.drop(columns="role").to_csv(N2 / "sam_checks.csv", index=False)
    print(v.groupby("role")[["visibility", "contact_clarity"]].describe().round(3))


if __name__ == "__main__":
    main()
