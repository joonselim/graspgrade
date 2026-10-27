"""Build the viewer demo data from tuning clips only (no held-out results). No API call.

    python scripts/viewer_data.py

Writes viewer/data/number2.json, viewer/data/number1.json and short mp4s in
viewer/media/. Until the paid runs exist, the number 2 score is the --check
model of number2_score.py (stand-in object grades) and Muse Spark fields say
"pending". Number 1 shows the pilot episodes with the cleaned glove and arm B.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA  # noqa: E402
from number2_score import assemble  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEW = ROOT / "viewer"
N_CLIPS = 24
LABELS = {"grade_camera_stability": "Camera stability", "grade_completeness": "Action completeness",
          "grade_hand_shape_fit": "Hand shape fit", "finger_occlusion": "Finger visibility",
          "object_size": "Object size next to the hand", "object_shape": "Object shape",
          "object_thickness": "Object thickness", "object_width": "Width to wrap", "object_handle": "Handle",
          "free_space": "Free space around the object", "grade_visibility": "Visibility",
          "grade_contact_clarity": "Contact clarity"}


def cut(src, start, frames, dst, fps=30, size=704):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start / fps:.4f}", "-i", str(src),
                    "-frames:v", str(frames), "-vf", f"scale={size}:-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def number2():
    d, items = assemble(check=True)
    train, show = d[d.role == "item_train"], d[d.role == "score_train"].copy()
    items = [c for c in items if train[c].notna().all() and show[c].notna().all()]
    m = LogisticRegression(C=0.5, max_iter=2000).fit(train[items], train.copyable)
    show["score"] = 100 * m.predict_proba(show[items])[:, 1]
    weights = dict(zip(items, np.round(m.coef_[0], 2).tolist()))
    pick = pd.concat([show.nlargest(N_CLIPS // 2, "score"), show.nsmallest(N_CLIPS // 2, "score")])
    clips = []
    for r in pick.sort_values("score", ascending=False).itertuples():
        src = next((ARIA / r.sequence).glob("*preview_rgb.mp4"))
        name = f"{r.clip_id}.mp4"
        cut(src, int(r.clip_start), int(r.clip_frames), VIEW / "media" / name)
        clips.append({
            "id": r.clip_id, "video": f"media/{name}", "object": r.object, "hand": r.hand, "person": r.person,
            "score": round(float(r.score), 1), "copyable_truth": bool(r.copyable),
            "contact": [(int(r.contact_start) - int(r.clip_start)) / 30,
                        (int(r.contact_start) + int(r.contact_frames) - int(r.clip_start)) / 30],
            "duration": int(r.clip_frames) / 30,
            "grades": [{"name": LABELS.get(c, c), "value": float(getattr(r, c)), "weight": weights[c]} for c in items],
            "muse_reason": "pending (Muse Spark not run yet)"})
    out = {"note": "Demo on tuning clips. Score = stand-in model (no Muse Spark yet). Not a result.",
           "presets": {"Strict": 70, "Balanced": 50, "Volume": 30}, "clips": clips}
    (VIEW / "data").mkdir(parents=True, exist_ok=True)
    (VIEW / "data" / "number2.json").write_text(json.dumps(out, indent=1))
    return len(clips)


def number1():
    from contact_cut import SIDES, ground_truth, segments
    segs = json.loads((ROOT / "out" / "contact" / "rule_segments.json").read_text())
    eps = [int(e) for e in segs]
    gt = ground_truth(eps)
    episodes = []
    for ep in eps:
        n30 = segs[str(ep)]["n30"]
        name = f"episode_{ep:06d}.mp4"
        cut(ROOT / "data" / "episodes_720p" / name, 0, n30, VIEW / "media" / name, size=640)
        episodes.append({"episode": ep, "video": f"media/{name}", "duration": n30 / 30, "hands": {
            s: {"glove": [[a / 30, b / 30] for a, b in segments(gt[ep]["tactile_clean"][s])],
                "arm_B": [[a / 30, b / 30] for a, b in segs[str(ep)].get(s, [])]} for s in SIDES}})
    out = {"note": "Pilot tuning episodes. Glove = cleaned tactile ground truth. Arm B = SAM rule chosen on the other episodes.",
           "episodes": episodes}
    (VIEW / "data" / "number1.json").write_text(json.dumps(out, indent=1))
    return len(episodes)


def main():
    print("number 2 clips:", number2())
    print("number 1 episodes:", number1())


if __name__ == "__main__":
    main()
