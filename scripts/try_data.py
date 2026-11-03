"""Data for the Try page (web/try.html): six held-out clips already scored in the final run. No API call.

    .venv/bin/python scripts/try_data.py

Refits the same score as number2_score.py --final to get the intercept and the
full weights, checks that intercept + weights x grades gives back each clip's
score in out/number2/final/number2_final_scores.csv, and writes
web/assets/try/data.json with the clips' videos cut from the HOT3D source as the
clips were cut (contact from motion capture, one second before and after).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA  # noqa: E402
from number2_score import assemble, complete_items  # noqa: E402
from viewer_data import LABELS, cut, muse_reply  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "out" / "number2"
OUT = ROOT / "web" / "assets" / "try"
# four high scored clips a robot hand can copy, two low scored clips it cannot. Picked so that the
# touch window from motion capture matches the grasp a person sees in the video.
CLIPS = ["P0001_15c4300c_left_01513", "P0001_4bf4e21a_left_01577", "P0001_9b6feab7_left_01691",
         "P0010_1573a837_left_01933", "P0001_a9d6c83d_right_01403", "P0001_f6cc0cc8_left_01665"]
# where each grade comes from
SOURCE = {"grade_camera_stability": "video", "grade_completeness": "video", "grade_hand_shape_fit": "video",
          "grade_visibility": "sam", "grade_contact_clarity": "sam"}


def main():
    d, items = assemble(check=False)
    train, test = d[d.role == "score_train"].copy(), d[d.role == "held_out"].copy()
    train, test, items, _ = complete_items(train, test, items, final=True)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(train[items], train.copyable)  # same as fit_score
    w, b = m.coef_[0], float(m.intercept_[0])
    final = pd.read_csv(N2 / "final" / "number2_final_scores.csv").set_index("clip_id").our_score
    fail = pd.read_csv(ROOT / "benchmark" / "number2" / "clips.csv").set_index("clip_id").failed_condition
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for cid in CLIPS:
        r = test[test.clip_id == cid].iloc[0]
        x = r[items].astype(float).values
        score = 100 / (1 + np.exp(-(b + float(w @ x))))
        if abs(score - 100 * final[cid]) > 0.05:
            raise SystemExit(f"{cid}: rebuilt score {score:.2f} is not the final {100 * final[cid]:.2f}")
        start, frames = int(r.clip_start), int(r.clip_frames)
        cut(next((ARIA / r.sequence).glob("*preview_rgb.mp4")), start, frames, OUT / f"{cid}.mp4")
        check, base = muse_reply("checklist", cid), muse_reply("baseline", cid)
        clips.append({
            "id": cid, "video": f"assets/try/{cid}.mp4", "object": r.object, "hand": r.hand,
            "person": r.person, "duration": round(frames / 30, 2),
            "contact": [round((int(r.contact_start) - start) / 30, 2),
                        round((int(r.contact_start) + int(r.contact_frames) - start) / 30, 2)],
            "score": round(score, 1), "copyable": bool(r.copyable),
            "failed": None if r.copyable else fail[cid],
            "grades": [{"key": c, "name": LABELS[c], "source": SOURCE.get(c, "muse"), "value": float(xi),
                        "weight": round(float(wi), 3)} for c, xi, wi in zip(items, x, w)],
            "muse_reason": check["reason"] if check else "",
            "muse_alone": {"probability": base["copyable_probability"], "reason": base["reason"]} if base else None})
        print(f"{cid} {r.object:16s} score {score:5.1f} copyable {bool(r.copyable)}")
    out = {"intercept": round(b, 3), "presets": {"Strict": 70, "Balanced": 50, "Volume": 30}, "clips": clips}
    (OUT / "data.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "data.json", "intercept", round(b, 3))


if __name__ == "__main__":
    main()
