"""Data for the Try page (web/try.html): six held-out clips already scored in the final run. No API call.

    .venv/bin/python scripts/try_data.py

Refits the same score as number2_score.py --final (through viewer_data.py) to get
the intercept and the full weights, checks that intercept + weights x grades gives
back each clip's final score, and writes web/assets/try/data.json with the clips'
videos (copied from viewer/media, run scripts/viewer_data.py first). The page only
replays these stored results, it runs nothing.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from number2_score import assemble, complete_items  # noqa: E402
from viewer_data import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "assets" / "try"
# four high scored clips a robot hand can copy, two low scored clips it cannot
CLIPS = ["P0010_1573a837_right_02542", "P0001_4bf4e21a_left_01577", "P0001_9b6feab7_left_01691",
         "P0010_1573a837_left_01933", "P0012_d6272ce1_left_00290", "P0001_15c4300c_left_01513"]
# where each grade comes from
SOURCE = {"grade_camera_stability": "video", "grade_completeness": "video", "grade_hand_shape_fit": "video",
          "grade_visibility": "sam", "grade_contact_clarity": "sam"}


def main():
    d, items = assemble(check=False)
    train, test = d[d.role == "score_train"].copy(), d[d.role == "held_out"].copy()
    train, test, items, _ = complete_items(train, test, items, final=True)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(train[items], train.copyable)  # same as fit_score
    w, b = m.coef_[0], float(m.intercept_[0])
    view = {c["id"]: c for c in json.loads((ROOT / "viewer" / "data" / "number2.json").read_text())["clips"]}
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for cid in CLIPS:
        r = test[test.clip_id == cid].iloc[0]
        x = r[items].astype(float).values
        score = 100 / (1 + np.exp(-(b + float(w @ x))))
        v = view[cid]
        if abs(score - v["score"]) > 0.06:
            raise SystemExit(f"{cid}: rebuilt score {score:.2f} is not the final {v['score']}")
        shutil.copy(ROOT / "viewer" / v["video"], OUT / f"{cid}.mp4")
        clips.append({
            "id": cid, "video": f"assets/try/{cid}.mp4", "object": v["object"], "hand": v["hand"],
            "person": v["person"], "duration": round(v["duration"], 2),
            "contact": [round(t, 2) for t in v["contact"]], "score": round(score, 1),
            "copyable": v["copyable_truth"], "failed": v["failed_condition"],
            "grades": [{"key": c, "name": LABELS[c], "source": SOURCE.get(c, "muse"), "value": float(xi),
                        "weight": round(float(wi), 3)} for c, xi, wi in zip(items, x, w)],
            "muse_reason": v["muse_reason"], "muse_alone": v["muse_alone"]})
        print(f"{cid} {v['object']:16s} score {score:5.1f} copyable {v['copyable_truth']}")
    out = {"intercept": round(b, 3), "presets": {"Strict": 70, "Balanced": 50, "Volume": 30},
           "held_out": {"clips": 1426, "auc_ours": 0.624, "auc_muse_alone": 0.501}, "clips": clips}
    (OUT / "data.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "data.json", "intercept", round(b, 3))


if __name__ == "__main__":
    main()
