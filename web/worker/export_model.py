"""Write web/worker/model.json, everything the worker needs to score a new clip.

The final number 2 run (scripts/number2_score.py --final, EXP-40) saved only the
12 weights, rounded. This refits the same models from the cached tuning files and
checks the weights against the saved ones, so the worker scores a clip the same
way. It only reads; nothing in out/ changes. Run in WSL from ~/graspgrade:

    .venv/bin/python web/worker/export_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import number2_score as ns  # noqa: E402
from hot3d_hand_shape import FEATS  # noqa: E402

OUT = Path(__file__).resolve().parent / "model.json"
ITEMS = ["grade_camera_stability", "grade_completeness", "grade_hand_shape_fit", "grade_visibility",
         "grade_contact_clarity"] + [f for f in ns.MUSE_ITEMS]


def main():
    d = ns.assemble(False)[0]

    # the score: L2 logistic regression on the 400 score_train clips, as in fit_score
    tr = d[d.role == "score_train"].dropna(subset=ITEMS)
    m = LogisticRegression(C=0.5, max_iter=2000).fit(tr[ITEMS], tr.copyable)
    weights = dict(zip(ITEMS, m.coef_[0].tolist()))
    saved = json.loads((REPO / "out" / "number2" / "final" / "number2_final.json").read_text())
    for k, v in saved["weights"].items():  # the saved weights are rounded to 3 decimals
        assert abs(weights[k] - v) < 6e-4, (k, weights[k], v)
    print("score_train clips", len(tr), "weights match the final run, intercept", round(float(m.intercept_[0]), 4))

    # the hand shape check: MediaPipe features -> probability -> tertile grade, item_train clips
    hs = pd.read_csv(ns.N2 / "hand_shape_video.csv")
    x = d[["clip_id", "role", "copyable"]].merge(hs, on="clip_id")
    cols = [f"v_{c}" for c in FEATS]
    is_train = x.role == ns.HAND_SHAPE_ROLE
    fill = x[is_train][cols].median()
    x[cols] = x[cols].fillna(fill)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(x[is_train][cols],
                                                                                         x[is_train].copyable)
    p = pipe.predict_proba(x[cols])[:, 1]
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]

    # SAM cut points from the score_train clips, as in number2_sam_checks
    sam = pd.read_csv(ns.N2 / "sam_checks.csv").merge(d[["clip_id", "role"]], on="clip_id")
    st = sam[sam.role == "score_train"]
    cam = json.loads((REPO / "out" / "hot3d_checks" / "summary.json").read_text())["cuts_from_tuning_terciles"]["camera"]

    model = {
        "note": "Refit from the cached tuning files by web/worker/export_model.py. Same models as EXP-40.",
        "items": ITEMS,
        "intercept": float(m.intercept_[0]),
        "weights": weights,
        "camera_cuts_px": cam,  # lower motion is better: grade 1 below cuts[0], 0.5 below cuts[1]
        "visibility_cuts": np.percentile(st.visibility, [33.3, 66.7]).tolist(),
        "contact_clarity_cuts": np.percentile(st.contact_clarity, [33.3, 66.7]).tolist(),
        "hand_shape": {
            "features": FEATS, "fill": fill.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "cuts": np.percentile(p[is_train.values], [33.3, 66.7]).tolist(),
        },
        "contact_shift_frames": json.loads((REPO / "out" / "number1" / "final" / "arms_score_held_out.json")
                                           .read_text())["shifts"]["arm_A"],
    }
    OUT.write_text(json.dumps(model, indent=1))
    print("wrote", OUT)
    for k in ("camera_cuts_px", "visibility_cuts", "contact_clarity_cuts", "contact_shift_frames"):
        print(k, model[k])
    print("hand shape cuts", np.round(model["hand_shape"]["cuts"], 4).tolist())


if __name__ == "__main__":
    main()
