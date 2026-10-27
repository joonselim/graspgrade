"""Contact sheet: one video crop per HOT3D object at a grasp, for judging object items by eye. No API call.

    python scripts/hot3d_object_sheet.py

Takes the first tuning clip of each object, crops the preview frame around
the object at the contact frame, and tiles all objects into
docs/experiments/img/exp19_object_sheet.jpg. Labels are not shown.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA, ORIENT, pose  # noqa: E402
from hot3d_copyable import Surface  # noqa: E402
from hot3d_hands import camera, to_pixels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    clips = pd.read_csv(ROOT / "out" / "hot3d_copyable" / "tune_v2_clips.csv")
    first = clips.sort_values(["object", "sequence", "clip_start"]).groupby("object").head(1)
    cells = []
    for _, e in first.iterrows():
        seq, k = e.sequence, int(e.frame)
        gt = ARIA / seq / "ground_truth"
        meta = json.loads((gt / "metadata.json").read_text())
        uid = int(dict(zip(meta["object_names"], meta["object_uids"]))[e.object])
        qa = pd.read_csv(gt / "masks" / "mask_qa_pass.csv")
        ts = np.sort(qa[qa["stream_id"] == "214-1"]["timestamp[ns]"].values)
        o = pd.read_csv(gt / "dynamic_objects.csv").set_index(["timestamp[ns]", "object_uid"]).loc[(ts[k], uid)]
        T = pose(o[["q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z"]].values, o[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].values)
        pts = (T[:3, :3] @ Surface(uid, 2000).pts.T).T + T[:3, 3]
        params, T_dc, head = camera(seq)
        uv, z = to_pixels(pts, ts[k], params, T_dc, head)
        u, v = ORIENT["rot90_cw"](uv[:, 0], uv[:, 1])
        ok = z > 0
        cap = cv2.VideoCapture(str(next((ARIA / seq).glob("*preview_rgb.mp4"))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        _, img = cap.read()
        cap.release()
        cx, cy = float(np.median(u[ok])), float(np.median(v[ok]))
        span = max(np.ptp(u[ok]), np.ptp(v[ok]), 200) * 0.75
        x0, y0 = int(np.clip(cx - span, 0, 1408 - 2 * span)), int(np.clip(cy - span, 0, 1408 - 2 * span))
        crop = cv2.resize(img[y0:y0 + int(2 * span), x0:x0 + int(2 * span)], (260, 260))
        for col, w in (((0, 0, 0), 5), ((255, 255, 255), 1)):
            cv2.putText(crop, e.object, (6, 250), 0, 0.6, col, w)
        cells.append(crop)
    while len(cells) % 6:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]
    cv2.imwrite(str(ROOT / "docs" / "experiments" / "img" / "exp19_object_sheet.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
