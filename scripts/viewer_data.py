"""Build the viewer data from the final held-out results (EXP-37, EXP-40). No API call.

    python scripts/viewer_data.py

Writes viewer/data/number2.json, viewer/data/number1.json and short mp4s in
viewer/media/ (not in git). Number 2: held-out clips, the score learned on the
400 score_train clips (the same model as number2_score.py --final, checked
against out/number2/final/number2_final_scores.csv), the twelve grades, the
Muse Spark checklist reason and Muse Spark alone. Number 1: the 22 held-out
episodes with the cleaned glove, arm A + shift, arm B and arm C + shift, with
the shifts of the final run (out/number1/final/arms_score_held_out.json).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hot3d_calib_check import ARIA  # noqa: E402
from number2_score import assemble, complete_items, fit_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEW = ROOT / "viewer"
N2 = ROOT / "out" / "number2"
N1 = ROOT / "out" / "number1"
N_EACH = 15  # highest and lowest scored held-out clips
LABELS = {"grade_camera_stability": "Camera stability", "grade_completeness": "Action completeness",
          "grade_hand_shape_fit": "Hand shape fit", "finger_occlusion": "Finger visibility",
          "object_size": "Object size next to the hand", "object_shape": "Object shape",
          "object_thickness": "Object thickness", "object_width": "Width to wrap", "object_handle": "Handle",
          "free_space": "Free space around the object", "grade_visibility": "Visibility",
          "grade_contact_clarity": "Contact clarity"}
ARMS = {"arm_A_shift": "A + shift, Muse Spark alone", "arm_B": "B, SAM rule", "arm_C_shift": "C + shift, SAM rule + Muse Spark"}


def cut(src, start, frames, dst, fps=30, size=704, crf=26):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start / fps:.4f}", "-i", str(src),
                    "-frames:v", str(frames), "-vf", f"scale={size}:-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def muse_reply(task, clip_id):
    p = N2 / "muse" / task / f"{clip_id}.json"
    return json.loads(p.read_text())["parsed"] if p.exists() else None


def number2():
    d, items = assemble(check=False)
    train, test = d[d.role == "score_train"].copy(), d[d.role == "held_out"].copy()
    train, test, items, _ = complete_items(train, test, items, final=True)
    test["our_score"], weights = fit_score(train, test, items)
    final = pd.read_csv(N2 / "final" / "number2_final_scores.csv")
    same = test[["clip_id", "our_score"]].merge(final, on="clip_id", suffixes=("", "_final"))
    if len(same) != len(test) or not np.allclose(same.our_score, same.our_score_final):
        raise SystemExit("viewer scores differ from the final run")
    fail = pd.read_csv(ROOT / "benchmark" / "number2" / "clips.csv")[["clip_id", "failed_condition"]]
    test = test.merge(fail, on="clip_id", how="left")
    test["score"] = 100 * test.our_score
    pick = pd.concat([test.nlargest(N_EACH, "score"), test.nsmallest(N_EACH, "score")])
    clips = []
    for r in pick.sort_values("score", ascending=False).itertuples():
        src = next((ARIA / r.sequence).glob("*preview_rgb.mp4"))
        name = f"{r.clip_id}.mp4"
        cut(src, int(r.clip_start), int(r.clip_frames), VIEW / "media" / name)
        check, base = muse_reply("checklist", r.clip_id), muse_reply("baseline", r.clip_id)
        clips.append({
            "id": r.clip_id, "video": f"media/{name}", "object": r.object, "hand": r.hand, "person": r.person,
            "score": round(float(r.score), 1), "copyable_truth": bool(r.copyable),
            "failed_condition": None if r.copyable else r.failed_condition,
            "contact": [(int(r.contact_start) - int(r.clip_start)) / 30,
                        (int(r.contact_start) + int(r.contact_frames) - int(r.clip_start)) / 30],
            "duration": int(r.clip_frames) / 30,
            "grades": [{"name": LABELS.get(c, c), "value": float(getattr(r, c)), "weight": round(weights[c], 2)}
                       for c in items],
            "muse_reason": check["reason"] if check else "",
            "muse_alone": {"probability": base["copyable_probability"], "reason": base["reason"]} if base else None})
    f = json.loads((N2 / "final" / "number2_final.json").read_text())
    summary = (f"Held-out, {f['our_score']['clips']} clips, 3 persons. AUC against the copyable label: "
               f"our score {f['our_score']['auc']:.3f}, Muse Spark alone {f['muse_alone']['auc']:.3f}, "
               f"object name reference {f['object_reference']['auc']:.3f}, objects never seen in training "
               f"{f['unseen_objects']['our_score']['auc']:.3f}.")
    out = {"note": f"The {N_EACH} highest and {N_EACH} lowest scored held-out clips. Score learned on the 400 "
                   "score_train clips. Ground truth: copyable v2 (5 mm), from motion capture and a robot hand model.",
           "summary": summary, "presets": {"Strict": 70, "Balanced": 50, "Volume": 30}, "clips": clips}
    (VIEW / "data").mkdir(parents=True, exist_ok=True)
    (VIEW / "data" / "number2.json").write_text(json.dumps(out, indent=1))
    return len(clips)


def number1():
    from contact_cut import SIDES, ground_truth, segments
    from muse_contact import arm_rows, load_a, load_c, score_rows
    eps = json.loads((ROOT / "splits" / "opengraph.json").read_text())["held_out"]
    done = json.loads((N1 / "final" / "arms_score_held_out.json").read_text())
    a_shift, c_shift = (tuple(done["shifts"][k]) for k in ("arm_A", "arm_C"))
    segs = json.loads((ROOT / "out" / "contact" / "rule_segments_applied.json").read_text())
    gt = ground_truth(eps)
    rows = arm_rows(eps, gt, load_a(eps), load_c(segs), segs, a_shift, c_shift)
    for name, r in done["scores"]["tactile_clean"].items():  # the same rows as the final run
        if name in ARMS:
            assert round(score_rows(rows[name], gt, "tactile_clean").result()["boundary_f1_5"], 3) == r["boundary_f1_5"]
    episodes = []
    for ep in eps:
        n30 = segs[str(ep)]["n30"]
        name = f"episode_{ep:06d}.mp4"
        cut(ROOT / "data" / "episodes_720p" / name, 0, n30, VIEW / "media" / name, size=480, crf=30)
        scores = {k: score_rows(rows[k], gt, "tactile_clean", {ep}).result() for k in ARMS}
        episodes.append({"episode": ep, "video": f"media/{name}", "duration": n30 / 30,
                         "scores": {k: {m: s[m] for m in ("boundary_f1_5", "boundary_f1_15", "contact_iou")}
                                    for k, s in scores.items()},
                         "hands": {s: {"glove": [[a / 30, b / 30] for a, b in segments(gt[ep]["tactile_clean"][s])],
                                       **{k: [[a / 30, b / 30] for a, b in rows[k][(ep, s)]] for k in ARMS}}
                                   for s in SIDES}})
    t = done["scores"]["tactile_clean"]
    summary = ("Held-out, 22 episodes. Boundary F1 at ±5 frames / contact IoU against the cleaned glove: " +
               ", ".join(f"{ARMS[k].split(',')[0]} {t[k]['boundary_f1_5']:.3f} / {t[k]['contact_iou']:.3f}"
                         for k in ARMS) + ".")
    out = {"note": "Glove = cleaned tactile ground truth. Shifts learned on tuning episodes only.",
           "summary": summary, "arms": ARMS, "episodes": episodes}
    (VIEW / "data" / "number1.json").write_text(json.dumps(out, indent=1))
    return len(episodes)


def main():
    print("number 2 clips:", number2())
    print("number 1 episodes:", number1())


if __name__ == "__main__":
    main()
