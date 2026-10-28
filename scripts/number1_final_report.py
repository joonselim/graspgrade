"""Number 1 final report on held-out: the rows the plan asks for that `muse_contact.py score` does not print.

    python scripts/number1_final_report.py

Run after `muse_contact.py score --split held_out --segments out/contact/rule_segments_applied.json`
(copied to out/number1/final/arms_score_held_out.json). It rebuilds the same rows with the same shifts
and checks that the pooled numbers match that file, then adds:
- the reference rows "always in contact" and "never in contact" (plan, What we compare),
- every row per hand (plan, Metrics),
- the start and end delay of arm B against the glove (plan, Metrics),
- paired bootstraps between arms (C - B, A + shift - C + shift). Added after the plan, descriptive only,
- the left hand in the 9 short held-out episodes where arms A and B disagree (EXP-34). Added after, descriptive.
No API call. Cached answers only.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muse_contact import (N1, ROOT, SIDES, TARGETS, arm_rows, load_a, load_c, paired_bootstrap,  # noqa: E402
                          score_rows)
from contact_cut import ground_truth  # noqa: E402

FINAL = N1 / "final"
SHORT = [18, 24, 27, 39, 40, 41, 42, 43, 45]  # EXP-34: arm A marks the left hand touching in at most 14% of frames
KEYS = ("boundary_f1_5", "precision_5", "recall_5", "boundary_f1_15", "contact_iou", "start_error_median",
        "end_error_median", "boundaries_pred", "boundaries_ref")


def main():
    eps = json.loads((ROOT / "splits" / "opengraph.json").read_text())["held_out"]
    done = json.loads((FINAL / "arms_score_held_out.json").read_text())
    a_shift, c_shift = (tuple(done["shifts"][k]) for k in ("arm_A", "arm_C"))
    segs = json.loads((ROOT / "out" / "contact" / "rule_segments_applied.json").read_text())
    gt = ground_truth(eps)
    rows = arm_rows(eps, gt, load_a(eps), load_c(segs), segs, a_shift, c_shift)
    rows["always_in_contact"] = {(ep, s): [(0, len(gt[ep]["label"]["left"]))] for ep in eps for s in SIDES}
    rows["never_in_contact"] = {(ep, s): [] for ep in eps for s in SIDES}

    out = {"shifts": {"arm_A": a_shift, "arm_C": c_shift}, "pooled": {}, "per_hand": {}}
    for t in TARGETS:
        out["pooled"][t] = {name: {k: v for k, v in score_rows(r, gt, t).result().items() if k in KEYS}
                            for name, r in rows.items()}
        out["per_hand"][t] = {s: {name: {k: v for k, v in score_rows({key: x for key, x in r.items() if key[1] == s},
                                                                     gt, t).result().items() if k in KEYS}
                                  for name, r in rows.items()} for s in SIDES}
        for name, r in done["scores"][t].items():  # the same numbers as the score run
            for k in ("boundary_f1_5", "boundary_f1_15", "contact_iou"):
                assert out["pooled"][t][name][k] == r[k], (t, name, k)
    out["arm_B_delay_frames"] = {k: out["pooled"]["tactile_clean"]["arm_B"][k]
                                 for k in ("start_error_median", "end_error_median")}
    out["paired_bootstrap_added_after"] = [paired_bootstrap(rows, gt, a, b, eps, tol=tol)
                                           for a, b in (("arm_B", "arm_C"), ("arm_C_shift", "arm_A_shift"))
                                           for tol in (5, 15)]
    short = {}
    for name in ("arm_A_shift", "arm_B", "arm_C_shift", "always_in_contact"):
        r = {(ep, "left"): rows[name][(ep, "left")] for ep in SHORT}
        short[name] = {k: v for k, v in score_rows(r, gt, "tactile_clean").result().items() if k in KEYS}
    lit = [float(np.mean(gt[ep]["tactile_clean"]["left"])) for ep in SHORT]
    short["glove_left_contact_share"] = {"mean": round(float(np.mean(lit)), 3), "min": round(min(lit), 3),
                                         "max": round(max(lit), 3)}
    out["short_episodes_left_hand_added_after"] = short
    (FINAL / "number1_final_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
