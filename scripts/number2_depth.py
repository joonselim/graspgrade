"""Number 2 held-out rows at the depth limits 2 mm and 10 mm (plan "Depth limit on held-out", EXP-41).

    python scripts/number2_depth.py

Everything learned stays on the 5 mm label of score_train, as in the final run
(number2_score.py --final). Only the held-out label the scores are judged
against changes. The object reference takes the tuning share at the same depth.
The 5 mm rows must equal out/number2/final/number2_final.json, or the script stops.
No API call. Writes out/number2/final/number2_depth.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import number2_score as ns  # noqa: E402

FINAL = ns.N2 / "final"
LABELS = {"5mm": "copyable", "2mm": "copyable_depth_2mm", "10mm": "copyable_depth_10mm"}


def with_label(t, col, bench):
    t = t.drop(columns="copyable").merge(bench[["clip_id", col]], on="clip_id", how="left")
    if t[col].isna().any():
        raise SystemExit(f"{col} missing for {int(t[col].isna().sum())} clips")
    return t.rename(columns={col: "copyable"}).assign(copyable=lambda x: x.copyable.astype(int))


def main():
    d, items = ns.assemble(False)
    train, test = d[d.role == "score_train"].copy(), d[d.role == "held_out"].copy()
    train, test, items, dropped = ns.complete_items(train, test, items, final=True)
    d = d[~d.clip_id.isin(dropped["train"] + dropped["test"])]
    test["our_score"], _ = ns.fit_score(train, test, items)
    test["plug_in"], _ = ns.fit_score(train, test, ["muse_baseline"] + items)
    grades = [c for c in ns.MUSE_ITEMS if c in items]
    test["R2_equal_weights"] = test[grades].mean(axis=1)
    test["R3_learned"], _ = ns.fit_score(train, test, grades)
    unseen = ns.unseen_scores(d, items, "score_train", "held_out")
    bench = pd.read_csv(ns.BENCH_CLIPS)[["clip_id"] + list(LABELS.values())]

    out = {"note": "scores learned on the 5 mm label (score_train), held-out judged at each depth", "depths": {}}
    for depth, col in LABELS.items():
        t, u, dd = (with_label(x, col, bench) for x in (test, unseen, d))
        t["object_reference"] = ns.object_reference(dd, t, "held_out")
        r = {"copyable_share": round(float(t.copyable.mean()), 3),
             "our_score": ns.metrics(t, "our_score"), "muse_alone": ns.metrics(t, "muse_baseline"),
             "object_reference": ns.metrics(t, "object_reference"), "muse_plus_checks": ns.metrics(t, "plug_in"),
             "unseen_objects": {"our_score": ns.metrics(u, "our_score"),
                                "muse_plus_checks": ns.metrics(u.dropna(subset=["plug_in"]), "plug_in")},
             "R2_equal_weights": ns.metrics(t, "R2_equal_weights"), "R3_learned": ns.metrics(t, "R3_learned"),
             "paired": [ns.paired_auc(t, "muse_baseline", "R2_equal_weights"),
                        ns.paired_auc(t, "muse_baseline", "R3_learned"),
                        ns.paired_auc(t, "R2_equal_weights", "R3_learned"),
                        ns.paired_auc(t, "R3_learned", "our_score"),
                        ns.paired_auc(t, "muse_baseline", "our_score")],
             "our_score_within_object": round(ns.pair_auc(t.copyable.values, t.our_score.values,
                                                          t.object.values)[0], 3)}
        out["depths"][depth] = r

    f = json.loads((FINAL / "number2_final.json").read_text())
    r5 = out["depths"]["5mm"]
    same = [r5["our_score"] == f["our_score"], r5["muse_alone"] == f["muse_alone"],
            r5["object_reference"] == f["object_reference"], r5["muse_plus_checks"] == f["muse_plus_checks"],
            r5["unseen_objects"]["our_score"] == f["unseen_objects"]["our_score"],
            r5["R2_equal_weights"] == f["checklist_vs_learning"]["R2_equal_weights"],
            r5["R3_learned"] == f["checklist_vs_learning"]["R3_learned"],
            r5["our_score_within_object"] == f["extra"]["inside_each_object"]["our_score"]["within_object"]]
    if not all(same):
        raise SystemExit(f"5 mm rows differ from the final run: {same}")
    (FINAL / "number2_depth.json").write_text(json.dumps(out, indent=1))
    for depth, r in out["depths"].items():
        print(depth, "copyable", r["copyable_share"], {k: r[k]["auc"] for k in
              ("our_score", "muse_alone", "object_reference", "muse_plus_checks", "R2_equal_weights", "R3_learned")},
              "unseen", r["unseen_objects"]["our_score"]["auc"], "within", r["our_score_within_object"])
        for p in r["paired"]:
            print("   ", p["rows"], p["difference"], p["ci95_by_sequence"], p["gain"])


if __name__ == "__main__":
    main()
