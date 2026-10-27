"""Fix the tuning / held-out splits. Run once, commit splits/*.json, never re-roll.

    python scripts/make_splits.py

OpenGraph: by episode, 50 tuning and 22 held-out.
HOT3D Aria: by person. Only sequences with have_hand_object_pose_gt are usable.
62 of 198 Aria sequences (persons P0004, P0005, P0008, P0016, P0020) ship with
empty hand and object pose files, so they are dropped. That leaves 9 persons,
of which 3 are held out.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261026  # hackathon start date
HELD_OUT_PERSONS = 3


def main():
    rng = np.random.default_rng(SEED)
    out = ROOT / "splits"
    out.mkdir(exist_ok=True)

    eps = rng.permutation(72).tolist()
    og = {"seed": SEED, "unit": "episode_index",
          "tune": sorted(eps[:50]), "held_out": sorted(eps[50:])}
    (out / "opengraph.json").write_text(json.dumps(og, indent=1))

    by_person = defaultdict(list)
    dropped = []
    for d in sorted((ROOT / "data/hot3d/aria").iterdir()):
        m = json.loads((d / "ground_truth/metadata.json").read_text())
        (by_person[m["participant_id"]] if m["have_hand_object_pose_gt"] else dropped).append(d.name)
    persons = sorted(by_person)
    held = sorted(rng.choice(persons, HELD_OUT_PERSONS, replace=False).tolist())
    tune = [p for p in persons if p not in held]
    h3 = {
        "seed": SEED, "unit": "participant_id", "dataset": "Hot3DAria",
        "tune_persons": tune, "held_out_persons": held,
        "tune": sorted(s for p in tune for s in by_person[p]),
        "held_out": sorted(s for p in held for s in by_person[p]),
        "dropped_no_gt": dropped,
    }
    (out / "hot3d_aria.json").write_text(json.dumps(h3, indent=1))

    print(f"OpenGraph: tune {len(og['tune'])}, held_out {len(og['held_out'])}")
    print(f"  held_out episodes: {og['held_out']}")
    print(f"HOT3D Aria: tune persons {tune} -> {len(h3['tune'])} seqs")
    print(f"  held_out persons {held} -> {len(h3['held_out'])} seqs, dropped {len(dropped)}")


if __name__ == "__main__":
    main()
