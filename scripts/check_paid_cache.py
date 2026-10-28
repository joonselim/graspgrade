"""Code check of the paid-call caches with fake clients. No API call, nothing in out/ is touched.

    python scripts/check_paid_cache.py

Checks, in a temporary folder:
1. SAM (sam_episodes.send): only a "completed" response counts as done.
2. Muse number 2 (muse_hot3d.call): a reply that does not parse is kept
   under unparsed/, is not read by number2_score.muse_items, and is sent again.
3. Muse arm A and arm C (muse_contact.a_call, c_call): the same for a reply
   without every frame (A) or without a frame number (C). Frame 0 (no
   change) is a valid arm C answer.
4. Arm C picks (muse_contact.load_c): a cached call for other frames than
   the current arm B segments stops the run.
5. number2_score.complete_items: in the final run a missing grade or a
   missing Muse alone answer stops the run, in check mode the item is left out.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import muse_contact  # noqa: E402
import muse_hot3d  # noqa: E402
import number2_score  # noqa: E402
import sam_episodes  # noqa: E402

USAGE = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, model_dump=lambda: {"prompt_tokens": 1000})


class FakeMuse:
    """chat.completions.create returns the given replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = self.completions = self

    def create(self, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))],
                               usage=USAGE)


class FakeSam:
    """A SAM stream that ends with the given status."""

    def __init__(self, status):
        self.status = status
        self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id="file-1"))
        self.responses = self

    def create(self, **kw):
        last = {"type": f"response.{self.status}",
                "response": {"status": self.status, "usage": {"video_frames_processed": 50}}}
        return iter([SimpleNamespace(type=e["type"], model_dump=lambda e=e: e)
                     for e in ({"type": "response.created"}, last)])


def expect(name, ok):
    print(("ok    " if ok else "FAIL  ") + name)
    return ok


def check_sam(tmp):
    clip = tmp / "sam" / "clip10.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"not a video")
    res = []
    info = sam_episodes.send(FakeSam("failed"), 1, clip, "hand")
    res.append(expect("SAM failed: no done file", not (clip.parent / "sam_hand_events.jsonl.gz").exists()))
    res.append(expect("SAM failed: events kept as .failed", (clip.parent / "sam_hand_events.failed.jsonl.gz").exists()
                      and info["status"] == "failed"))
    sam_episodes.send(FakeSam("completed"), 1, clip, "hand")
    res.append(expect("SAM completed: done file", (clip.parent / "sam_hand_events.jsonl.gz").exists()))
    return res


def check_muse_hot3d(tmp):
    muse_hot3d.N2 = number2_score.N2 = tmp / "n2"
    r = SimpleNamespace(clip_id="clip_a", hand="right")
    frames = tmp / "n2" / "muse_frames" / "clip_a"
    frames.mkdir(parents=True)
    for j in range(6):
        (frames / f"f{j}.jpg").write_bytes(b"jpg")
    good = json.dumps({k: "high" for k in muse_hot3d.CHECKLIST_KEYS})
    rec = muse_hot3d.call(FakeMuse(["I cannot tell."]), "checklist", r)
    done = tmp / "n2" / "muse" / "checklist" / "clip_a.json"
    res = [expect("Muse n2 unparsed: not done", rec["parsed"] is None and not done.exists()),
           expect("Muse n2 unparsed: kept in unparsed/",
                  len(list((done.parent / "unparsed").glob("clip_a_*.json"))) == 1)]
    d = pd.DataFrame({"clip_id": ["clip_a"]})
    res.append(expect("Muse n2 unparsed: muse_items sees no grade",
                      number2_score.muse_items(d, "checklist")["object_size"].isna().all()))
    muse_hot3d.call(FakeMuse([good]), "checklist", r)
    res.append(expect("Muse n2 parsed: done and read", done.exists()
                      and number2_score.muse_items(d, "checklist")["object_size"].iloc[0] == 1.0))
    return res


def check_arm_a(tmp):
    muse_contact.N1 = tmp / "n1"
    muse_contact.read_frames = lambda ep, wanted: {k: b"jpg" for k in wanted}
    b = [0, 6, 12]
    part = json.dumps({"frames": [{"i": 1, "left": True, "right": False}, {"i": 2, "left": True, "right": False}]})
    full = json.dumps({"frames": [{"i": i, "left": True, "right": False} for i in (1, 2, 3)]})
    _, done = muse_contact.a_call(FakeMuse([part]), 999, b)
    folder = muse_contact.a_dir() / "episode_000999"
    res = [expect("arm A partial: not done", not done and not (folder / "00000.json").exists()),
           expect("arm A partial: kept in unparsed/, not read", len(list((folder / "unparsed").glob("*.json"))) == 1
                  and muse_contact.load_a([999]) == {})]
    _, done = muse_contact.a_call(FakeMuse([full]), 999, b)
    res.append(expect("arm A full: done and read", done and len(muse_contact.load_a([999])[999]) == 3))
    return res


def check_arm_c(tmp):
    segs = {"999": {"n30": 300, "left": [[50, 100]], "right": []}, "rule": {}}
    start, end = muse_contact.c_jobs(segs)
    res = []
    for reply, want in (("no json here", None), ('{"frame": 9}', None), ('{"frame": "x"}', None),
                        ('{"frame": 0, "confidence": 0.9}', 0), ('{"frame": 3, "confidence": 0.8}', 3)):
        res.append(expect(f"arm C reply {reply!r} -> {want}", muse_contact.c_pick(reply) == want))
    _, done = muse_contact.c_call(FakeMuse(["sorry"]), start)
    res.append(expect("arm C unparsed: not done, kept in unparsed/", not done and muse_contact.c_cached(start) is None
                      and len(list((muse_contact.c_key(start).parent / "unparsed").glob("*.json"))) == 1))
    muse_contact.c_call(FakeMuse(['{"frame": 3}']), start)
    muse_contact.c_call(FakeMuse(['{"frame": 0}']), end)
    picks = muse_contact.load_c(segs)
    res.append(expect("arm C picks read for the same segments",
                      picks == {(999, "left", 0, "start"): 3, (999, "left", 0, "end"): 0}))
    moved = {"999": {"n30": 300, "left": [[60, 100]], "right": []}}
    try:
        muse_contact.load_c(moved)
        stopped = False
    except SystemExit:
        stopped = True
    res.append(expect("arm C picks for changed segments: stops", stopped))
    res.append(expect("arm C with no segments: no picks", muse_contact.load_c(None) == {}))
    return res


def check_complete_items():
    train = pd.DataFrame({"a": [1.0, 0.5], "b": [1.0, 0.0], "muse_baseline": [0.2, 0.9]})
    test = pd.DataFrame({"a": [1.0, np.nan], "b": [0.5, 0.5], "muse_baseline": [0.3, 0.4]})
    res = [expect("check mode: incomplete item left out", number2_score.complete_items(train, test, ["a", "b"], False)
                  == ["b"])]
    for name, tr, te in (("missing grade", train, test),
                         ("missing Muse alone", train, test.assign(a=1.0, muse_baseline=[0.3, np.nan])),
                         ("no Muse alone column", train.drop(columns="muse_baseline"),
                          test.assign(a=1.0).drop(columns="muse_baseline"))):
        try:
            number2_score.complete_items(tr, te, ["a", "b"], True)
            stopped = False
        except SystemExit:
            stopped = True
        res.append(expect(f"final: {name} stops", stopped))
    res.append(expect("final: complete items kept",
                      number2_score.complete_items(train, test.assign(a=1.0), ["a", "b"], True) == ["a", "b"]))
    return res


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        res = check_sam(tmp) + check_muse_hot3d(tmp) + check_arm_a(tmp) + check_arm_c(tmp) + check_complete_items()
    print(f"{sum(res)} of {len(res)} passed")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
