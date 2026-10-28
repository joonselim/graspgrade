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
5. number2_score.complete_items: in the final run a clip without a Muse
   answer is dropped from every row up to 5% of a set, more stops the run, and
   any other missing grade stops it. In check mode the item is left out.
   Arm A frames without an answer count as not touching, an arm C boundary
   without a pick keeps arm B.
6. number2_score.object_reference: in the final run the object share comes
   from all tuning clips, in check mode never from the score_train test clips.
7. number2_score.pair_auc and draw_rates (comparison 6 and the extra rows):
   the AUC inside groups equals sklearn per group, rates leave out the clip's
   own person and use (copyable + 1) / (n + 2).
8. Soft object check (muse_soft): a reply that does not parse is sent once
   more, two such replies count as unsure, a call that raises is not cached,
   the metrics and verdict follow the plan, an empty label blocks sending.
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
import muse_soft  # noqa: E402
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


def stops(f):
    try:
        f()
        return False
    except SystemExit:
        return True


def check_complete_items():
    train = pd.DataFrame({"a": [1.0, 0.5], "b": [1.0, 0.0], "muse_baseline": [0.2, 0.9]})
    test = pd.DataFrame({"a": [1.0, np.nan], "b": [0.5, 0.5], "muse_baseline": [0.3, 0.4]})
    res = [expect("check mode: incomplete item left out, no clip dropped",
                  number2_score.complete_items(train, test, ["a", "b"], False)[2:] == (["b"], {}))]
    # final run: 40 clips per set, a local grade and the seven Muse grades
    items = ["local"] + number2_score.MUSE_ITEMS
    full = pd.DataFrame({"clip_id": [f"c{i}" for i in range(40)], "local": 0.5, "muse_baseline": 0.4,
                         **{c: 1.0 for c in number2_score.MUSE_ITEMS}})
    tr, te = full.assign(clip_id="t" + full.clip_id), full

    def final(train, test, its=items):
        return number2_score.complete_items(train, test, its, True)

    one, three = te.copy(), te.copy()
    one.loc[5, "object_size"] = np.nan
    three.loc[[1, 2], "muse_baseline"] = np.nan
    three.loc[3, "free_space"] = np.nan
    out = final(tr, one)
    res += [expect("final: complete Muse answers keep every clip", len(final(tr, te)[1]) == 40),
            expect("final: 1 of 40 clips without a Muse grade is dropped from the rows",
                   len(out[1]) == 39 and out[3] == {"train": [], "test": ["c5"]} and len(out[0]) == 40),
            expect("final: 3 of 40 clips (7.5%) without a Muse answer stops", stops(lambda: final(tr, three))),
            expect("final: a missing local grade stops", stops(lambda: final(tr, te.assign(local=[np.nan] + [0.5] * 39)))),
            expect("final: no Muse alone column stops",
                   stops(lambda: final(tr.drop(columns="muse_baseline"), te.drop(columns="muse_baseline")))),
            expect("final: a Muse checklist item absent from the items stops",
                   stops(lambda: final(tr, te, items[:-1])))]
    return res


def check_never_parsed_fallbacks():
    """Plan "Replies that never parse": arm A frames without an answer are not touching, arm C keeps arm B."""
    masks = muse_contact.a_masks(1, 60, {"0": {"left": True, "right": False}})  # frames 6 to 59 never answered
    segs = {"1": {"n30": 300, "left": [[50, 100]], "right": []}}
    return [expect("arm A fallback: frames without an answer are not touching",
                   masks["left"][:6].all() and not masks["left"][6:].any() and not masks["right"].any()),
            expect("arm C fallback: a boundary without a pick keeps arm B and is not a moved boundary",
                   muse_contact.c_segments(segs, {})[(1, "left")] == [(50, 100)]
                   and muse_contact.c_segments(segs, {}, moved_only=True)[(1, "left")] == {"start": [], "end": []})]


def check_object_reference():
    d = pd.DataFrame({"role": ["item_train"] * 2 + ["score_train"] * 2 + ["held_out"] * 2,
                      "object": ["mug", "can", "mug", "can", "mug", "bowl"],
                      "copyable": [1, 0, 0, 0, 1, 1]})
    test = d[d.role == "held_out"]
    final = number2_score.object_reference(d, test, "held_out").tolist()
    check = number2_score.object_reference(d, d[d.role == "score_train"], "score_train").tolist()
    return [expect("final: object share from item_train and score_train, unseen object gets their base rate",
                   final == [0.5, 0.25]),
            expect("check mode: object share from item_train only", check == [1.0, 0.0])]


def check_comparison6_and_extra():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 300), rng.integers(0, 5, 300).astype(float)  # many ties
    g = rng.choice(["a", "b", "c"], 300)
    pooled, table = number2_score.pair_auc(y, s, g)
    want = sum(roc_auc_score(y[g == k], s[g == k]) * table.loc[k, "pairs"] for k in "abc") / table.pairs.sum()
    res = [expect("pair AUC equals the pair-weighted sklearn AUC per group, with ties", abs(pooled - want) < 1e-12),
           expect("pair AUC with one group equals sklearn AUC",
                  abs(number2_score.pair_auc(y, s, np.zeros(300))[0] - roc_auc_score(y, s)) < 1e-12)]
    pool = pd.DataFrame({"object": ["mug"] * 6, "person": ["P1", "P1", "P2", "P2", "P2", "P2"],
                         "copyable": [1, 1, 0, 0, 0, 0]})
    clips = pd.DataFrame({"object": ["mug", "mug", "bowl"], "person": ["P1", "P3", "P1"]})
    r = np.random.default_rng(0)
    res.append(expect("rates from all clips of other persons: (0+1)/(4+2), (2+1)/(6+2), unseen (0+1)/(0+2)",
                      np.allclose(number2_score.draw_rates(clips, pool, None, r, True), [1 / 6, 3 / 8, 0.5])))
    res.append(expect("rates from k = 2 clips stay in [1/4, 3/4]",
                      all(np.isin(number2_score.draw_rates(clips, pool, 2, r, False)[:2], [0.25, 0.5, 0.75]))))
    return res


def check_soft(tmp):
    muse_soft.OUT = tmp / "soft"
    res = [expect(f"soft parse {t!r} -> {w}", (muse_soft.parse(t) or {}).get("softness") == w)
           for t, w in (('{"softness": " Soft "}', "soft"), ('{"softness": "hard"}', None), ("no json", None),
                        ('{"object": "can", "softness": "squeezable", "reason": "x"}', "squeezable"))]
    items = []
    for k in range(3):
        r = SimpleNamespace(item=f"og_000001_left_{k:05d}", dataset="opengraph", hand="left")
        d = muse_soft.OUT / "frames" / r.item
        d.mkdir(parents=True)
        for j in range(3):
            (d / f"f{j}.jpg").write_bytes(b"jpg")
        items.append(r)
    unparsed = muse_soft.OUT / "muse" / "unparsed"
    rec = muse_soft.call(FakeMuse(["sorry", "still no"]), items[0])
    res.append(expect("soft: two unparsed replies -> cached as unsure, both kept in unparsed/",
                      rec["parsed"] == {"softness": "unsure", "object": None, "reason": None, "forced_unsure": True}
                      and muse_soft.answer_path(items[0].item).exists()
                      and len(list(unparsed.glob(f"{items[0].item}_*.json"))) == 2))
    rec = muse_soft.call(FakeMuse(["sorry", '{"softness": "rigid"}']), items[1])
    res.append(expect("soft: second reply parses -> rigid, one kept in unparsed/, both calls paid",
                      rec["parsed"]["softness"] == "rigid" and len(list(unparsed.glob(f"{items[1].item}_*.json"))) == 1
                      and abs(rec["cost_usd"] - 2 * (1000 * muse_soft.PRICE_IN + 100 * muse_soft.PRICE_OUT)) < 1e-9))
    try:
        muse_soft.call(FakeMuse([]), items[2])
    except IndexError:
        pass
    res.append(expect("soft: a call that raises is not cached", not muse_soft.answer_path(items[2].item).exists()))
    a = muse_soft.answers([r.item for r in items])
    res.append(expect("soft: answers read back, the failed item missing",
                      a.softness.to_dict() == {items[0].item: "unsure", items[1].item: "rigid"}))

    lab = pd.DataFrame({"dataset": ["hot3d", "hot3d"] + ["opengraph"] * 5,
                        "object": ["can", "bottle", "sponge", "wrap", "glove", "plant", "cup"],
                        "labeler_1": [""] * 7, "labeler_2": [""] * 7,
                        "agreed": ["rigid", "squeezable", "soft", "soft", "soft", "soft", "rigid"]})
    hot = pd.DataFrame({"object": ["can"] * 10 + ["bottle"] * 10, "split": "held_out",
                        "softness": ["rigid"] * 9 + ["soft"] + ["squeezable"] * 10, "forced_unsure": False})
    og = pd.DataFrame({"object": ["sponge", "wrap", "glove", "plant"] * 5 + ["cup"] * 4,
                       "split": ["held_out", "tune"] * 12,
                       "softness": ["soft"] * 12 + ["unsure"] * 3 + ["rigid"] * 5 + ["rigid"] * 4,
                       "forced_unsure": False})
    m = muse_soft.soft_metrics(hot, og, lab)
    res.append(expect("soft metrics: caught 15/20, HOT3D held back 1/20, squeezable kept -> pass",
                      m["caught"]["all"] == 0.75 and m["hot3d_held_back"]["all"] == 0.05
                      and m["hot3d_held_back"]["by_label"] == {"rigid": 0.1, "squeezable": 0.0} and m["verdict"] == "pass"))
    m = muse_soft.soft_metrics(hot, og[og.object != "plant"], lab)
    res.append(expect("soft metrics: 15 segments of 3 soft objects -> too few", m["verdict"].startswith("too few")))
    m = muse_soft.soft_metrics(hot.assign(softness="unsure"), og, lab)
    res.append(expect("soft metrics: every HOT3D clip held back -> fail", m["verdict"] == "fail"))
    res.append(expect("soft labels: an empty agreed cell blocks sending",
                      muse_soft.labels_complete(lab) and not muse_soft.labels_complete(lab.assign(agreed=["rigid", ""] * 3 + [""]))))
    return res


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        res = check_sam(tmp) + check_muse_hot3d(tmp) + check_arm_a(tmp) + check_arm_c(tmp) + check_complete_items() \
            + check_never_parsed_fallbacks() + check_object_reference() + check_comparison6_and_extra() \
            + check_soft(tmp)
    print(f"{sum(res)} of {len(res)} passed")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
