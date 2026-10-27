"""Compare SAM 3.1 object prompts on the smoke-test clip against the hand result.

    python scripts/sam_prompt_compare.py "small object" "round container"
    python scripts/sam_prompt_compare.py "small object" bottle \
        --dir out/meta_smoke/ep53 --contact-frame 50

Reads out/meta_smoke/sam_<prompt>_events.jsonl written by meta_smoke.py.
For each prompt it reports how many frames have an object, how stable the
object ids are, how often an object box sits on top of a hand box (the glove
taken for an object), and whether object and right-hand boxes touch before and
after the tactile contact at clip frame 51 (source frame 321). It also writes
a 2x2 overlay tile per prompt to out/meta_smoke/.
"""

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster, parse_responses_stream, video_segmentation_format

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "meta_smoke"
CONTACT_FRAME = 51  # first frame of tactile contact, clip index
TILE_FRAMES = [0, 50, 100, 140]
FRAME_AREA = 1280 * 720


def load(prompt):
    events = [json.loads(line) for line in (OUT / f"sam_{prompt.replace(' ', '_')}_events.jsonl").open()]

    async def replay():
        for e in events:
            yield e

    async def go():
        s = parse_responses_stream(replay(), video_segmentation_format())
        async for _ in s:
            pass
        return await s.final_result()

    res = asyncio.run(go())
    boxes, masks = {}, {}
    for r in res.records:
        if r.frame is None:
            continue
        f = r.frame.frame_index
        if r.kind == "box":
            boxes.setdefault(f, {})[r.object_id] = (r.left, r.top, r.right, r.bottom)
        elif r.kind == "mask":
            masks.setdefault(f, {})[r.object_id] = r
    return boxes, masks


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def touches(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def right_hand(hand_boxes):
    # the hand whose box centre is furthest right is the right hand in this clip
    if not hand_boxes:
        return None
    return max(hand_boxes.values(), key=lambda b: (b[0] + b[2]) / 2)


def full_mask(rec):
    m = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
    out = np.zeros((720, 1280), np.uint8)
    x0, y0 = int(rec.bounds.left), int(rec.bounds.top)
    out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m[:720 - y0, :1280 - x0]
    return out


def mask_gaps(boxes, masks, hands, hand_masks, n):
    """Per frame, pixel gap between the right-hand mask and the nearest object mask."""
    gaps = []
    for f in range(n):
        rb = right_hand(hands.get(f, {}))
        objs = [full_mask(r) for r in masks.get(f, {}).values()]
        objs = [o for o in objs if o.any()]
        if rb is None or not objs:
            gaps.append(np.nan)
            continue
        rid = next(k for k, v in hands[f].items() if v == rb)
        dist = cv2.distanceTransform((1 - full_mask(hand_masks[f][rid])).astype(np.uint8), cv2.DIST_L2, 3)
        gaps.append(min(float(dist[o > 0].min()) for o in objs))
    return np.array(gaps)


def summarize(prompt, hands, hand_masks):
    boxes, masks = load(prompt)
    n = 150
    per_frame = [len(boxes.get(f, {})) for f in range(n)]
    all_boxes = [b for f in boxes for b in boxes[f].values()]
    on_hand = sum(1 for f in boxes for b in boxes[f].values()
                  if any(iou(b, h) > 0.5 for h in hands.get(f, {}).values()))
    pre = [f for f in range(CONTACT_FRAME)]
    post = [f for f in range(CONTACT_FRAME, n)]

    def touch_rate(frames):
        hits = [any(touches(b, right_hand(hands.get(f, {}))) for b in boxes.get(f, {}).values())
                for f in frames if right_hand(hands.get(f, {}))]
        return round(100 * sum(hits) / len(hits), 1) if hits else None

    ids = sorted({i for f in boxes for i in boxes[f]})
    gaps = mask_gaps(boxes, masks, hands, hand_masks, n)

    def near_rate(frames):
        g = gaps[frames]
        g = g[~np.isnan(g)]
        return round(100 * float(np.mean(g <= 3)), 1) if len(g) else None

    return {
        "prompt": prompt,
        "frames_with_object_pct": round(100 * sum(c > 0 for c in per_frame) / n, 1),
        "mean_objects_per_frame": round(sum(per_frame) / n, 2),
        "object_ids": ids,
        "frames_per_id": {i: sum(i in boxes.get(f, {}) for f in range(n)) for i in ids},
        "median_box_area_pct": round(100 * float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in all_boxes])) / FRAME_AREA, 2) if all_boxes else None,
        "boxes_on_a_hand_pct": round(100 * on_hand / len(all_boxes), 1) if all_boxes else None,
        "object_touches_right_hand_before_contact_pct": touch_rate(pre),
        "object_touches_right_hand_after_contact_pct": touch_rate(post),
        "right_hand_mask_within_3px_before_contact_pct": near_rate(pre),
        "right_hand_mask_within_3px_after_contact_pct": near_rate(post),
    }


def tile(prompt, hand_boxes):
    boxes, masks = load(prompt)
    cap = cv2.VideoCapture(str(OUT / "clip.mp4"))
    frames, i = {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in TILE_FRAMES:
            frames[i] = img
        i += 1
    cap.release()
    colours = [(0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
    cells = []
    for f in TILE_FRAMES:
        img = frames[f].copy()
        for h in hand_boxes.get(f, {}).values():
            cv2.rectangle(img, (int(h[0]), int(h[1])), (int(h[2]), int(h[3])), (255, 0, 0), 2)
        for k, (oid, rec) in enumerate(sorted(masks.get(f, {}).items())):
            c = colours[k % len(colours)]
            m = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
            x0, y0 = int(rec.bounds.left), int(rec.bounds.top)
            roi = img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
            sel = m[:roi.shape[0], :roi.shape[1]] > 0
            roi[sel] = (0.4 * roi[sel] + 0.6 * np.array(c)).astype(np.uint8)
            b = boxes[f][oid]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
            cv2.putText(img, f"id{oid}", (int(b[0]), int(b[1]) - 6), 0, 1.0, c, 2)
        cv2.putText(img, f"{prompt}  f{f}", (20, 50), 0, 1.4, (0, 0, 0), 3)
        cells.append(cv2.resize(img, (640, 360)))
    grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    path = OUT / f"compare_{prompt.replace(' ', '_')}.jpg"
    cv2.imwrite(str(path), grid)
    return path


def main():
    global OUT, CONTACT_FRAME
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--dir", default=str(OUT), help="folder written by meta_smoke.py")
    ap.add_argument("--contact-frame", type=int, default=CONTACT_FRAME)
    args = ap.parse_args()
    OUT, CONTACT_FRAME = Path(args.dir).resolve(), args.contact_frame

    hands, hand_masks = load("hand")
    report = {}
    for p in args.prompts:
        boxes, _ = load(p)
        if not boxes:
            report[p] = {"prompt": p, "frames_with_object_pct": 0.0}
            print(json.dumps(report[p]))
            continue
        report[p] = summarize(p, hands, hand_masks)
        report[p]["tile"] = str(tile(p, hands).relative_to(ROOT))
        print(json.dumps(report[p], indent=1))
    (OUT / "prompt_compare.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
