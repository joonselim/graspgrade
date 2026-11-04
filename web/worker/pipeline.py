"""Score one new human egocentric video with the GraspGrade pipeline.

Same steps and the same learned values as the final runs (EXP-37, EXP-40), with
the code of scripts/ imported read-only:

1. Contact timing, method A: frames at 5 fps to Muse Spark, 10 per call, asking
   per hand whether it touches an object (scripts/muse_contact.py). Gaps of 0.2 s
   are joined, pieces under 0.2 s dropped, then the timing shift learned on the
   glove (+7 frames at the start, -7 at the end, model.json).
2. Grasp clips: contacts joined across gaps of 0.5 s or less, kept if 0.3 s or
   longer, padded by 1 s each side (scripts/hot3d_clips.py).
3. SAM 3.1 on the whole video at 10 fps, prompts "hand" and "small object"
   (scripts/sam_episodes.py), then visibility and contact clarity per clip
   (scripts/contact_cut.py, scripts/number2_sam_checks.py).
4. The Muse Spark checklist, 6 frames per clip (scripts/muse_hot3d.py, prompt
   number2_checklist.txt, version 1).
5. Camera stability, motion completeness and hand shape from the video at
   10 fps: optical flow and MediaPipe Hands (scripts/hot3d_local_checks.py,
   scripts/hot3d_hand_shape.py).
6. The score: 100 * sigmoid(intercept + sum of weight * grade), model.json.

Frames sent to Muse Spark keep the video's aspect ratio, the HOT3D clips were
square. The prompt for step 1 drops the line about beige gloves (OpenGraph).
"""
import base64
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import muse_contact as mc  # noqa: E402
import muse_hot3d as mh  # noqa: E402
import sam_episodes as se  # noqa: E402
from contact_cut import frame_table  # noqa: E402
from number2_sam_checks import clip_values as sam_values  # noqa: E402
from hot3d_hand_shape import shape  # noqa: E402

MODEL = json.loads((Path(__file__).resolve().parent / "model.json").read_text())
SIDES = ["left", "right"]
MAX_CLIPS = 20
MERGE_GAP, MIN_LEN, PAD = 15, 9, 30          # scripts/hot3d_clips.py, 30 fps frames
SMALL = 352                                   # optical flow size, scripts/hot3d_local_checks.py
FLIP = {"Left": 1, "Right": 0}                # MediaPipe label -> slot, human egocentric view is not mirrored
NAMES = {
    "grade_camera_stability": "Camera stability", "grade_completeness": "Action completeness",
    "grade_hand_shape_fit": "Hand shape fit", "grade_visibility": "Visibility",
    "grade_contact_clarity": "Contact clarity", "finger_occlusion": "Finger visibility",
    "object_size": "Object size next to the hand", "object_shape": "Object shape",
    "object_thickness": "Object thickness", "object_width": "Width to wrap", "object_handle": "Handle",
    "free_space": "Free space around the object",
}


# ---------------------------------------------------------------- video

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    return float(d["format"]["duration"]), int(d["streams"][0]["width"]), int(d["streams"][0]["height"])


def normalize(src, dst):
    """30 fps, at most 720 px high, upright (ffmpeg applies phone rotation), no audio."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "fps=30,scale=-2:'min(720,ih)'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(dst)],
                   check=True)


def estimate(duration_s):
    """Dollars, from the per call prices in scripts/: method A, SAM on two prompts, one checklist per clip."""
    n30 = int(duration_s * 30)
    calls = math.ceil(math.ceil(n30 / mc.A_STEP) / mc.A_BATCH)
    a = calls * ((mc.A_BATCH * mc.TOKENS_PER_FRAME + 300) * mc.PRICE_IN + mc.TOKENS_OUT * mc.PRICE_OUT)
    sam = 2 * max(se.SAM_MIN_FRAMES, math.ceil(n30 / 3)) * se.SAM_PRICE_FRAME
    clips = min(MAX_CLIPS, max(1, math.ceil(duration_s / 4)))
    return round(a + sam + clips * 0.011, 2)


def scan(path, n_hint):
    """One pass: jpgs every 6th frame (method A), and at every 3rd frame camera motion and MediaPipe hands."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    jpgs, fr, cam, lm2d, lm3d = {}, [], [], [], []
    prev, i = None, 0
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % mc.A_STEP == 0:
                jpgs[i] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            if i % 3 == 0:
                grey = cv2.cvtColor(cv2.resize(img, (SMALL, SMALL)), cv2.COLOR_BGR2GRAY)
                motion = np.nan
                if prev is not None:
                    pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
                    if pts is not None and len(pts) >= 8:
                        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, grey, pts, None)
                        good = st.ravel() == 1
                        if good.sum() >= 8:
                            M, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                               ransacReprojThreshold=2.0)
                            if M is not None:
                                motion = float(np.hypot(M[0, 2], M[1, 2]))
                prev = grey
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                a2, a3, sc = np.full((2, 21, 2), np.nan), np.full((2, 21, 3), np.nan), np.zeros(2)
                if res.multi_handedness:
                    for h, l2, l3 in zip(res.multi_handedness, res.multi_hand_landmarks,
                                         res.multi_hand_world_landmarks):
                        c = h.classification[0]
                        slot = FLIP[c.label]
                        if c.score <= sc[slot]:
                            continue
                        sc[slot] = c.score
                        a2[slot] = [(p.x, p.y) for p in l2.landmark]
                        a3[slot] = [(p.x, p.y, p.z) for p in l3.landmark]
                fr.append(i)
                cam.append(motion)
                lm2d.append(a2)
                lm3d.append(a3)
            i += 1
    cap.release()
    return i, jpgs, {"frames": np.array(fr), "cam": np.array(cam), "lm2d": np.array(lm2d), "lm3d": np.array(lm3d)}


def frames_at(path, wanted, max_side=720):
    """jpg bytes of the wanted frames, longer side at most max_side, aspect kept."""
    cap = cv2.VideoCapture(str(path))
    out, i, last, wanted = {}, 0, max(wanted), set(wanted)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i in wanted:
            h, w = img.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            out[i] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        i += 1
    cap.release()
    return out


# ---------------------------------------------------------------- 1, 2: contact and clips

def contact_prompt():
    lines = (REPO / "scripts" / "prompts" / "number1_arm_a.txt").read_text().splitlines()
    return "\n".join(l for l in lines if "beige gloves" not in l).replace(
        "They are numbered 1 to {n}.", "They are numbered 1 to {n}.\nThe left hand is on the person's left side.")


def ask_contact(client, batch, jpgs, text):
    """One method A call, sent again once if the reply misses frames. Returns ({frame: {left, right}}, cost)."""
    cost, ans = 0.0, {}
    for _ in range(2):
        reply, _, c = mc.ask(client, text.replace("{n}", str(len(batch))), [jpgs[k] for k in batch])
        cost += c
        d = mc.first_json(reply)
        ans = {}
        if d and isinstance(d.get("frames"), list):
            for x in d["frames"]:
                try:
                    i = int(x["i"])
                    if 1 <= i <= len(batch):
                        ans[batch[i - 1]] = {s: bool(x.get(s)) for s in SIDES}
                except (KeyError, TypeError, ValueError):
                    pass
        if len(ans) == len(batch):
            break
    return ans, cost  # frames still without an answer count as not touching


def contacts(answers, n30):
    out = {}
    for s in SIDES:
        m = np.zeros(n30, bool)
        for k, v in answers.items():
            if v.get(s):
                m[k:k + mc.A_STEP] = True
        a, b = MODEL["contact_shift_frames"]
        out[s] = mc.shift_segments(mc.clean(m), n30, a, b)
    return out


def clips_from(segs, n30):
    rows = []
    for s in SIDES:
        merged = []
        for a, b in segs[s]:
            if merged and a - merged[-1][1] <= MERGE_GAP:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            if b - a < MIN_LEN:
                continue
            c0, c1 = max(0, a - PAD), min(n30, b + PAD)
            rows.append({"clip_id": f"{s}_{a:05d}", "hand": s, "clip_start": c0, "clip_frames": c1 - c0,
                         "contact_start": a, "contact_frames": b - a})
    rows.sort(key=lambda r: -r["contact_frames"])  # the longest grasps first when there are too many
    return pd.DataFrame(rows[:MAX_CLIPS]).sort_values("clip_start").reset_index(drop=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- 4, 5, 6: grades and score

def checklist(client, clip, path):
    cs, cf, c0, c1 = clip.contact_start, clip.contact_frames, clip.clip_start, clip.clip_start + clip.clip_frames
    idx = [c0, cs, cs + cf // 3, cs + 2 * cf // 3, cs + cf - 1, c1 - 1]
    jp = frames_at(path, idx)
    content = [{"type": "text", "text": mh.prompt("checklist", clip.hand)}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jp[k]).decode()}}
        for k in idx if k in jp]
    cost = 0.0
    for _ in range(2):
        r = client.chat.completions.create(model="muse-spark-1.3", reasoning_effort="low",
                                           messages=[{"role": "user", "content": content}])
        u = r.usage
        cost += (u.prompt_tokens * mh.PRICE_IN + u.completion_tokens * mh.PRICE_OUT) if u else 0.0
        parsed = mh.parse("checklist", r.choices[0].message.content)
        if parsed:
            return parsed, cost
    return None, cost


def tier(x, cuts, higher_is_better=True):
    if x is None or np.isnan(x):
        return 0.0
    if not higher_is_better:
        x, cuts = -x, [-cuts[1], -cuts[0]]
    return 1.0 if x >= cuts[1] else (0.5 if x >= cuts[0] else 0.0)


def video_grades(v, c):
    fr, slot = v["frames"], SIDES.index(c.hand)
    c1, e = c.clip_start + c.clip_frames, c.contact_start + c.contact_frames
    sel = (fr >= c.clip_start) & (fr < c1)
    con = (fr >= c.contact_start) & (fr < e)
    reach, rel = (fr >= c.clip_start) & (fr < c.contact_start), (fr >= e) & (fr < c1)
    seen = ~np.isnan(v["lm2d"][:, slot, 0, 0])
    t = v["cam"][sel]
    cam = float(np.nanpercentile(t, 90)) if np.isfinite(t).any() else np.nan

    def ok(m):
        w = v["lm2d"][m & seen, slot, 0]
        moved = float(np.linalg.norm(w.max(0) - w.min(0))) if len(w) >= 2 else 0.0
        return (float(seen[m].mean()) if m.any() else 0.0) >= 0.5 and moved >= 0.03

    hs = MODEL["hand_shape"]
    pick = con & seen
    f = np.nanmedian(shape(v["lm3d"][pick, slot].astype(float)), 0) if pick.any() else np.full(len(hs["fill"]), np.nan)
    f = np.where(np.isnan(f), hs["fill"], f)
    z = (f - np.array(hs["mean"])) / np.array(hs["scale"])
    p = 1 / (1 + math.exp(-(hs["intercept"] + float(np.dot(hs["coef"], z)))))
    return {"grade_camera_stability": tier(cam, MODEL["camera_cuts_px"], higher_is_better=False),
            "grade_completeness": (ok(reach) + ok(rel)) / 2,
            "grade_hand_shape_fit": tier(p, hs["cuts"])}


def run(src, work, progress, client):
    """Score the video at src. progress(text) reports steps. Returns (result, cost in dollars)."""
    work.mkdir(parents=True, exist_ok=True)
    cost = {"contact": 0.0, "sam": 0.0, "checklist": 0.0}
    norm = work / "norm.mp4"
    progress("영상을 30fps로 맞추는 중")
    normalize(src, norm)
    duration, W, H = probe(norm)

    progress("손 추적과 카메라 움직임 계산 중 (MediaPipe)")
    n30, jpgs, vid = scan(norm, int(duration * 30))

    progress("접촉 시점 판정 중 (Muse Spark)")
    samples = list(range(0, n30, mc.A_STEP))
    batches = [samples[i:i + mc.A_BATCH] for i in range(0, len(samples), mc.A_BATCH)]
    text, answers = contact_prompt(), {}
    with ThreadPoolExecutor(4) as pool:
        for ans, c in pool.map(lambda b: ask_contact(client, b, jpgs, text), batches):
            answers.update(ans)
            cost["contact"] += c
    segs = contacts(answers, n30)
    clips = clips_from(segs, n30)
    hands_s = {s: [[a / 30, b / 30] for a, b in segs[s]] for s in SIDES}
    base = {"duration_s": round(duration, 2), "width": W, "height": H, "hands": hands_s,
            "model": "EXP-37 contact timing, EXP-40 score", "clips": []}
    if clips.empty:
        return {**base, "cost": cost}, sum(cost.values())

    progress(f"잡기 {len(clips)}개, 손과 물체 마스크 계산 중 (SAM 3.1)")
    sam_dir = work / "sam"
    sam_dir.mkdir(exist_ok=True)
    clip10 = sam_dir / "clip10.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(norm), "-vf",
                    "select='not(mod(n,3))',setpts=N/(10*TB)", "-r", "10", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(clip10)], check=True)
    for prompt in se.PROMPTS:
        se.send(client, 0, clip10, prompt)
        info = json.loads((sam_dir / f"sam_{prompt.replace(' ', '_')}.json").read_text())
        if info["status"] != "completed":
            raise RuntimeError(f"SAM 3.1 did not complete ({prompt}: {info['status']})")
        cost["sam"] += max(se.SAM_MIN_FRAMES, info.get("frames_billed") or 0) * se.SAM_PRICE_FRAME
    n10 = math.ceil(n30 / 3)
    rec, _ = frame_table(sam_dir, "_events.jsonl.gz", n10, (W, H))
    sam = {r["clip_id"]: r for r in sam_values(rec, 0, clips)}

    progress(f"체크리스트 질문 중 (Muse Spark, 클립 {len(clips)}개)")
    with ThreadPoolExecutor(4) as pool:
        muse = list(pool.map(lambda c: checklist(client, c, norm), clips.itertuples()))

    out = []
    for c, (m, mcost) in zip(clips.itertuples(), muse):
        cost["checklist"] += mcost
        g = video_grades(vid, c)
        g["grade_visibility"] = tier(sam[c.clip_id]["visibility"], MODEL["visibility_cuts"])
        g["grade_contact_clarity"] = tier(sam[c.clip_id]["contact_clarity"], MODEL["contact_clarity_cuts"])
        for k in mh.CHECKLIST_KEYS:
            g[k] = m[k] if m else 0.0
        z = MODEL["intercept"] + sum(MODEL["weights"][k] * g[k] for k in MODEL["items"])
        out.append({"hand": c.hand, "start": round(c.clip_start / 30, 2),
                    "end": round((c.clip_start + c.clip_frames) / 30, 2),
                    "contact": [round(c.contact_start / 30, 2), round((c.contact_start + c.contact_frames) / 30, 2)],
                    "score": round(100 / (1 + math.exp(-z)), 1),
                    "object": (m or {}).get("object_name"), "reason": (m or {}).get("reason"),
                    "checklist_answered": m is not None,
                    "grades": [{"name": NAMES[k], "value": g[k], "weight": round(MODEL["weights"][k], 3)}
                               for k in MODEL["items"]]})
    return {**base, "clips": out, "cost": {k: round(v, 4) for k, v in cost.items()}}, sum(cost.values())
