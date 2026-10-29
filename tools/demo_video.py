"""Render the 3 minute submission demo video from the final results.

Follows the scene plan in docs/DEMO_VIDEO.md. Everything on screen comes from
files that already exist: the viewer data and media (viewer/data, viewer/media),
the demo figures and cards (out/demo, made by scripts/demo_figures.py). No API
call. The video has no sound. The narration is burned in as captions and also
written as an .srt file and a plain text script, so a voice can be recorded later.

Run in WSL from ~/graspgrade:

    .venv/bin/python tools/demo_video.py            # all scenes and the final video
    .venv/bin/python tools/demo_video.py --only 3a  # one scene, for checking

Output: out/demo/video/ (scene files, captions) and out/demo/graspgrade_demo.mp4.

With a voice: write the text with --voice-text, synthesize one WAV per shot with
tools/demo_voice.ps1 (Windows built-in voice), then run this again. When
out/demo/voice/<shot>.wav exists, each shot is as long as its voice plus a short
pause, the captions follow the voice, and the voice is mixed under the video.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIEW = ROOT / "viewer"
FIG = ROOT / "out" / "demo"
WORK = FIG / "video"
VOICE = FIG / "voice"
LEAD, TAIL = 0.4, 0.5  # silence before and after the voice in each shot
W, H, FPS = 1920, 1080, 30
CAP_H = 130  # bottom band kept free for captions

INK, INK2, MUTED, RULE, TINT = "#1d1d1f", "#3b4250", "#6a7382", "#dde1e7", "#f3f5f8"
BLUE, ORANGE, GREEN, RED, GRAY = "#2a78d6", "#eb6834", "#2b8a3e", "#c0392b", "#9aa2ae"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def canvas():
    return Image.new("RGB", (W, H), "white")


def header(d, left, right=None):
    d.text((80, 44), left, font=font(34, True), fill=INK)
    if right:
        f = font(26)
        d.text((W - 80 - d.textlength(right, font=f), 52), right, font=f, fill=MUTED)
    d.line([(80, 100), (W - 80, 100)], fill=RULE, width=2)


# ---------------------------------------------------------------- narration
# The plan's narration, split by shot so each caption sits on the shot it talks
# about. Scene 5 states the soft object result as the scope of the score, the
# way the report does.
NARRATION = {
    "1": "Robot hands can learn to grasp by watching people. But much hand video cannot teach a robot hand. "
         "The fingers are hidden, or the grasp is one a robot hand cannot copy.",
    "2": "GraspGrade takes first-person hand video, cuts it into clips at the moment of touch, and gives each clip "
         "a score from 0 to 100 for how likely a robot hand can copy it. SAM 3.1 tracks the hands and objects. "
         "Muse Spark answers a short checklist about each clip. We check both steps against real sensor data.",
    "3a": "First, when does a touch start and end? We check this against a touch-sensing glove. The top track is "
          "the glove. Below it are three methods: Muse Spark alone, our SAM rule, and the SAM rule checked by Muse "
          "Spark. Video shows a touch before the glove feels pressure, so we learn one timing shift, on tuning "
          "episodes only.",
    "3b": "On 22 held-out episodes, Muse Spark with the shift is best. Its boundary F1 within five frames is 0.195, "
          "against 0.090 for the SAM rule. All methods are low at five frames, because the glove measures pressure "
          "and video cannot see pressure.",
    "3c": "When touches are many and short, like in this episode, the SAM rule checked by Muse Spark does better.",
    "4a": "Second, can a robot hand copy this grasp? The ground truth comes from motion capture in the HOT3D "
          "dataset from Meta. We place an Allegro robot hand on each human grasp and check if it reaches the same "
          "contact points without passing through the object. This mug clip scores 78, and the robot hand can copy "
          "it. The score is a logistic regression on twelve grades, so every weight is visible.",
    "4b": "This keyboard clip scores 10. The robot hand would pass through the keyboard. Asked alone, Muse Spark "
          "said yes, at 85 percent.",
    "4c": "This glasses clip scores 78 and is copyable. Muse Spark alone said 5 percent.",
    "4d": "On 1,426 held-out clips from three people, our score reaches an AUC of 0.624. Muse Spark alone, asked "
          "the same question, is at 0.501, which is chance. The gain comes from asking a checklist instead of one "
          "question.",
    "5a": "We also show where the score stops working. When an object was seen in training, its name alone does "
          "better, at 0.771. On new objects our score keeps 0.571.",
    "5b": "Inside one object it is at chance. These two mugs get almost the same score, but only one grasp is "
          "copyable.",
    "5c": "The score is for rigid objects. Every rule was written before we scored the held-out data, and we "
          "scored it once.",
    "6": "The code, the viewer, and both ground truths are open, as a benchmark with its own scorers. "
         "GraspGrade finds the hand videos a robot can learn from.",
}


def caption_chunks(text, limit=88):
    """Split narration into caption lines of at most about `limit` characters."""
    out = []
    for sent in re.split(r"(?<=[.?!])\s+", text.strip()):
        while len(sent) > limit:
            cut = sent.rfind(", ", 0, limit)
            if cut < 30:
                cut = sent.rfind(" ", 0, limit)
                part, sent = sent[:cut], sent[cut + 1:]
            else:
                part, sent = sent[:cut + 1], sent[cut + 2:]
            out.append(part)
        out.append(sent)
    return out


def ts(t, sep):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", sep) if sep == "," else f"{int(h)}:{int(m):02d}:{s:05.2f}"


def write_captions(scenes):
    """Spread each shot's narration over its voice (or the shot), by length of text."""
    events, t0 = [], 0.0
    for key, dur, voice in scenes:
        chunks = caption_chunks(NARRATION[key])
        total = sum(len(c) + 12 for c in chunks)
        t = t0 + (LEAD if voice else 0.3)
        span = voice if voice else dur - 0.6
        for c in chunks:
            d = span * (len(c) + 12) / total
            events.append((t, t + d - 0.05, c))
            t += d
        t0 += dur
    srt = "\n".join(f"{i}\n{ts(a, ',')} --> {ts(b, ',')}\n{c}\n" for i, (a, b, c) in enumerate(events, 1))
    (WORK / "captions.srt").write_text(srt, encoding="utf-8")
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding",
           "Style: Cap,DejaVu Sans,42,&H00FFFFFF,&H00FFFFFF,&H40000000,&H40000000,0,0,0,0,100,100,0,0,3,14,0,2,120,120,40,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ass += [f"Dialogue: 0,{ts(a, '.')},{ts(b, '.')},Cap,,0,0,0,,{c}" for a, b, c in events]
    (WORK / "captions.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")
    script = "\n\n".join(f"Shot {k} ({d:.0f} s)\n{NARRATION[k]}" for k, d, _ in scenes)
    (WORK / "narration.txt").write_text(script + "\n", encoding="utf-8")


# ---------------------------------------------------------------- scene helpers
def encode(args, out, dur):
    """Run one ffmpeg graph to a scene file with a short fade from and to white."""
    vf_tail = f"fade=t=in:st=0:d=0.3:color=white,fade=t=out:st={dur - 0.3:.2f}:d=0.3:color=white,format=yuv420p"
    run(["ffmpeg", "-y", "-loglevel", "error"] + args[0] + ["-filter_complex", args[1] + f",{vf_tail}[v]",
         "-map", "[v]", "-t", f"{dur}", "-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         str(out)])


def still(png, out, dur):
    encode(([ "-loop", "1", "-framerate", str(FPS), "-i", str(png)], "[0:v]scale=1920:1080,setsar=1"), out, dur)


def figure_frame(fig):
    """A result figure, scaled into the area above the caption band."""
    im = canvas()
    f = Image.open(FIG / fig).convert("RGB")
    box_w, box_h = W - 160, H - 40 - CAP_H
    s = min(box_w / f.width, box_h / f.height)
    f = f.resize((int(f.width * s), int(f.height * s)), Image.LANCZOS)
    im.paste(f, ((W - f.width) // 2, 30))
    return im


def scope_card():
    """The scope of the score, stated the way the report states it (EXP-42)."""
    im = canvas()
    d = ImageDraw.Draw(im)
    x = 240
    d.text((x, 250), "Scope: rigid objects", font=font(64, True), fill=INK)
    lines = [("The score is for rigid objects, which is what HOT3D records.", INK2),
             ("Filtering out soft objects with Muse Spark is not reliable yet:", INK2),
             ("it caught 51.7% of soft segments, below the 70% rule set in advance.", MUTED)]
    for i, (t, c) in enumerate(lines):
        d.text((x, 400 + i * 64), t, font=font(38), fill=c)
    return im


def save(im, name):
    p = WORK / name
    im.save(p)
    return p


# ---------------------------------------------------------------- scenes
def scene1(out, dur, n2):
    im = canvas()
    d = ImageDraw.Draw(im)
    t = "Not every hand video can teach a robot hand."
    f = font(52, True)
    d.text(((W - d.textlength(t, font=f)) / 2, 56), t, font=f, fill=INK)
    bg = save(im, "s1_bg.png")
    a, b = VIEW / "media" / "P0010_99385ad8_left_00073.mp4", VIEW / "media" / "P0012_d6272ce1_left_00290.mp4"
    size, y = 780, 150
    graph = (f"[1:v]scale={size}:{size}[a];[2:v]scale={size}:{size}[b];"
             f"[0:v][a]overlay={W // 2 - size - 30}:{y}[x];[x][b]overlay={W // 2 + 30}:{y}")
    encode((["-loop", "1", "-framerate", str(FPS), "-i", str(bg),
             "-stream_loop", "-1", "-i", str(a), "-stream_loop", "-1", "-i", str(b)], graph), out, dur)


def episode_scene(out, dur, n1, ep, hand, note):
    e = [x for x in n1["episodes"] if x["episode"] == ep][0]
    tracks = e["hands"][hand]
    # the window of this length with the most glove boundaries
    starts = [s for seg in tracks["glove"] for s in seg]
    best, w0 = -1, 0.0
    for c in range(0, int((e["duration"] - dur) * 10) + 1):
        a = c / 10
        n = sum(a <= s <= a + dur for s in starts)
        if n > best:
            best, w0 = n, a
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Test 1  ·  When does a touch start and end?", "Held-out: 22 episodes, 574 glove boundaries")
    vx, vy, vw, vh = (W - 1024) // 2, 118, 1024, 576
    d.text((vx, vy + vh + 6), f"OpenGraph episode {ep}, {hand} hand, {w0:.0f} s to {w0 + dur:.0f} s",
           font=font(22), fill=MUTED)
    rows = [("glove", "Tactile glove (ground truth)", GREEN),
            ("arm_A_shift", "Muse Spark alone + shift", BLUE),
            ("arm_B", "SAM rule", ORANGE),
            ("arm_C_shift", "SAM rule + Muse Spark check", GRAY)]
    lx, tx, tw, ty, rh = 150, 620, 1080, vy + vh + 46, 44
    for i, (k, lab, col) in enumerate(rows):
        y = ty + i * rh
        d.text((lx, y + 6), lab, font=font(24, k == "glove"), fill=INK if k == "glove" else INK2)
        d.rectangle([tx, y + 8, tx + tw, y + rh - 8], fill=TINT)
        for s, t in tracks[k]:
            a, b = max(s, w0), min(t, w0 + dur)
            if b > a:
                d.rectangle([tx + (a - w0) / dur * tw, y + 8, tx + (b - w0) / dur * tw, y + rh - 8], fill=col)
        if k != "glove":
            sc = e["scores"][k]["boundary_f1_5"]
            d.text((tx + tw + 14, y + 8), f"F1 ±5  {sc:.2f}", font=font(22), fill=MUTED)
    if note:
        d.text((lx, ty + 4 * rh + 4), note, font=font(22), fill=MUTED)
    bg = save(im, f"s3_ep{ep}_bg.png")
    head = Image.new("RGBA", (4, 4 * rh + 6), INK)
    hp = save(head, "playhead.png")
    graph = (f"[1:v]scale={vw}:{vh}[v1];[0:v][v1]overlay={vx}:{vy}[x];"
             f"[x][2:v]overlay=x='{tx}+t/{dur}*{tw}':y={ty - 3}:eval=frame")
    encode((["-loop", "1", "-framerate", str(FPS), "-i", str(bg),
             "-ss", f"{w0}", "-t", f"{dur}", "-i", str(VIEW / e["video"]),
             "-loop", "1", "-framerate", str(FPS), "-i", str(hp)], graph), out, dur)


GRADE = {1.0: ("High", BLUE, "white"), 0.5: ("Mid", "#cfd6df", INK), 0.0: ("Low", "#eef0f3", MUTED)}


def clip_scene(out, dur, n2, cid, line):
    c = [x for x in n2["clips"] if x["id"] == cid][0]
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Test 2  ·  Can a robot hand copy this grasp?", "Held-out: 1,426 clips from 3 people")
    size, vx, vy = 800, 90, 130
    px = vx + size + 70
    obj = c["object"].replace("_", " ")
    d.text((px, 130), f"{obj}, {c['hand']} hand, {c['person']}", font=font(26), fill=MUTED)
    d.text((px, 168), f"{c['score']:.0f}", font=font(110, True), fill=INK)
    sw = d.textlength(f"{c['score']:.0f}", font=font(110, True))
    d.text((px + sw + 14, 228), "/ 100  GraspGrade score", font=font(28), fill=MUTED)
    ok = c["copyable_truth"]
    truth = "Robot hand can copy it" if ok else f"Robot hand cannot copy it ({c['failed_condition']})"
    d.text((px, 312), "Ground truth: " + truth, font=font(28, True), fill=GREEN if ok else RED)
    d.text((px, 356), f"Muse Spark asked directly: {c['muse_alone']['probability'] * 100:.0f}% copyable",
           font=font(28), fill=INK2)
    d.text((px, 420), "12 grades  ·  weight", font=font(22, True), fill=MUTED)
    gy = 456
    for g in sorted(c["grades"], key=lambda g: -abs(g["weight"])):
        lab, bgc, fg = GRADE[g["value"]]
        d.text((px, gy), g["name"], font=font(24), fill=INK2)
        cx = px + 470
        d.rounded_rectangle([cx, gy - 2, cx + 86, gy + 30], radius=14, fill=bgc)
        d.text((cx + 43 - d.textlength(lab, font=font(20, True)) / 2, gy + 2), lab, font=font(20, True), fill=fg)
        d.text((cx + 110, gy), f"{g['weight']:+.2f}", font=font(24), fill=MUTED)
        gy += 36
    if line:
        d.text((vx, vy + size + 16), line, font=font(24), fill=MUTED)
    bg = save(im, f"s4_{cid}.png")
    graph = f"[1:v]scale={size}:{size}[a];[0:v][a]overlay={vx}:{vy}"
    encode((["-loop", "1", "-framerate", str(FPS), "-i", str(bg),
             "-stream_loop", "-1", "-i", str(VIEW / c["video"])], graph), out, dur)


def compare_scene(out, dur, n2):
    ids = ["P0010_1573a837_right_02542", "P0001_b2bcbe28_left_01861"]
    cs = [[x for x in n2["clips"] if x["id"] == i][0] for i in ids]
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Where the score stops working", "Same object, almost the same score")
    size, y = 760, 130
    xs = [W // 2 - size - 30, W // 2 + 30]
    for x, c in zip(xs, cs):
        ok = c["copyable_truth"]
        t = f"Score {c['score']:.0f}  ·  " + ("robot hand can copy" if ok else f"cannot copy ({c['failed_condition']})")
        d.text((x, y + size + 14), t, font=font(30, True), fill=GREEN if ok else RED)
    bg = save(im, "s5_compare.png")
    graph = (f"[1:v]scale={size}:{size},tpad=stop_mode=clone:stop_duration={dur}[a];"
             f"[2:v]scale={size}:{size},tpad=stop_mode=clone:stop_duration={dur}[b];"
             f"[0:v][a]overlay={xs[0]}:{y}[x];[x][b]overlay={xs[1]}:{y}")
    encode((["-loop", "1", "-framerate", str(FPS), "-i", str(bg),
             "-i", str(VIEW / cs[0]["video"]), "-i", str(VIEW / cs[1]["video"])], graph), out, dur)


# ---------------------------------------------------------------- plan
def build(only=None):
    WORK.mkdir(parents=True, exist_ok=True)
    n1 = json.loads((VIEW / "data" / "number1.json").read_text())
    n2 = json.loads((VIEW / "data" / "number2.json").read_text())
    parts = [
        ("1", "1", 15, lambda o, d: scene1(o, d, n2)),
        ("2", "2", 20, lambda o, d: still(FIG / "demo_card_scene2.png", o, d)),
        ("3a", "3", 22, lambda o, d: episode_scene(o, d, n1, 36, "right", None)),
        ("3b", "3", 15, lambda o, d: still(save(figure_frame("demo_number1.png"), "s3_fig.png"), o, d)),
        ("3c", "3", 13, lambda o, d: episode_scene(o, d, n1, 7, "right",
                                                   "Many short touches: here the SAM rule checked by Muse Spark is closer.")),
        ("4a", "4", 22, lambda o, d: clip_scene(o, d, n2, "P0010_99385ad8_left_00073", None)),
        ("4b", "4", 13, lambda o, d: clip_scene(o, d, n2, "P0012_d6272ce1_left_00290", None)),
        ("4c", "4", 10, lambda o, d: clip_scene(o, d, n2, "P0001_b2bcbe28_left_03432", None)),
        ("4d", "4", 15, lambda o, d: still(save(figure_frame("demo_number2_ours.png"), "s4_fig.png"), o, d)),
        ("5a", "5", 8, lambda o, d: still(save(figure_frame("demo_number2_limits.png"), "s5_fig.png"), o, d)),
        ("5b", "5", 9, lambda o, d: compare_scene(o, d, n2)),
        ("5c", "5", 6, lambda o, d: still(save(scope_card(), "s5_scope.png"), o, d)),
        ("6", "6", 12, lambda o, d: still(FIG / "demo_card_scene6.png", o, d)),
    ]
    voice = {k: wav_len(VOICE / f"{k}.wav") for k, *_ in parts if (VOICE / f"{k}.wav").exists()}
    if voice:
        # each shot as long as its voice, but never shorter than the plan
        parts = [(k, g, round(max(d * 0.6, LEAD + voice[k] + TAIL), 2), fn) for k, g, d, fn in parts]
    for key, _, dur, fn in parts:
        if only and key != only:
            continue
        fn(WORK / f"scene_{key}.mp4", dur)
        print("scene", key, dur, "s")
    if only:
        return
    write_captions([(k, dur, voice.get(k)) for k, _, dur, _ in parts])
    (WORK / "list.txt").write_text("".join(f"file 'scene_{k}.mp4'\n" for k, *_ in parts))
    joined = WORK / "joined.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"),
         "-c", "copy", str(joined)])
    audio = []
    if voice:
        # one voice track: each shot's WAV after a short lead, padded to the shot length
        ins, chains = [], []
        for i, (k, _, dur, _) in enumerate(parts):
            ins += ["-i", str(VOICE / f"{k}.wav")]
            chains.append(f"[{i}:a]aresample=48000,adelay={int(LEAD * 1000)},apad,atrim=0:{dur}[a{i}]")
        graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(parts))) + \
            f"concat=n={len(parts)}:v=0:a=1,loudnorm=I=-16:TP=-1.5[a]"
        track = WORK / "voice.m4a"
        run(["ffmpeg", "-y", "-loglevel", "error"] + ins + ["-filter_complex", graph, "-map", "[a]",
             "-c:a", "aac", "-b:a", "160k", str(track)])
        audio = ["-i", str(track), "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
    final = FIG / "graspgrade_demo.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)] + audio +
        ["-vf", f"subtitles={WORK / 'captions.ass'}", "-c:v", "libx264", "-crf", "19", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)] + audio +
        ["-c:v", "copy", "-movflags", "+faststart", str(FIG / "graspgrade_demo_no_captions.mp4")])
    print("wrote", final, sum(p[2] for p in parts), "s")


def wav_len(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--voice-text", action="store_true", help="write out/demo/voice/narration.json and stop")
    a = ap.parse_args()
    if a.voice_text:
        VOICE.mkdir(parents=True, exist_ok=True)
        (VOICE / "narration.json").write_text(json.dumps(NARRATION, indent=1), encoding="utf-8")
    else:
        build(a.only)
