"""Render the 3 minute submission demo video from the final results, in English or Korean.

Starts from the scene plan in docs/DEMO_VIDEO.md, opens on the question of why robots
lack data and which problem GraspGrade takes, and closes on what comes next.
Everything on screen comes from files that already exist: the viewer data and media (viewer/data, viewer/media),
the demo figures and cards (out/demo, made by scripts/demo_figures.py). No API
call. The narration is burned in as captions and also written as an .srt file
and a plain text script.

Run in WSL from ~/graspgrade:

    .venv/bin/python web/tools/demo_video.py               # English, out/demo/graspgrade_demo.mp4
    .venv/bin/python web/tools/demo_video.py --lang ko     # Korean, out/demo/graspgrade_demo_ko.mp4
    .venv/bin/python web/tools/demo_video.py --only 3a     # one scene, for checking

With a voice: write the text with --voice-text, synthesize one WAV per shot with
web/tools/demo_voice.ps1 (Windows built-in voice, -Voice "Microsoft Heami Desktop"
and -Dir ...\\voice_ko for Korean), then run this again. When the voice folder has
<shot>.wav, each shot is as long as its voice plus a short pause, the captions
follow the voice, and the voice is mixed under the video.

The Korean video draws its own charts and title cards with the same numbers, so
that no English text is left on screen. Korean text uses Malgun Gothic from the
Windows fonts folder.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]  # the repository, web/tools/ is two levels down
VIEW = ROOT / "viewer"
FIG = ROOT / "out" / "demo"
LEAD, TAIL = 0.4, 0.5  # silence before and after the voice in each shot
W, H, FPS = 1920, 1080, 30
CAP_H = 130  # bottom band kept free for captions

INK, INK2, MUTED, RULE, TINT = "#1d1d1f", "#3b4250", "#6a7382", "#dde1e7", "#f3f5f8"
BLUE, ORANGE, GREEN, RED, GRAY = "#2a78d6", "#eb6834", "#2b8a3e", "#c0392b", "#9aa2ae"

WINFONTS = Path("/mnt/c/Windows/Fonts")
FONTS = {
    "en": ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "DejaVu Sans"),
    "ko": (str(WINFONTS / "malgun.ttf"), str(WINFONTS / "malgunbd.ttf"), "Malgun Gothic"),
}
LANG = "en"
WORK = FIG / "video"
VOICE = FIG / "voice"


def setup(lang, voice="sapi"):
    """voice: sapi for the Windows built-in voice, el for ElevenLabs (English only, web/tools/demo_voice_el.py)."""
    global LANG, WORK, VOICE
    LANG = lang
    tag = "" if lang == "en" else f"_{lang}"
    if voice == "el":
        tag += "_el"
    WORK = FIG / f"video{tag}"
    VOICE = FIG / f"voice{tag}"


def font(size, bold=False):
    return ImageFont.truetype(FONTS[LANG][1 if bold else 0], size)


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


# ---------------------------------------------------------------- words on screen
TEXT = {
    "en": {
        "news": ["Meta Max, a humanoid of its own (reported)",
                 "Stand-in footage. No public images of Max yet.",
                 [("Height", "about 5 ft"), ("Weight", "about 100 lb"), ("Job", "help with chores"),
                  ("Built", "in 3 months, with Fourier"), ("Next", "version 2 climbs stairs")]],
        "news_marks": ["five feet", "100 pounds", "chores", "three months", "second version"],
        "intro_marks": ["Language models", "Robot data", "A human hand", "And most hand video", "This is where"],
        "intro": ["Why is physical AI still far from its GPT moment?",
                  "Language models learned from the whole internet. Robots have no data at that scale.",
                  [("Robot data is slow and costly", "Recorded by remote control, one robot at a time"),
                   ("A human hand is not a robot hand", "Each grasp must be moved onto the robot (retargeting)"),
                   ("Most hand video is not usable", "83 of 2,000 internet clips passed a quality check (Do as I Do)")]],
        "s1_title": "First-person hand video: easy to collect, hard to choose",
        "future": ["Done in this hackathon, and next", "This hackathon", "Next",
                   [("Rigid objects (HOT3D)", "Touch check for soft objects (now 51.7%, rule 70%)"),
                    ("One robot hand, geometric check", "Physics-based retargeting and more robot hands"),
                    ("Score at the object level", "Rank grasps of the same object (now AUC 0.506)"),
                    ("Lab video, 3 held-out people", "Video from homes and factories")]],
        "t1": "Test 1  ·  When does a touch start and end?", "t1_right": "Held-out: 22 episodes, 574 glove boundaries",
        "ep_line": "OpenGraph episode {ep}, {hand} hand, {a:.0f} s to {b:.0f} s",
        "rows": ["Tactile glove (measured)", "Muse Spark alone + shift", "SAM rule", "SAM rule + Muse Spark check"],
        "t2": "Test 2  ·  Can a robot hand copy this grasp?", "t2_right": "Held-out: 1,426 clips from 3 people",
        "clip_meta": "{obj}, {hand} hand, {person}", "score_suffix": "/ 100  GraspGrade score",
        "truth_ok": "Robot hand can copy it",
        "truth_no": "Robot hand cannot copy it ({why})", "muse_direct": "Muse Spark asked directly: {p:.0f}% copyable",
        "grades_head": "12 grades  ·  weight", "grade": {1.0: "High", 0.5: "Mid", 0.0: "Low"},
        "cmp_head": "Where the score stops working", "cmp_right": "Same object, almost the same score",
        "cmp_ok": "Score {s:.0f}  ·  robot hand can copy", "cmp_no": "Score {s:.0f}  ·  cannot copy ({why})",
        "hand": {"left": "left", "right": "right"}, "why": {}, "obj": {},
    },
    "ko": {
        "news": ["Meta의 휴머노이드 맥스 (보도)",
                 "대체 영상. 맥스의 공개 사진은 아직 없음.",
                 [("키", "약 152 cm"), ("무게", "약 45 kg"), ("용도", "집안일 돕기"),
                  ("제작", "3개월, Fourier와 함께"), ("다음", "계단 오르는 2번째 버전")]],
        "news_marks": ["152", "45", "집안일", "석 달", "두 번째"],
        "intro_marks": ["언어 모델은", "원격 조종", "사람 손은", "그리고 대부분의", "바로 여기서"],
        "intro": ["피지컬 AI는 왜 아직 GPT 모먼트까지 갈 길이 멀까?",
                  "언어 모델은 인터넷 전체로 배웠지만, 로봇에게는 그만한 데이터가 없음",
                  [("로봇 데이터는 느리고 비쌈", "원격 조종으로 로봇 한 대씩 기록"),
                   ("사람 손은 로봇 손과 다름", "잡기마다 로봇 손으로 옮겨야 함 (리타게팅)"),
                   ("대부분의 손 영상은 쓸 수 없음", "인터넷 클립 2,000개 중 83개만 품질 검사 통과 (Do as I Do)")]],
        "s1_title": "1인칭 손 영상: 모으기는 쉽고, 고르기는 어렵다",
        "future": ["해커톤에서 한 것과 다음 단계", "이번 해커톤", "다음 단계",
                   [("단단한 물체 (HOT3D)", "부드러운 물체 촉각 확인 (현재 51.7%, 기준 70%)"),
                    ("로봇 손 하나, 기하 검사", "물리 기반 리타게팅, 다른 로봇 손"),
                    ("물체 수준의 점수", "같은 물체 안 잡기 순위 (현재 AUC 0.506)"),
                    ("실험실 영상, 검증 3명", "가정과 공장 영상")]],
        "t1": "평가 1  ·  접촉은 언제 시작하고 끝나는가?", "t1_right": "검증: 에피소드 22개, 장갑 경계 574개",
        "ep_line": "OpenGraph 에피소드 {ep}, {hand}, {a:.0f}초 ~ {b:.0f}초",
        "rows": ["촉각 장갑 (실제 접촉)", "Muse Spark 단독 + 시점 보정", "SAM 규칙", "SAM 규칙 + Muse Spark 확인"],
        "t2": "평가 2  ·  로봇 손이 이 잡기를 따라 할 수 있는가?", "t2_right": "검증: 3명의 클립 1,426개",
        "clip_meta": "{obj}, {hand}, {person}", "score_suffix": "/ 100  GraspGrade 점수",
        "truth_ok": "로봇 손이 따라 할 수 있음",
        "truth_no": "로봇 손이 따라 할 수 없음 ({why})", "muse_direct": "Muse Spark에게 바로 물음: {p:.0f}% 가능",
        "grades_head": "등급 12개  ·  가중치", "grade": {1.0: "상", 0.5: "중", 0.0: "하"},
        "cmp_head": "점수가 통하지 않는 곳", "cmp_right": "같은 물체, 거의 같은 점수",
        "cmp_ok": "{s:.0f}점  ·  로봇 손이 따라 할 수 있음", "cmp_no": "{s:.0f}점  ·  따라 할 수 없음 ({why})",
        "hand": {"left": "왼손", "right": "오른손"},
        "why": {"collision": "충돌", "grip": "쥐기", "reach": "도달"},
        "obj": {"mug white": "흰 머그컵", "keyboard": "키보드", "aria small": "안경 (Aria)",
                "mug patterned": "무늬 머그컵"},
        "grade_names": {
            "Camera stability": "카메라 안정성", "Action completeness": "동작 완결성", "Hand shape fit": "손 모양 적합도",
            "Visibility": "가시성", "Contact clarity": "접촉 명확성", "Finger visibility": "손가락 가림",
            "Object size next to the hand": "손 대비 크기", "Object shape": "모양", "Object thickness": "두께",
            "Width to wrap": "감쌀 폭", "Handle": "손잡이", "Free space around the object": "빈 공간"},
    },
}


def T(key):
    return TEXT[LANG][key]


# ---------------------------------------------------------------- narration
# The narration, split by shot so each caption sits on the shot it talks about.
# It opens with why robots lack data and which problem GraspGrade takes, shows
# the two tests in short, and ends with what the hackathon covered and what
# comes next. The soft object result is part of the next steps (EXP-42).
NARRATION = {
    "en": {
        "n0": "Good morning, robotics enthusiasts. Meta has reportedly built a humanoid of its own. Max stands five "
              "feet tall, weighs about 100 pounds, and is designed to help with chores. Meta built it in three months "
              "with Chinese robotics firm Fourier, and a second version that can climb stairs is already in "
              "development.",
        "0a": "So why is physical AI still so far from its GPT moment? Language models learned from the whole "
              "internet. Robots have no data at that scale. Robot data from remote control is slow and costly. A "
              "human hand is not a robot hand. And most hand video is not usable. This is where GraspGrade comes in.",
        "0b": "First-person hand video is the easiest data to collect and grow. It has two problems: moving each "
              "grasp onto a robot hand, and choosing which videos are worth using. We solve the second problem.",
        "2": "GraspGrade cuts first-person hand video into clips at each touch, and scores each clip from 0 to 100 "
             "for how likely a robot hand can copy it. SAM 3.1 tracks the hands and objects. Muse Spark answers a "
             "short checklist. We check both steps against real sensor data.",
        "3a": "First, when does a touch start and end? The top track is a touch-sensing glove. Below it are Muse Spark "
              "alone, our SAM rule, and the SAM rule checked by Muse Spark.",
        "3b": "On 22 held-out episodes, Muse Spark with a timing shift learned on tuning data is best, with a boundary "
              "F1 of 0.195 against 0.090 for the SAM rule. All methods are low, because video cannot see pressure.",
        "4a": "Second, can a robot hand copy this grasp? We find the real answer with motion capture from Meta's "
              "HOT3D dataset, by placing an Allegro robot hand on each human grasp. This mug clip scores 78, and the robot "
              "hand can copy it. The score weighs twelve grades, and every weight is visible.",
        "4b": "This keyboard clip scores 10. The robot hand would pass through the keyboard. Muse Spark alone said "
              "yes, at 85 percent.",
        "4d": "On 1,426 held-out clips from three people, our score reaches an AUC of 0.624. Muse Spark alone, asked "
              "the same question, is at 0.501, which is chance. The gain comes from the checklist.",
        "5a": "On new objects our score keeps 0.571. When an object was seen before, its name alone does better, "
              "at 0.771.",
        "5b": "Inside one object it is at chance. These two mugs score almost the same, but only one grasp is "
              "copyable.",
        "7": "In this hackathon we scored rigid objects with one robot hand, on lab video. Next, we would add a touch "
             "check for soft objects, test the top clips with physics-based retargeting and more robot hands, rank "
             "grasps of the same object, and test on video from homes and factories.",
        "6": "Every rule was fixed before we scored the held-out data. The code, the viewer, and both labeled datasets "
             "are open as a benchmark. GraspGrade finds the hand videos a robot can learn from.",
    },
    "ko": {
        "n0": "로봇 팬 여러분, 좋은 아침입니다. 메타가 자체 휴머노이드를 만들었다는 보도가 나왔습니다. 이름은 맥스로, "
              "키는 약 152센티미터, 무게는 약 45킬로그램이고, 집안일을 돕도록 설계됐습니다. 메타는 중국 로봇 회사 "
              "Fourier와 함께 석 달 만에 만들었고, 계단을 오르는 두 번째 버전도 이미 개발 중입니다.",
        "0a": "그렇다면 피지컬 AI는 왜 아직 GPT 모먼트까지 갈 길이 멀까요? 언어 모델은 인터넷 전체로 배웠지만, 로봇에게는 "
              "그만한 데이터가 없습니다. 원격 조종으로 모으는 로봇 데이터는 느리고 비쌉니다. 사람 손은 로봇 손과 "
              "다릅니다. 그리고 대부분의 손 영상은 쓸 수 없습니다. 바로 여기서 GraspGrade가 시작됩니다.",
        "0b": "1인칭 손 영상은 가장 쉽게 모으고 가장 크게 키울 수 있는 데이터입니다. 하지만 두 가지 문제가 있습니다. "
              "사람의 잡기를 로봇 손으로 옮기는 리타게팅, 그리고 어떤 영상이 쓸 만한지 판단하는 일입니다. 우리는 이 "
              "두 번째 문제를 풉니다.",
        "2": "GraspGrade는 1인칭 손 영상을 손이 닿는 순간마다 클립으로 자르고, 로봇 손이 따라 할 수 있는 정도를 "
             "0점에서 100점으로 매깁니다. SAM 3.1이 손과 물체를 추적하고, Muse Spark가 짧은 체크리스트에 답합니다. "
             "두 단계 모두 실제 센서 데이터로 검증했습니다.",
        "3a": "첫째, 접촉은 언제 시작하고 언제 끝날까요? 맨 위 줄은 촉각 장갑입니다. 그 아래는 Muse Spark 단독, SAM "
              "규칙, 그리고 Muse Spark가 확인한 SAM 규칙입니다.",
        "3b": "검증 에피소드 22개에서는 튜닝 데이터로 시점을 보정한 Muse Spark가 가장 좋았습니다. 경계 F1이 0.195로, SAM 규칙의 "
              "0.090보다 높습니다. 영상은 압력을 볼 수 없어서 모든 방법의 점수가 낮습니다.",
        "4a": "둘째, 로봇 손이 이 잡기를 따라 할 수 있을까요? 실제 답은 Meta의 HOT3D 모션캡처에서 사람의 잡기마다 "
              "Allegro 로봇 손을 올려 확인했습니다. 이 머그컵 클립은 78점이고, 로봇 손이 따라 할 수 있습니다. 점수는 "
              "등급 12개에 가중치를 매겨 계산하고, 모든 가중치가 보입니다.",
        "4b": "이 키보드 클립은 10점입니다. 로봇 손이 키보드를 뚫고 들어갑니다. Muse Spark 단독은 85퍼센트로 된다고 "
              "답했습니다.",
        "4d": "세 사람의 검증 클립 1,426개에서 우리 점수의 AUC는 0.624입니다. 같은 질문을 받은 Muse Spark 단독은 "
              "0.501로 우연 수준입니다. 이득은 체크리스트에서 나옵니다.",
        "5a": "처음 보는 물체에서도 우리 점수는 0.571을 유지합니다. 본 적 있는 물체라면 물체 이름만으로 0.771이 더 "
              "높습니다.",
        "5b": "같은 물체 안에서는 우연 수준입니다. 이 두 머그컵은 점수가 거의 같지만, 한 잡기만 따라 할 수 있습니다.",
        "7": "이번 해커톤에서는 실험실 영상의 단단한 물체를 로봇 손 하나로 채점했습니다. 다음으로는 부드러운 물체를 "
             "위한 촉각 확인, 물리 기반 리타게팅과 다른 로봇 손으로 상위 클립 확인, 같은 물체 안의 잡기 순위, 그리고 "
             "가정과 공장 영상에서의 검증을 고려하고 있습니다.",
        "6": "모든 규칙은 검증 데이터를 채점하기 전에 정했습니다. 코드, 뷰어, 두 정답 데이터는 벤치마크로 공개합니다. "
             "GraspGrade는 로봇이 배울 수 있는 손 영상을 찾아 줍니다.",
    },
}

# how the Korean voice should say names and decimals; the captions keep the written form
SAY_KO = [("GPT", "지피티"), ("피지컬 AI", "피지컬 에이아이"), ("GraspGrade", "그래스프 그레이드"), ("Muse Spark", "뮤즈 스파크"), ("Fourier", "푸리에"), ("SAM", "샘"), ("HOT3D", "핫 쓰리디"),
          ("Allegro", "알레그로"), ("AUC", "에이유씨"), ("F1", "에프원"), ("Meta", "메타"), ("1인칭", "일인칭")]
DIGITS_KO = "영일이삼사오육칠팔구"


DIGITS_EN = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def voice_text(text):
    if LANG != "ko":
        # said the way people say it: 0.771 is "point seven seven one", 0.090 is "point oh nine"
        return re.sub(r"\b0\.(\d+)", lambda m: "point " + " ".join(DIGITS_EN[int(c)] for c in m.group(1).rstrip("0")),
                      text)
    for a, b in SAY_KO:
        text = text.replace(a, b)
    return re.sub(r"(\d+)\.(\d+)", lambda m: m.group(1).replace("0", "영") + " 점 " +
                  " ".join(DIGITS_KO[int(c)] for c in m.group(2).rstrip("0")), text)


def caption_chunks(text):
    """Split narration into caption lines short enough for one line on screen."""
    limit = 88 if LANG == "en" else 40
    out = []
    for sent in re.split(r"(?<=[.?!])\s+", text.strip()):
        if LANG == "ko":
            # equal parts cut at the space nearest each mark, so no line is a lone word
            n = -(-len(sent) // limit)
            for _ in range(n - 1):
                aim = len(sent) // n
                inside = {m.start() + j for m in re.finditer(r"Muse Spark|SAM 3\.1", sent) for j in range(len(m[0]))}
                cut = min((i for i, ch in enumerate(sent) if ch == " " and i not in inside), key=lambda i: abs(i - aim))
                out.append(sent[:cut])
                sent, n = sent[cut + 1:], n - 1
            out.append(sent)
            continue
        while len(sent) > limit:
            cut = sent.rfind(", ", 0, limit)
            if cut < limit // 3:
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
        chunks = caption_chunks(NARRATION[LANG][key])
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
           f"Style: Cap,{FONTS[LANG][2]},42,&H00FFFFFF,&H00FFFFFF,&H40000000,&H40000000,0,0,0,0,100,100,0,0,3,14,0,2,"
           "120,120,40,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ass += [f"Dialogue: 0,{ts(a, '.')},{ts(b, '.')},Cap,,0,0,0,,{c}" for a, b, c in events]
    (WORK / "captions.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")
    script = "\n\n".join(f"Shot {k} ({d:.0f} s)\n{NARRATION[LANG][k]}" for k, d, _ in scenes)
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


def intro_card(stage=5, with_title=True):
    """Why robots lack data, and the one problem GraspGrade takes (the last row, in blue).
    stage: how much is shown under the question, 0 the question alone, 1 with the line under it,
    2 to 4 with rows, 5 with the last row marked as the GraspGrade problem."""
    im = canvas()
    d = ImageDraw.Draw(im)
    title, sub, rows = T("intro")
    if with_title:
        d.text((140, 110), title, font=intro_title_font(), fill=INK)
    if stage >= 1:
        d.text((140, 215), sub, font=font(34), fill=MUTED)
    for i, (head, detail) in enumerate(rows[:max(stage - 1, 0)]):
        y = 340 + i * 170
        ours = i == len(rows) - 1 and stage >= 5
        if ours:
            d.rectangle([110, y - 22, W - 110, y + 128], fill="#eef3fd")
            f = font(40, True)
            d.text((W - 150 - d.textlength("GraspGrade", font=f), y + 34), "GraspGrade", font=f, fill=BLUE)
        d.text((140, y), str(i + 1), font=font(52, True), fill=BLUE if ours else GRAY)
        d.text((220, y + 2), head, font=font(44, True), fill=INK if ours else INK2)
        d.text((220, y + 66), detail, font=font(30), fill=MUTED)
    return im


# stand-in humanoid footage for the news shot: one 16 s take of a robot in a kitchen (start, length in seconds).
# The file is a downloaded YouTube video kept in web/vids, which is not in git.
NEWS_SRC = ROOT / "web" / "vids" / ("YTDown.com_YouTube_Media_eGSrUaca0BM_Which-Humanoid-Robots-Are-Actually-"
                                    "Working-in-2026-One-Is-Remote-Controlled_001_720p.mp4")
NEWS_TAKE = (512.0, 15.9)


def voice_times(key, marks, dur):
    """When the voice of a shot reaches each mark, from the mark's place in the text."""
    text = NARRATION[LANG][key]
    wav = VOICE / f"{key}.wav"
    span = wav_len(wav) if wav.exists() else dur - LEAD - TAIL
    return [LEAD + span * text.index(m) / len(text) for m in marks]


def news_card(stage):
    """The reported facts about Max, next to the stand-in footage. stage: how many facts are shown."""
    im = canvas()
    d = ImageDraw.Draw(im)
    title, note, rows = T("news")
    header(d, title)
    d.text((80, 150 + 648 + 14), note, font=font(24), fill=MUTED)
    d.text((1300, 160), "Max", font=font(64, True), fill=INK)
    for i, (lab, val) in enumerate(rows[:stage]):
        y = 280 + i * 104
        d.text((1300, y), lab, font=font(26), fill=MUTED)
        d.text((1300, y + 34), val, font=font(36, True), fill=BLUE if i == len(rows) - 1 else INK)
    return im


def news_scene(out, dur):
    times = voice_times("n0", T("news_marks"), dur)
    ins, graph = [], []
    for stage in range(len(times) + 1):
        ins += ["-loop", "1", "-framerate", str(FPS), "-i", str(save(news_card(stage), f"sn_news_{stage}.png"))]
    last = "0:v"
    for stage, t in enumerate(times, 1):
        graph.append(f"[{stage}:v]format=rgba,fade=t=in:st={t:.2f}:d=0.35:alpha=1[f{stage}];"
                     f"[{last}][f{stage}]overlay=0:0[o{stage}]")
        last = f"o{stage}"
    # the take, slowed down when the shot is longer than the take
    start, length = NEWS_TAKE
    ins += ["-ss", f"{start}", "-t", f"{min(length, dur)}", "-i", str(NEWS_SRC)]
    slow = max(dur / length, 1.0)
    graph.append(f"[{len(times) + 1}:v]setpts=(PTS-STARTPTS)*{slow:.3f},fps={FPS},scale=1152:648,setsar=1[foot]")
    graph.append(f"[{last}][foot]overlay=80:150:eof_action=repeat")
    encode((ins, ";".join(graph)), out, dur)


def intro_title_font():
    size = 60
    while ImageDraw.Draw(canvas()).textlength(T("intro")[0], font=font(size, True)) > W - 280:
        size -= 2
    return font(size, True)


def intro_scene(out, dur):
    """The intro card, built up line by line as the voice reaches each line. The question starts
    a little above the middle, like a title, and moves up to its place as the first line appears."""
    times = voice_times("0a", T("intro_marks"), dur)
    ins, graph, last = [], [], "0:v"
    for stage in range(len(times) + 1):
        im = intro_card(stage, with_title=False)
        ins += ["-loop", "1", "-framerate", str(FPS), "-i", str(save(im, f"s0_intro_{stage}.png"))]
    for stage, t in enumerate(times, 1):
        graph.append(f"[{stage}:v]format=rgba,fade=t=in:st={t:.2f}:d=0.35:alpha=1[f{stage}];"
                     f"[{last}][f{stage}]overlay=0:0[o{stage}]")
        last = f"o{stage}"
    # the question on its own layer, eased from the middle to the top left in 0.7 s
    f = intro_title_font()
    q = Image.new("RGBA", (int(ImageDraw.Draw(canvas()).textlength(T("intro")[0], font=f)) + 4, 120), (0, 0, 0, 0))
    ImageDraw.Draw(q).text((0, 0), T("intro")[0], font=f, fill=INK)
    ins += ["-loop", "1", "-framerate", str(FPS), "-i", str(save(q, "s0_question.png"))]
    t0, move = times[0] - 0.5, 0.7
    p = f"clip((t-{t0:.2f})/{move},0,1)"
    ease = f"({p}*{p}*(3-2*{p}))"
    xc, yc = (W - q.width) // 2, 400
    graph.append(f"[{last}][{len(times) + 1}:v]overlay=x='{xc}+(140-{xc})*{ease}':y='{yc}+(110-{yc})*{ease}'"
                 ":eval=frame")
    encode((ins, ";".join(graph)), out, dur)


def future_card():
    """What the hackathon covered, next to the step that would widen each part."""
    im = canvas()
    d = ImageDraw.Draw(im)
    title, left, right, rows = T("future")
    d.text((140, 90), title, font=font(56, True), fill=INK)
    lx, rx = 140, 880
    d.text((lx, 240), left, font=font(30, True), fill=MUTED)
    d.text((rx, 240), right, font=font(30, True), fill=BLUE)
    for i, (done, nxt) in enumerate(rows):
        y = 320 + i * 140
        d.line([(lx, y - 22), (W - 100, y - 22)], fill=RULE, width=2)
        d.text((lx, y + 10), done, font=font(32), fill=INK2)
        d.line([(rx - 120, y + 32), (rx - 50, y + 32)], fill=BLUE, width=6)
        d.polygon([(rx - 30, y + 32), (rx - 55, y + 16), (rx - 55, y + 48)], fill=BLUE)
        d.text((rx, y + 10), nxt, font=font(32, True), fill=INK)
    return im


# ---------------------------------------------------------------- Korean charts and cards (same numbers as the figures)
N1_ROWS = [("Muse Spark 단독 (A)", 0.074, 0.821, GRAY), ("A + 시점 보정", 0.195, 0.850, BLUE),
           ("SAM 규칙 (B)", 0.090, 0.667, ORANGE), ("SAM 규칙 + Muse Spark (C)", 0.112, 0.679, GRAY),
           ("C + 시점 보정", 0.137, 0.690, GRAY), ("항상 접촉", 0.042, 0.600, "#c9ced6")]
N2_ROWS = [("우리 점수", 0.624, 0.589, 0.661), ("Muse Spark 단독, 같은 질문", 0.501, 0.469, 0.529),
           ("물체 이름, 본 적 있는 물체", 0.771, 0.744, 0.792), ("우리 점수, 처음 보는 물체", 0.571, 0.534, 0.609)]


def chart_n1_ko():
    im = canvas()
    d = ImageDraw.Draw(im)
    d.text((140, 70), "접촉은 언제 시작하고 끝나는가?  검증 에피소드 22개, 장갑 경계 574개", font=font(38, True), fill=INK)
    panels = [("경계 F1 (5프레임 이내)", 1, 0.30, 820), ("접촉 구간 겹침 (IoU)", 2, 1.0, 1420)]
    lab_x, y0, rh, bw = 140, 230, 92, 480
    for (title, col, top, px) in panels:
        d.text((px - 60, 160), title, font=font(30, True), fill=INK2)
    for i, row in enumerate(N1_ROWS):
        y = y0 + i * rh
        d.text((lab_x, y + 8), row[0], font=font(30, row[3] == BLUE), fill=INK)
        for (title, col, top, px) in panels:
            v = row[col]
            x0 = px - 60
            d.rectangle([x0, y, x0 + bw, y + 46], fill=TINT)
            d.rectangle([x0, y, x0 + bw * v / top, y + 46], fill=row[3])
            d.text((x0 + bw * v / top + 12, y + 6), f"{v:.3f}", font=font(30, row[3] == BLUE), fill=INK)
    d.text((lab_x, y0 + 6 * rh + 20), "정답: 촉각 장갑. 시점 보정은 튜닝 에피소드에서만 배움.", font=font(26), fill=MUTED)
    return im


def chart_n2_ko(highlight):
    """AUC bars with 95% intervals. highlight: {row index: colour}, the rest gray."""
    im = canvas()
    d = ImageDraw.Draw(im)
    d.text((140, 70), "로봇 손이 따라 할 수 있는 잡기인가?  검증 클립 1,426개, 3명", font=font(38, True), fill=INK)
    x0, x1, lo, hi = 760, 1760, 0.3, 0.9
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    y0, rh = 230, 150
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        d.line([(X(t), y0 - 20), (X(t), y0 + 4 * rh - 40)], fill=RULE if t != 0.5 else "#8a919c", width=2)
        d.text((X(t) - 22, y0 + 4 * rh - 30), f"{t:.1f}", font=font(26), fill=MUTED)
    d.text((X(0.5) - 30, y0 - 60), "우연", font=font(26), fill=MUTED)
    for i, (lab, v, a, b) in enumerate(N2_ROWS):
        y = y0 + i * rh
        col = highlight.get(i, "#c4c9d0")
        d.text((140, y + 18), lab, font=font(32, i in highlight), fill=INK)
        d.rectangle([x0, y + 10, X(v), y + 76], fill=col)
        d.line([(X(a), y + 43), (X(b), y + 43)], fill=INK, width=3)
        d.text((max(X(v), X(b)) + 16, y + 22), f"{v:.3f}", font=font(32, i in highlight), fill=INK)
    d.text((140, y0 + 4 * rh + 30), "AUC: 로봇 손 정답 라벨과 비교 (가로선은 시퀀스 단위 95% 구간)", font=font(26), fill=MUTED)
    return im


def card2_ko():
    im = canvas()
    d = ImageDraw.Draw(im)
    d.text((115, 80), "GraspGrade", font=font(120, True), fill=INK)
    d.rectangle([115, 345, 640, 755], fill="#eef3fd")
    d.text((155, 380), "입력", font=font(32, True), fill=BLUE)
    d.text((155, 460), "1인칭 손 영상", font=font(58, True), fill=INK)
    d.text((155, 560), "길이 상관없음", font=font(36), fill=MUTED)
    d.line([(680, 545), (780, 545)], fill=BLUE, width=12)
    d.polygon([(800, 545), (765, 515), (765, 575)], fill=BLUE)
    d.text((840, 380), "출력", font=font(32, True), fill=BLUE)
    for i, t in enumerate(["닿는 순간마다 자른 클립", "0~100점: 로봇 손이 따라 할 수 있는가?", "한 줄 이유"]):
        y = 470 + i * 110
        d.text((840, y), str(i + 1), font=font(48, True), fill=BLUE)
        d.text((910, y + 4), t, font=font(44), fill=INK)
    d.text((115, 880), "SAM 3.1  ·  Muse Spark", font=font(44), fill=MUTED)
    return im


def card6_ko():
    im = canvas()
    d = ImageDraw.Draw(im)
    c = lambda t, f, y, fill: d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=fill)  # noqa: E731
    c("GraspGrade", font(130, True), 230, INK)
    c("로봇이 배울 수 있는 손 영상을 찾아 줍니다.", font(48), 420, MUTED)
    for i, (h, s) in enumerate([("코드", "단계마다 스크립트 하나"), ("뷰어", "검증 클립과 에피소드"),
                                ("벤치마크", "두 정답 데이터, 독립 채점 코드")]):
        x = W / 2 + (i - 1) * 520
        f1, f2 = font(54, True), font(34)
        d.text((x - d.textlength(h, font=f1) / 2, 590), h, font=f1, fill=BLUE)
        d.text((x - d.textlength(s, font=f2) / 2, 670), s, font=f2, fill=MUTED)
    c("데이터: OpenGraph Ego-Tactile (CC-BY-4.0)  ·  HOT3D (Meta)", font(30), 860, MUTED)
    return im


def save(im, name):
    p = WORK / name
    im.save(p)
    return p


# ---------------------------------------------------------------- scenes
def scene1(out, dur, n2):
    im = canvas()
    d = ImageDraw.Draw(im)
    t = T("s1_title")
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
    header(d, T("t1"), T("t1_right"))
    vx, vy, vw, vh = (W - 1024) // 2, 118, 1024, 576
    d.text((vx, vy + vh + 6), T("ep_line").format(ep=ep, hand=T("hand")[hand], a=w0, b=w0 + dur),
           font=font(22), fill=MUTED)
    keys = [("glove", GREEN), ("arm_A_shift", BLUE), ("arm_B", ORANGE), ("arm_C_shift", GRAY)]
    lx, tx, tw, ty, rh = 150, 620, 1080, vy + vh + 46, 44
    for i, ((k, col), lab) in enumerate(zip(keys, T("rows"))):
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


GRADE_COLORS = {1.0: (BLUE, "white"), 0.5: ("#cfd6df", INK), 0.0: ("#eef0f3", MUTED)}


def clip_scene(out, dur, n2, cid, line):
    c = [x for x in n2["clips"] if x["id"] == cid][0]
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, T("t2"), T("t2_right"))
    size, vx, vy = 800, 90, 130
    px = vx + size + 70
    obj = c["object"].replace("_", " ")
    obj = T("obj").get(obj, obj)
    d.text((px, 130), T("clip_meta").format(obj=obj, hand=T("hand")[c["hand"]], person=c["person"]),
           font=font(26), fill=MUTED)
    d.text((px, 168), f"{c['score']:.0f}", font=font(110, True), fill=INK)
    sw = d.textlength(f"{c['score']:.0f}", font=font(110, True))
    d.text((px + sw + 14, 228), T("score_suffix"), font=font(28), fill=MUTED)
    ok = c["copyable_truth"]
    why = T("why").get(c["failed_condition"], c["failed_condition"])
    truth = T("truth_ok") if ok else T("truth_no").format(why=why)
    d.text((px, 312), truth, font=font(28, True), fill=GREEN if ok else RED)
    d.text((px, 356), T("muse_direct").format(p=c["muse_alone"]["probability"] * 100), font=font(28), fill=INK2)
    d.text((px, 420), T("grades_head"), font=font(22, True), fill=MUTED)
    names = TEXT[LANG].get("grade_names", {})
    gy = 456
    for g in sorted(c["grades"], key=lambda g: -abs(g["weight"])):
        lab = T("grade")[g["value"]]
        bgc, fg = GRADE_COLORS[g["value"]]
        d.text((px, gy), names.get(g["name"], g["name"]), font=font(24), fill=INK2)
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
    header(d, T("cmp_head"), T("cmp_right"))
    size, y = 760, 130
    xs = [W // 2 - size - 30, W // 2 + 30]
    for x, c in zip(xs, cs):
        ok = c["copyable_truth"]
        why = T("why").get(c["failed_condition"], c["failed_condition"])
        t = T("cmp_ok").format(s=c["score"]) if ok else T("cmp_no").format(s=c["score"], why=why)
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
    ko = LANG == "ko"
    fig = lambda name, png, im_ko: (lambda o, d: still(save(im_ko() if ko else figure_frame(png), name), o, d))  # noqa
    card = lambda png, im_ko: (lambda o, d: still(save(im_ko(), png) if ko else FIG / png, o, d))  # noqa: E731
    parts = [
        ("n0", "0", 16, news_scene),
        ("0a", "0", 22, intro_scene),
        ("0b", "1", 16, lambda o, d: scene1(o, d, n2)),
        ("2", "2", 15, card("demo_card_scene2.png", card2_ko)),
        ("3a", "3", 16, lambda o, d: episode_scene(o, d, n1, 36, "right", None)),
        ("3b", "3", 12, fig("s3_fig.png", "demo_number1.png", chart_n1_ko)),
        ("4a", "4", 18, lambda o, d: clip_scene(o, d, n2, "P0010_99385ad8_left_00073", None)),
        ("4b", "4", 9, lambda o, d: clip_scene(o, d, n2, "P0012_d6272ce1_left_00290", None)),
        ("4d", "4", 13, fig("s4_fig.png", "demo_number2_ours.png", lambda: chart_n2_ko({0: BLUE, 1: ORANGE}))),
        ("5a", "5", 9, fig("s5_fig.png", "demo_number2_limits.png", lambda: chart_n2_ko({2: ORANGE, 3: BLUE}))),
        ("5b", "5", 8, lambda o, d: compare_scene(o, d, n2)),
        ("7", "7", 18, lambda o, d: still(save(future_card(), "s7_future.png"), o, d)),
        ("6", "6", 10, card("demo_card_scene6.png", card6_ko)),
    ]
    voice = {k: wav_len(VOICE / f"{k}.wav") for k, *_ in parts if (VOICE / f"{k}.wav").exists()}
    if voice:
        # each shot as long as its voice, but never shorter than the plan
        parts = [(k, g, round(max(d * 0.6, LEAD + voice[k] + TAIL), 2) if k in voice else d, fn)
                 for k, g, d, fn in parts]
    for key, _, dur, fn in parts:
        if only and key != only:
            continue
        fn(WORK / f"scene_{key}.mp4", dur)
        print("scene", key, dur, "s")
    if only:
        if only in voice:
            # one shot with its voice, for listening before the whole video is made
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(WORK / f"scene_{only}.mp4"),
                 "-i", str(VOICE / f"{only}.wav"), "-filter_complex",
                 f"[1:a]aresample=48000,adelay={int(LEAD * 1000)},apad[a]", "-map", "0:v", "-map", "[a]",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(WORK / f"preview_{only}.mp4")])
            print("wrote", WORK / f"preview_{only}.mp4")
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
    tag = WORK.name.removeprefix("video")
    final = FIG / f"graspgrade_demo{tag}.mp4"
    subs = f"subtitles={WORK / 'captions.ass'}" + (f":fontsdir={WINFONTS}" if LANG == "ko" else "")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)] + audio +
        ["-vf", subs, "-c:v", "libx264", "-crf", "19", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)] + audio +
        ["-c:v", "copy", "-movflags", "+faststart", str(FIG / f"graspgrade_demo{tag}_no_captions.mp4")])
    print("wrote", final, sum(p[2] for p in parts), "s")


def wav_len(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ko"], default="en")
    ap.add_argument("--voice", choices=["sapi", "el"], default="sapi")
    ap.add_argument("--only")
    ap.add_argument("--voice-text", action="store_true", help="write the voice folder's narration.json and stop")
    a = ap.parse_args()
    setup(a.lang, a.voice)
    if a.voice_text:
        VOICE.mkdir(parents=True, exist_ok=True)
        (VOICE / "narration.json").write_text(
            json.dumps({k: voice_text(v) for k, v in NARRATION[LANG].items()}, indent=1, ensure_ascii=False),
            encoding="utf-8")
    else:
        build(a.only)
