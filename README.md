# GraspGrade

GraspGrade takes first-person videos of hands using objects, cuts them into
clips at the moment of contact, and gives each clip a score for how likely it
is that a robot hand can copy it.

Video is the only input. Two kinds of sensor data are used only to check the
work:

1. **When does a touch start and end?** Checked against a touch-sensing
   glove (OpenGraph Ego-Tactile).
2. **Can a robot hand copy this grasp?** Checked against motion capture of
   hands and objects (Meta HOT3D), mapped onto an Allegro robot hand.

Meta models: SAM 3.1 tracks hands and objects. Muse Spark answers the
checklist questions and is also the baseline we compare with.

Every comparison, threshold and pass rule was written in
[docs/MEASUREMENT_PLAN.md](docs/MEASUREMENT_PLAN.md) before the held-out
data was scored. Every experiment has a page with its setup, numbers, cost
and figures in [docs/experiments/](docs/experiments/README.md).

## Status

All runs are done. Meta API spend: $143.27 of the $150 credit.

| Part | State |
| --- | --- |
| Number 1: contact timing | final result on 22 held-out episodes (EXP-37) |
| Number 2: robot copyable score | final result on 1,426 held-out clips (EXP-40), also at two other depth limits (EXP-41) |
| Soft object check | done, fails its pass rule (EXP-42) |
| Viewer | static page with the final held-out results |
| Benchmark | both ground truths as label files with standalone scorers, in `benchmark/` (EXP-28) |

## Results

### Number 1: when does a touch start and end? (EXP-37)

22 held-out episodes, 574 glove boundaries. Ground truth: the touch glove,
cleaned of pressure flicker under 0.1 s. A boundary counts as found within
±5 frames (0.167 s) or ±15 frames (0.5 s).

| Row | F1 ±5 | F1 ±15 | Contact IoU |
| --- | --- | --- | --- |
| A. Muse Spark alone | 0.074 | 0.273 | 0.821 |
| **A + timing shift** | **0.195** | **0.376** | **0.850** |
| B. SAM rule | 0.090 | 0.214 | 0.667 |
| C. SAM rule + Muse Spark boundary check | 0.112 | 0.227 | 0.679 |
| C + timing shift | 0.137 | 0.266 | 0.690 |
| Always in contact | 0.042 | 0.042 | 0.600 |

- The timing shift is learned from the glove on tuning episodes only. Video
  shows a touch before the glove feels pressure, so Muse Spark answers early.
  The shift adds value to both Muse Spark arms (paired bootstrap over
  episodes, F1 at ±5): A +0.121 (0.062 to 0.172), C +0.025 (0.019 to 0.032).
- Muse Spark alone with the shift is the best row on all three metrics.
- The Muse Spark check improves the SAM rule (C over B, +0.022 at ±5, 0.004
  to 0.037, a comparison added after the run), but C stays below A + shift. The SAM rule makes about three times too many
  boundaries, and C can only move them.
- All rows are low at ±5 frames, because the glove measures pressure, which
  video cannot see.

### Number 2: can a robot hand copy this grasp? (EXP-40, EXP-41)

1,426 held-out clips from 3 persons, 46.4% copyable. The score is a logistic
regression on twelve grades (three local checks, two SAM checks, seven Muse
Spark checklist grades), learned on 400 tuning clips. AUC with a 95%
interval from a bootstrap over the 51 held-out sequences.

| Row | AUC | 95% interval |
| --- | --- | --- |
| **Our score** | **0.624** | 0.589 to 0.661 |
| Muse Spark alone, asked the same question | 0.501 | 0.469 to 0.529 |
| Muse Spark alone plus our checks | 0.624 | 0.590 to 0.661 |
| Object name reference (tuning share per object) | 0.771 | 0.744 to 0.792 |
| Our score, objects never seen in training | 0.571 | 0.534 to 0.609 |
| Object name reference, objects never seen | 0.500 | by construction |

- Muse Spark alone cannot tell copyable from non-copyable clips. Asking it
  the checklist instead lifts the AUC to about 0.62 (+0.117, 0.073 to 0.166).
  Learning the weights adds nothing over equal weights (+0.005), and the
  local and SAM checks add nothing on top (+0.001).
- The object name alone beats our score, because held-out persons grasp the
  same objects as the tuning persons. On objects never seen in training our
  score keeps 0.571, where the object name has nothing to say.
- Inside one object our score is at chance (0.506). It ranks objects, not
  grasps of the same object.
- Once 5 or more labeled clips of an object exist, the object rate alone is
  as good as our score combined with it.
- Depth limit (EXP-41): the copyable label allows a robot mesh point up to
  5 mm inside the object, fixed in advance. Judged against a 10 mm label the
  picture holds (our score 0.632, Muse Spark alone 0.541). Against a 2 mm
  label, where only 20% of clips are copyable, our score is at chance (0.521)
  and Muse Spark alone is below it (0.417).

### Soft object check (EXP-42)

The score is meant for rigid objects. One more Muse Spark question per clip
(rigid, squeezable, soft or unsure) should take soft objects out. Ground
truth: object labels by two team members, set before any call (agreement
81.7%, Cohen's kappa 0.613).

| Metric | Value | Pass rule | Met |
| --- | --- | --- | --- |
| Soft OpenGraph segments held back | 51.7% (31 of 60, 9 objects) | at least 70% | No |
| Rigid HOT3D clips held back | 2.1% (30 of 1,426) | at most 10% | Yes |

The check fails. Muse Spark mostly calls tape squeezable or rigid. Bubble
wrap, gloves and mats are mostly caught. It also holds back 18% of rigid
OpenGraph segments. Dropping the held-back HOT3D clips leaves the number 2
AUC unchanged (0.625).

## Viewer

`viewer/` is a static page with three tabs, built from the final held-out
results.

- **Clip scores** (number 2): the 15 highest and 15 lowest scored held-out
  clips. Each clip shows the video, the contact interval from motion capture,
  the score, the ground truth with the failing condition, the twelve grades
  with their learned weights, the Muse Spark checklist reason, and what Muse
  Spark alone answered. Presets filter by score (Strict 70+, Balanced 50+,
  Volume 30+).
- **Contact timing** (number 1): the 22 held-out episodes. For each hand, one
  track each for the glove, A + shift, B and C + shift, with F1 and IoU per
  episode. Click a track to jump in the video.
- **Touch check** (soft object check): the 30 HOT3D held-out clips the soft
  object check holds back, with the Muse Spark answer and reason and the
  object label. Marked as not validated, because the check failed its pass
  rule (EXP-42).

Direct links open one view: `#clip=ID`, `#episode=36&hand=right`,
`#compare=ID1,ID2`.

Build the data and the short videos (no API call), then serve the page:

```bash
.venv/bin/python scripts/viewer_data.py    # writes viewer/data/*.json and viewer/media/*.mp4
cd viewer && python3 -m http.server 8765    # then open http://localhost:8765
```

`viewer/media/` is not in git. It is cut from the datasets, so it needs the
data folders described below.

## Demo video

`web/tools/demo_video.py` renders the 3 minute demo video from the viewer data,
the viewer videos and the demo figures, following the scene plan in
`docs/DEMO_VIDEO.md`. No API call. The narration is a synthesized voice
(`web/tools/demo_voice.ps1`) with captions on screen. `--lang ko` makes the
Korean version with the Korean voice (Microsoft Heami, rate 3), its own Korean
charts and cards, and Malgun Gothic text.

```bash
.venv/bin/python scripts/demo_figures.py   # figures and title cards, out/demo/
.venv/bin/python web/tools/demo_video.py   # out/demo/graspgrade_demo.mp4
.venv/bin/python web/tools/demo_video.py --lang ko   # out/demo/graspgrade_demo_ko.mp4
```

## Layout

| Path | What |
| --- | --- |
| `scripts/` | all code, one script per step, each with a usage note at the top |
| `scripts/prompts/` | Muse Spark prompts |
| `docs/MEASUREMENT_PLAN.md` | what is compared, on which data, decided before the results |
| `docs/experiments/` | one page per experiment, with figures |
| `docs/DATA_NOTES.md`, `docs/ENVIRONMENT.md` | facts about the data, setup notes |
| `splits/` | fixed tuning and held-out splits (seed 20261026), soft object labels |
| `benchmark/` | ground truth label files and standalone scorers |
| `viewer/` | static viewer, reads `viewer/data/*.json` |
| `docs/DEMO_VIDEO.md`, `docs/demo/` | demo video plan, its figures and title cards |
| `web/` | everything outside the experiments: the hosted site (report pages `index.html`, `en.html`, `logs.html` and their `assets/`, the labeling page, the video check page, `api/`, `middleware.js`, `vercel.json`), its builders and the demo video tools in `web/tools/`, and the video check worker in `web/worker/` |

## Running it

The code runs in WSL2 Ubuntu 24.04 with Python 3.12.

```bash
bash scripts/setup_wsl.sh          # system packages, venv, pinned packages
cp .env.example .env               # then fill in MODEL_API_KEY
.venv/bin/python scripts/verify_env.py
```

Scripts that call the Meta API only send requests with `--send`. Without it
they prepare the inputs and print the cost. Answers are cached, so a rerun
sends only what is missing.

The final results come from these commands (no API call once the answers are
cached):

```bash
.venv/bin/python scripts/muse_contact.py score --split held_out --segments out/contact/rule_segments_applied.json
.venv/bin/python scripts/number1_final_report.py
.venv/bin/python scripts/number2_score.py --final
.venv/bin/python scripts/number2_depth.py
.venv/bin/python scripts/muse_soft.py score
```

## Data

- OpenGraph Ego-Tactile (CC-BY-4.0): 72 episodes, both hands with tactile gloves.
- HOT3D (Meta, Project Aria): the 136 Aria sequences with hand and object poses.

Neither dataset is in this repository.

## Limits

- The robot check is geometric, not a physics test, and uses one robot hand,
  the Allegro.
- The number 2 result depends on the depth limit. It holds at the 5 mm limit
  fixed in advance and at 10 mm, not at 2 mm.
- The object name alone predicts copyability better than our score when the
  object was seen in training, and our score does not rank grasps inside one
  object.
- The soft object step does not pass its rule, so soft objects are not kept
  out of the score reliably.
- The held-out set for number 2 has three people. Everything is tested on
  lab data.
