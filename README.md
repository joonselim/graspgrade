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

The full measurement plan is in [docs/MEASUREMENT_PLAN.md](docs/MEASUREMENT_PLAN.md).
Every experiment has a page with its setup, numbers, cost and figures in
[docs/experiments/](docs/experiments/README.md).

## Status

The free work is done. The paid runs (SAM 3.1 and Muse Spark) are prepared
and have dry runs with cost estimates, but most have not been sent yet.

| Part | State |
| --- | --- |
| Data checks, splits, calibration | done (EXP-01 to EXP-09) |
| Number 1: contact timing, SAM rule (arm B) | code done, pilot on 3 episodes (EXP-14, EXP-22, EXP-24) |
| Number 1: Muse Spark arms A and C | code done, best possible scores measured (EXP-21) |
| Number 2: copyable ground truth | done, label v2 accepted on rules fixed in advance (EXP-15, EXP-16, EXP-20) |
| Number 2: clips | 2,385 tuning and 1,426 held-out clips, cut from motion capture (EXP-11) |
| Number 2: checklist | object items checked with stand-ins (EXP-17 to EXP-19), paid pilot on HOT3D done (EXP-23), fisheye kept raw (EXP-25) |
| Viewer | static page with demo data from tuning clips |

## Main findings so far

- SAM 3.1 at 10 fps tracks hands and the grasped object as well as at 30 fps
  (EXP-05), which cuts the SAM cost to a third.
- The glove signal needs cleaning: pressure flickers around the threshold.
  The main ground truth joins dropouts and drops blips under 0.1 s (EXP-14).
- A simple copyable label built by scaling the human hand to the robot was
  mostly measuring a side effect of the retargeting (EXP-15). The accepted
  label fits the robot fingertips to the human ones and checks collisions on
  the real robot link meshes (EXP-16). 47% of grasps are copyable, and the
  label does not depend on the size of the person's hand.
- Under that label, copyability is mostly a property of the object. Seven
  numbers about object size and shape predict it about as well as knowing
  the object (AUC 0.70 against 0.74), and they still work on objects the model
  has never seen (0.67), which the object name cannot do (EXP-18).

## Layout

| Path | What |
| --- | --- |
| `scripts/` | all code, one script per step, each with a usage note at the top |
| `scripts/prompts/` | Muse Spark prompts |
| `docs/MEASUREMENT_PLAN.md` | what is compared, on which data, decided before the results |
| `docs/experiments/` | one page per experiment, with figures |
| `docs/DATA_NOTES.md`, `docs/ENVIRONMENT.md` | facts about the data, setup notes |
| `splits/` | fixed tuning and held-out splits (seed 20261026) |
| `viewer/` | static viewer, reads `viewer/data/*.json` |

## Running it

The code runs in WSL2 Ubuntu 24.04 with Python 3.12.

```bash
bash scripts/setup_wsl.sh          # system packages, venv, pinned packages
cp .env.example .env               # then fill in MODEL_API_KEY
.venv/bin/python scripts/verify_env.py
```

Scripts that call the Meta API only send requests with `--send`. Without it
they prepare the inputs and print the cost. Examples:

```bash
.venv/bin/python scripts/sam_episodes.py --split tune          # number 1, SAM plan and cost
.venv/bin/python scripts/muse_contact.py a --split held_out    # number 1, Muse Spark alone, cost
.venv/bin/python scripts/sam_hot3d.py --role held_out          # number 2, SAM plan and cost
.venv/bin/python scripts/muse_hot3d.py --task checklist --role held_out
.venv/bin/python scripts/number2_score.py --check              # number 2 code check on tuning clips
```

To look at the viewer locally:

```bash
cd viewer && python3 -m http.server 8765
```

## Data

- OpenGraph Ego-Tactile (CC-BY-4.0): 72 episodes, both hands with tactile gloves.
- HOT3D (Meta, Project Aria): the 136 Aria sequences with hand and object poses.

Neither dataset is in this repository.

## Limits

- The robot check is geometric, not a physics test, and uses one robot hand,
  the Allegro.
- The copyable share depends on the allowed collision depth (18% at 2 mm,
  47% at 5 mm, 66% at 10 mm). We fixed 5 mm before computing and report the
  others.
- The held-out set for number 2 has three people.
- Everything is tested on lab data first.
