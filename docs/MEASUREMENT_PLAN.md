# GraspGrade measurement plan

Version 2026-10-27. This page is the single source for what we compare and
on which data. It brings the work back to the comparisons promised in the
proposal (October 21). Where an experiment page says something different,
this page wins.

## Why this page exists

After the first experiments the work drifted from the proposal:

- Number 1 moved to a learned model with 13 features, and the Muse Spark
  arms were not run yet.
- Number 1 used the dataset labels as the main ground truth, while the
  proposal promised the touch glove signal.
- Number 2 had two clip sets (EXP-07 and EXP-11) without saying which one
  gives the reported numbers.

The plan below fixes all three. Nothing here needs new data or a new model.

## General rules

- Every threshold, weight and prompt choice is made on the tuning split.
- Every reported number comes from the held-out split.
- Tolerance windows and metrics are fixed now, before any held-out run.
- Video is the only input. Glove and motion capture data are used only to
  score and to check.

## Number 1: from video alone, when does a touch start and end?

### Data and ground truth

- OpenGraph Ego-Tactile, 72 episodes, 30 fps. Tuning: 50 episodes.
  Held-out: 22 episodes (`splits/opengraph.json`).
- Main ground truth (fixed 2026-10-27): the touch glove, cleaned. A hand is
  in contact when the sum of its 80 taxels is above 800 (EXP-04). Pressure
  dropouts of 0.1 s or less are joined, and pressure blips shorter than
  0.1 s are dropped (EXP-14). This removes threshold flicker that no video
  method can see.
- Checks, reported next to it: the raw glove (no cleaning) and the dataset
  contact labels. The labels agree with the raw glove on 98.7% (left) and
  97.2% (right) of frames.

### What we compare (the three arms from the proposal)

| Arm | What it is | Meta models |
| --- | --- | --- |
| A. Muse Spark alone | Muse Spark looks at video frames and says, per frame and hand, whether the hand touches an object. No SAM. | Muse Spark |
| B. SAM rule | SAM 3.1 masks of hands (`hand`) and objects (`small object`) at 10 fps. Contact signal from mask overlap plus co-movement (the object moves with the hand, after removing camera motion). Thresholds set on tuning episodes. | SAM 3.1 |
| C. SAM rule plus Muse Spark check | Arm B, then for each start and end, Muse Spark sees about 7 frames around it at 30 fps and picks the first or last touch frame, with a reason and a confidence. | SAM 3.1, Muse Spark |

Reference rows, so the numbers can be read: "always in contact" and
"never in contact". The logistic regression from EXP-06 stays as an
appendix. It is not one of the compared arms.

Details fixed now:

- All arms output contact segments per hand at 30 fps, and all use the same
  rules to join short gaps and drop very short segments (set on tuning).
- Arm A uses frames at 5 fps (one every 6 frames). Several frames go in one
  call, with their times.
- Arm B keeps the start shift measured on tuning: visible contact begins
  about 0.2 s before glove pressure (EXP-06, to be measured again on all 50
  tuning episodes). This is the correction the proposal asked for.

### Metrics

- Boundary F1 at ±5 frames (0.167 s) and ±15 frames (0.5 s). Starts and ends
  are matched one to one, each only to its own kind.
- Contact IoU over all frames.
- Reported per hand and pooled, on the cleaned glove (main), the raw glove
  and the labels.
- The start and end delay of arm B against the glove, on tuning and on held-out.

### Open decisions from the pilot (EXP-14)

- Decided: the raw glove flickers around the threshold (on three episodes,
  21 of 124 segments are shorter than 0.2 s and 28 gaps are 0.1 s or less),
  so the main ground truth is the cleaned glove (see Data and ground truth).
- Arm B joins touches that follow each other, because the hand stays near
  some object. Proposed: also break a contact when the nearest object
  changes or the hand leaves for a moment, chosen on tuning episodes.
- Arm C only refines the boundaries arm B finds, so arm B sets its recall.

### What we expect

The glove measures pressure, which video cannot see. Arm B already showed
starts about 0.2 s early, and gaps where a hand rests on an object without
pressure. All three arms may score low at ±5 frames. The claim is the
comparison between arms, and we report ±5 frames as it comes out.

## Number 2: does our score pick clips a robot hand can copy?

### Data and ground truth

- HOT3D Aria, the 136 sequences with hand and object poses. Tuning: 6
  people, 85 sequences. Held-out: 3 people, 51 sequences
  (`splits/hot3d_aria.json`).
- Clips: the EXP-11 set, cut from motion capture (two fingertips within
  15 mm of the object, 1 s added before and after, object moved at least
  3 cm, clip passes the dataset QA). One clip belongs to one hand.
  The list is `out/hot3d_clips/clips_labeled.csv`.
- Ground truth: the copyable rule fixed on tuning in EXP-10 (near 1.5 cm,
  deep 1 cm, wrist shift up to 2 cm).
- EXP-07 (2,152 object moves) was only for pricing. It is not used for any
  reported number.

### Final clip sets

| Use | Clips | Copyable |
| --- | --- | --- |
| Train the score weights | 400 random tuning clips (seed 20261026) from 2,385 | 53.3% in all tuning clips |
| Report AUC | all 1,426 held-out clips | 51.7% |

### Our score

- Seven checks, each graded high, mid or low (1, 0.5, 0): visibility,
  camera stability, contact clarity, action completeness, finger occlusion
  (Muse Spark), object property (Muse Spark), retargeting recoverability
  (MediaPipe and dex-retargeting, Allegro). Since EXP-13 the last one is
  measured as hand shape fit from MediaPipe hand points during contact.
- A logistic regression on the 400 tuning clips turns them into a 0 to 100 score.
- A check with no spread on the tuning clips is dropped from the
  regression and explained in the demo (proposal risk 5).
- Gate: no contact or a scene cut gives 0.

### What we compare

| # | Comparison | Why |
| --- | --- | --- |
| 1 | Our score vs Muse Spark alone, asked the same question with no checklist | The main comparison from the proposal |
| 2 | Object-only reference: the copyable share of each object on the tuning clips, given to each held-out clip | Shows how much object identity alone predicts (26% to 94% by object). It uses object names from the dataset, so it is a reference, not a video method. |
| 3 | Muse Spark alone vs Muse Spark plus our seven checks in one logistic regression | The plug-in framing: GraspGrade adds to an existing filter instead of replacing it |

### Metrics

- AUC with a 95% bootstrap interval. The bootstrap resamples the 51
  held-out sequences, not single clips, because clips from one sequence are related.
- AUC per held-out person (P0001, P0010, P0012), because the copyable share
  ranges from 30% to 73% between them.
- Share of copyable clips in our top 20% against the base rate.
- All clips, and one-hand clips only (about a third of clips have both hands
  on the object).

### Checks before any paid call

- Single-check AUC of the three local checks on tuning clips (EXP-12).
- Object-only reference AUC on tuning, leave one person out.
- If the local checks carry no signal, we strengthen retargeting
  recoverability toward what the proposal says ("whether the hand shape fits
  a robot hand") before spending credit.
- Result (EXP-12): the three local checks give AUC 0.49 to 0.53 for
  copyable, and add nothing to object identity, which reaches 0.70 (leave
  one tuning person out, mean over people). So the retargeting check gets
  hand shape features first (hand opening, finger curl, grasp type), tested
  for free on tuning clips before any paid call.
- Result (EXP-13): hand shape from video (opening, finger curl, spread,
  closure, pinch vs power) predicts copyable at AUC 0.63, object identity at
  0.69, both together at 0.72. The retargeting recoverability check becomes
  "hand shape fit": a logistic regression on the ten video features, graded
  by tuning terciles, trained on tuning clips outside the 400 used for the
  score weights.
- Pooled AUC over people with different copyable shares ranks people
  instead of clips (EXP-12). Per-person AUC is reported next to pooled AUC.

### Open items

- Fisheye: the plan said to undistort before MediaPipe. MediaPipe now runs
  on the raw fisheye preview. Either we undistort, or we report it as a limit.

## Budget

Estimates at measured prices, $150 in total.

| Item | Cost |
| --- | --- |
| Spent so far | $1.53 |
| Number 1, SAM: remaining 47 tuning episodes / 22 held-out episodes | $11.89 / $5.45 |
| Number 1, Muse arm A at 5 fps (held-out, plus prompt tuning on a few tuning episodes) | about $14 |
| Number 1, Muse arm C (held-out boundaries, plus tuning) | about $9 |
| Number 2, SAM: all held-out clips plus 400 tuning clips, overlapping clips in one request | about $27 |
| Number 2, Muse: checklist call and baseline call per clip, 1,826 clips | about $44 |
| Total | about $113 |

Muse costs assume about 1,100 tokens per frame (measured on 720p OpenGraph
frames). Each arm is first run on one tuning episode or a few clips to
measure the real cost before the full run.
