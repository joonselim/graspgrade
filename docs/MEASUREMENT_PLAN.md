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
- Measured ceilings (EXP-21, pilot episodes, perfect answers): arm A at 5 fps
  0.956 boundary F1 at ±5 frames, arm C 0.149 because arm B finds 52 of 190
  boundaries. Before the paid run, either raise arm B recall on the 50 tuning
  episodes, or let arm C also sample inside long arm B contacts. To decide
  with the user.
- Decided 2026-10-28: raise arm B recall on the 50 tuning episodes first,
  including an option that stops long holds from being cut into pieces. Arm C
  stays as planned and does not sample inside arm B contacts.
- EXP-22 (pilot): letting arm B split contacts when the nearest object
  changes doubles its boundary F1 and the arm C ceiling (0.32), but over-splits
  long holds. The option is in the arm B grid, to be settled on the 50 tuning
  episodes.

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

### Label method check (written 2026-10-27, before computing)

Feedback on EXP-10 showed that the current label may measure a retargeting
side effect. With wrist-to-fingertip vectors scaled by the hand size ratio,
Allegro fingertips float about 25 mm off cans that the person holds firmly.
The copyable share per person also falls as the scale factor rises
(r = -0.78 over 9 people). Three label methods are compared on tuning clips
only, with the EXP-10 thresholds unchanged (near 1.5 cm, deep 1 cm, wrist
shift up to 2 cm):

1. vector retargeting, scale = Allegro length / person length (current),
2. vector retargeting, scale 1.0,
3. position retargeting: Allegro fingertips and finger bases are fitted to
   the human positions, with a free hand base.

How we choose, fixed before looking at the results:

- Fidelity: on clips where the human grasp is solid (thumb and one more
  human fingertip within 15 mm of the surface), the median distance from each
  Allegro fingertip to the matching human fingertip. Lower is better.
- Hand size: the rank correlation between the hand length of the person and
  the label, inside each object (weighted mean). It should be near 0.
- The copyable share must stay between 20% and 80%.
- Rule: among methods with an absolute hand size correlation of 0.1 or less
  and a share inside the band, take the one with the best fidelity. If none
  passes, we report that and revisit the definition.
- Copyable share per object is reported, but it is not used to choose.

Result (EXP-15): no method passes. `ratio` (the EXP-10 label) has poor
fidelity (31 mm) and depends on hand size (+0.19). The faithful methods
(`scale1` 6.9 mm, `position` 11.4 mm) call 94% to 96% of grasps copyable,
outside the band. The label is on hold, and with it every number built on it
(EXP-11 shares, EXP-12 object reference, EXP-13 hand shape check). The
EXP-11 clips are not affected. Next: a new definition with `position`
retargeting, collision checked on the Allegro link meshes, and a reach check
on the human contact points, written here before computing.

### Copyable definition v2 (written 2026-10-27, before computing, EXP-16)

Retargeting and geometry:

- Position retargeting (dex-retargeting Allegro offline config): Allegro
  fingertips and finger bases fitted to the human ones at the contact frame,
  with a free hand base.
- The Allegro hand surface is points sampled on the visual mesh of every
  link (about 300 per link), placed by forward kinematics.
- The hand may then shift up to 2 cm (1 cm grid), as in EXP-10.

A clip is copyable when, for at least one shift, all three hold:

1. Grip: the Allegro thumb tip and at least one other Allegro fingertip are
   within 1.5 cm of the object surface.
2. Reach: for every human finger (thumb, index, middle, ring) whose tip is
   within 15 mm of the surface, the matching Allegro fingertip is within
   2.0 cm of that human fingertip. The little finger has no Allegro match and
   is not checked.
3. No collision: no Allegro mesh point is more than 5 mm inside the object.

Why these numbers: 1.5 cm is the human contact distance used since EXP-09.
2.0 cm is about the width of an Allegro fingertip. 5 mm allows for mocap and
model error (EXP-09 fingertips sit about 2 mm from the surface).

Acceptance, fixed now:

- Main thresholds above. The copyable share must be between 20% and 80%.
- Hand size: rank correlation of hand length and label inside objects,
  absolute value 0.1 or less.
- Also reported, not used to accept: a table over reach (1, 2, 3 cm) and
  depth (2, 5, 10 mm), the share per object and per person, and which of the
  three conditions fails.
- If the main thresholds fail, the thresholds are not moved. We report the
  result and decide with the user how to change the number 2 question.

Result (EXP-16): accepted. 47.0% copyable on 2,385 tuning clips, hand size
correlation inside objects +0.04 (44% to 54% by person). Failures: collision
25.5%, reach 10.9%, grip 3.9%. The share depends strongly on the depth
limit (18% at 2 mm, 66% at 10 mm), which is reported with every result.
The v2 label replaces the EXP-10 label for number 2. EXP-12 and EXP-13 are
to be recomputed on it (tuning), then the held-out v2 labels are computed
with the same fixed thresholds.

Result (EXP-17, tuning, leave one person out): object-only reference 0.741,
three free checks 0.529, video hand shape 0.521 (its EXP-13 signal came from
the retargeting side effect), video hand shape plus object 0.751, mocap hand
shape upper bound 0.617. The checklist as designed measures video quality,
while the v2 label is mostly about the object. This is the open problem for
number 2 before any paid call.

### Final clip sets

| Use | Clips | Copyable |
| --- | --- | --- |
| Train the score weights | 400 random tuning clips (seed 20261026) from 2,385 | 47.0% in all tuning clips (v2) |
| Report AUC | all 1,426 held-out clips | 46.4% (v2, EXP-20) |

Ground truth is the copyable v2 label (EXP-16), with the thresholds fixed on
tuning. Object shares agree between the splits (rank correlation 0.85, EXP-20).

### Option A check (written 2026-10-27, before computing, EXP-18)

Option A keeps the question and the v2 label, and turns the "object
property" check into several concrete object items that Muse Spark would
rate from the video. Before any paid call we test, for free, whether object
geometry can carry the signal that object identity carries (EXP-17: 0.741).

- Features per object, from its 3D model only: the three sides of its
  oriented bounding box (largest, middle, smallest), flatness (smallest over
  largest), elongation (middle over largest), volume, and fill (volume over
  box volume). Each clip gets the features of its object.
- Model: logistic regression on these features, tuning clips, v2 label.
- Two tests:
  1. leave one person out, AUC per person and the mean over the six
     people, next to the object identity reference (0.741),
  2. leave one object out: the model never sees the object it scores.
     AUC over all clips pooled, with a 95% interval that resamples
     sequences. The object identity reference cannot score an unseen object,
     so it is at 0.5 here by construction.
- Decision rule:
  - pass if both AUCs are 0.65 or more: go ahead with option A,
  - fail if either is below 0.60: option A has no basis, move to option D or C,
  - in between: weak, decide with the user.

Result (EXP-18): pass. Object geometry from the 3D models reaches AUC 0.695
leaving one person out (object identity 0.742) and 0.673 leaving one object
out (0.645 to 0.698, object identity 0.5). The largest side of the object is
the strongest feature. Option A goes ahead: the object property check becomes
concrete object items for Muse Spark.

### Object items for Muse Spark (written 2026-10-27, before computing, EXP-19)

Six items. Muse Spark rates each one from the clip frames as high, mid or
low. High always means "easier for a large four-finger robot hand". The
direction was set from what the items mean, not from the data. The
anchors use the visible human hand as the ruler (wrist to middle fingertip,
about 19 cm).

| Item | High | Mid | Low |
| --- | --- | --- | --- |
| Size next to the hand: longest side | up to 0.8 hand lengths (about 15 cm) | 0.8 to 1.3 (15 to 25 cm) | over 1.3 (over 25 cm) |
| Shape: middle side over longest side | 0.6 or more (blocky) | 0.3 to 0.6 | under 0.3 (stick-like) |
| Thickness: shortest side | 4 cm or more | 2 to 4 cm | under 2 cm (flat or thin) |
| Width the hand must wrap: middle side | up to 8 cm | 8 to 14 cm | over 14 cm |
| Handle: a part shaped to be held | yes | partly | no |
| Free space: gap to the nearest other object at the grasp | 10 cm or more | 4 to 10 cm | under 4 cm |

Free check with stand-ins, before any paid call:

- Grades computed from the truth instead of Muse Spark: the first four from
  the 3D model box, handle by eye from one video frame per object (judged
  before looking at any label, EXP-19 contact sheet), free space from mocap
  object poses at the contact frame.
- The six grades in a logistic regression, tuning clips, v2 label: leave one
  person out (mean AUC over the six people) and leave one object out
  (pooled AUC).
- Pass if both are 0.65 or more (the EXP-18 bar). This tells whether the
  coarse three-level items keep the signal of the exact numbers (EXP-18: 0.695
  and 0.673). Fail if either is below 0.60. It does not tell whether Muse
  Spark reads the items correctly.

Result (EXP-19): pass. Six stand-in grades reach 0.683 leaving one person out
and 0.666 leaving one object out, close to the exact sizes. Size next to the
hand does most of the work. Handle and width point the other way from their
meaning once size is known. The regression learns these directions on
tuning clips. Before the full paid run, a short Muse Spark pilot on a few
tuning clips compares its grades with these stand-ins.

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
- Fisheye check (EXP-25, written before computing). Free, tuning persons
  only. Contact frames of tuning clips, MediaPipe on the raw preview and on
  the same frame turned into a pinhole image with the dataset lens model.
  Measured against the mocap hand (UmeTrack landmarks projected with the
  dataset lens model): hand found, 2D point error over hand size, and the
  error of the ten hand shape features from MediaPipe 3D points. We
  undistort only if undistortion raises the share of frames with the hand
  found by 5 points or more, or lowers the median 2D error by 20% or more,
  or lowers the median hand shape feature error by 20% or more. Otherwise
  we keep the raw preview and report fisheye as a limit. Muse frames stay
  raw in both cases, because testing Muse on undistorted frames costs credit.

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
