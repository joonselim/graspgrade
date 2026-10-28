# GraspGrade measurement plan

Version 2026-10-27, updated 2026-10-28 for the October 28 proposal
([PROPOSAL.md](PROPOSAL.md)): comparison 6, the extra rows and the soft
object check. This page is the single source for what we compare and on
which data. It brings the work back to the comparisons promised in the
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

### Timing shift for every arm (written 2026-10-28, before any arm A or C call)

Until now only arm B got a timing shift learned from the glove. Arms A and C
also see visible contact, which starts before glove pressure, so without a
shift they are judged unfairly against the glove. Each arm therefore gets
its own shift, learned on tuning episodes only, and both versions are
reported.

- Shift: the median start error and the median end error (prediction minus
  glove, boundaries matched one to one within 15 frames, the code of
  `contact_cut.Score`) on tuning, with the sign turned. Starts and ends get
  separate shifts.
- Arm A: the shift is measured on the tuning episodes that are sent to arm A
  for the prompt check (P4: episodes 0, 30, 53) and applied to every held-out
  arm A segment.
- Arm C: Muse Spark moves some arm B boundaries and keeps the others. Kept
  boundaries already carry the arm B shift. The arm C shift is measured only
  on boundaries Muse Spark moved, on 100 starts and 100 ends drawn at random
  (seed 20261026) from the arm B output on tuning episodes, and applied only
  to moved boundaries on held-out.
- Rows in the number 1 table: A, A + shift, B, C, C + shift, and the two
  reference rows. The main metric for the shift is boundary F1 at ±5 frames.
- Test: for A + shift against A and for C + shift against C, a paired
  bootstrap over the 22 held-out episodes (1,000 draws, the same episodes for
  both rows in each draw, F1 recomputed from the pooled counts). The shift
  adds value if the 95% interval of the difference lies above 0.
- What this shows: a value that only the glove can give, put on top of a
  Meta model output. It is a constant per arm, so it is reported as a
  calibration from sensor truth, not as a new model.
- Code check (EXP-27): with perfect answers made 6 frames early, the learned
  shifts bring arm A back from 0.863 to 0.956 and arm C from 0.149 to 0.307,
  the no-lag ceilings.

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
- Settled (EXP-32, P2, 50 tuning episodes): leaving one episode out, arm B
  reaches boundary F1 0.153 at ±5 and 0.294 at ±15, IoU 0.683 (always in
  contact 0.623), with 4,880 boundaries for 1,910. The rule for held-out,
  chosen on all 50: gap 2 px, object move 0.05, co-move 0.7, no fill, split
  on a change of nearest object after 3 frames, shift +4 at the start and -1
  at the end. Arm C ceiling with it: 0.302.

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
| 4 | Unseen objects: comparisons 1 and 2 again, with every learned part of our score refit without the object being scored (see "Leave one object out on held-out" below) | Held-out persons grasp the same objects as the tuning persons (EXP-20), so the object-only reference may beat our score on held-out. Our score can also rate an object it has never seen. The object-only reference cannot (0.5 by construction). |
| 5 | Three Muse Spark rows on the same clips: Muse Spark alone (baseline prompt), Muse Spark checklist grades with fixed equal weights (no learning), Muse Spark checklist grades with a logistic regression trained on the 400 score_train clips | Splits what the checklist adds from what learning on the labels adds (see "Checklist against learning" below) |
| 6 | Our score with the object rate added when 0, 5, 20 or all labeled tuning clips of the object exist (see "Object rate as an add-on" below) | Our score is the default for objects with no labels. The object rate is an add-on for the few objects that have them |
| extra | AUC inside each object, and AUC for each reason a clip fails (see "What the score can and cannot tell apart" below) | Shows whether the score ranks grasps or only objects, and which failures it sees |

### Metrics

- AUC with a 95% bootstrap interval. The bootstrap resamples the 51
  held-out sequences, not single clips, because clips from one sequence are related.
- AUC per held-out person (P0001, P0010, P0012), because the copyable share
  ranges from 30% to 73% between them.
- Share of copyable clips in our top 20% against the base rate.
- All clips, and one-hand clips only (about a third of clips have both hands
  on the object).

### Muse Spark run in two steps, with a gate (written 2026-10-28, before any P7 call)

The free check of EXP-19 used stand-in grades. The P1 pilot (EXP-23) showed
that Muse Spark grades match them in 27 of 30 cases, but on 5 clips only.
Before paying for all held-out clips, the real Muse Spark grades must show the
same signal on tuning clips.

- P7a: the checklist call and the baseline call on the 400 score_train
  clips only (tuning persons). About $7.7 at the EXP-23 price.
- Gate, computed on these 400 clips only, v2 label, nothing from held-out:
  - model: logistic regression on the six Muse Spark object item grades
    (size, shape, thickness, width, handle, free space), as in EXP-19,
  - test 1: leave one person out, mean AUC over the six tuning people,
  - test 2: leave one object out, pooled AUC with a 95% interval that
    resamples sequences.
- Decision rule (the EXP-18 and EXP-19 bar):
  - pass if both AUCs are 0.65 or more: send P7b,
  - stop if either is below 0.60: P7b is not sent, and the next step is
    decided with the user,
  - in between: weak. Decided by the user on 2026-10-28, before P7a: P7b is
    sent. EXP-26 showed that 400 clips give 0.657 even with perfect grades,
    so "weak" is the expected verdict for a good grader.
- Reported with the gate, but not part of it: the same two tests on the
  EXP-19 stand-in grades of the same 400 clips, the Muse Spark baseline AUC
  per person, and how often each Muse Spark item equals its stand-in.
- P7b: the checklist call and the baseline call on all 1,426 held-out clips.
  About $27.6.
- Code check (EXP-26, stand-in grades, no Muse): perfect grades on these
  400 clips give 0.657 and 0.666, so the gate passes by a small margin. With
  random grade errors at the P1 agreement level (about 90%) the expected
  verdict is "weak". The rule above is unchanged.

### Checklist against learning (written 2026-10-28, before any P7 call)

The main comparison (our score against Muse Spark alone) mixes two things:
asking Muse Spark through the checklist, and learning weights from the
labels. Comparison 5 separates them. All three rows use Muse Spark answers
only, on the same held-out clips.

| Row | Score | What it adds over the row above |
| --- | --- | --- |
| R1 | Muse Spark alone: the baseline probability | (start) |
| R2 | Mean of the seven Muse Spark checklist grades (finger occlusion and the six object items), equal weights. High always means "easier for the robot hand", set from the item meaning before any data (EXP-19), so no learning is needed. | asking through the checklist |
| R3 | Logistic regression on the same seven grades, trained on the 400 score_train clips | learning from the labels |

- Tests on held-out, paired bootstrap over the 51 held-out sequences (1,000
  draws, the same sequences for both rows in each draw): R3 minus R2
  (value of learning, the main question of this section), R2 minus R1, and
  R3 minus R1. A difference counts as a gain if its 95% interval lies above 0.
  All three are reported whatever they show.
- Our full score (R3 plus the local and SAM checks) stays comparison 1, and
  its gain over R3 is reported the same way.
- Power, checked for free before P7 with stand-in grades (EXP-27): how often
  a held-out sized set (3 persons) finds R3 minus R2 above 0, with stand-in
  grades as they are and with 15% of grades replaced at random (about the P1
  agreement). It is a check on the design, not a result.
- Result (EXP-27): on tuning with stand-ins, R3 minus R2 is +0.067 (0.014 to
  0.120). Pseudo held-out sets of 3 persons find it in 95% of sets, and in
  77% with 15% of grades replaced at random.

### Leave one object out on held-out (written 2026-10-28, before computing)

Reported in the final run next to the main results, whichever way it comes out.

- For each object in the held-out clips, every part of our score that learns
  from labels is fit again without that object: the score weights
  (score_train clips), the hand shape fit model and its cut points (item_train
  clips). Cut points that use no labels (SAM check terciles) stay as they are.
  The refit score then rates the held-out clips of that object.
- The scores of all held-out clips are pooled into one AUC with a 95% interval
  that resamples the 51 held-out sequences, plus AUC per held-out person.
- Next to it, on the same clips: Muse Spark alone (it learns nothing from
  our labels, so it is unchanged), Muse Spark plus our checks refit the same
  way, and the object-only reference, which has no share for an unseen object
  and gives every clip the training base rate (AUC 0.5).
- The main comparisons (1 to 3) stay as they are. This one shows whether our
  score keeps its value on objects it has not seen.

### Object rate as an add-on, comparison 6 (written 2026-10-28, before computing)

The October 28 proposal makes our score the default and the object rate an
add-on for objects that already have labels. Comparison 6 measures what the
object rate adds as the number of labeled clips of the object grows.

- Labeled clips: the copyable v2 labels of the tuning clips (all 2,385).
  Every held-out object has 38 or more tuning clips.
- k = 0, 5, 20 and all. For k = 5 and 20, k clips of the object are drawn
  at random from its tuning clips. "All" is every tuning clip of the object.
- Object rate from k clips: (copyable clips + 1) / (k + 2), so a rate from
  few clips stays near one half.
- Combined score: a logistic regression with two inputs, the logit of our
  score and the logit of the object rate. One model per k.
  - Trained on the 400 score_train clips. The our score input of a training
    clip comes from a score fit without that clip's person (leave one person
    out). Its object rate is drawn only from tuning clips of other persons.
  - Applied to held-out clips, with our score from the final fit and the
    object rate drawn from the tuning clips of the object.
- k = 0 is our score alone, which is comparison 1.
- Draws: 20 for k = 5 and k = 20 (seed 20261026 plus the draw number), each
  with its own training and held-out rates. The reported AUC is the mean over
  the 20 draws. k = all has one draw.
- Rows for each k on held-out: our score, the object rate alone, and the
  combined score.
- Tests, paired bootstrap over the 51 held-out sequences (1,000 draws,
  bootstrap draw b uses rate draw b modulo 20): combined minus object rate
  alone (does our score add when labels exist) and combined minus our score
  (does the rate add). A difference counts as a gain if its 95% interval lies
  above 0. All are reported whatever they show.
- Objects with no labels are comparison 4, not this one.
- Code check (EXP-31, tuning, stand-in grades): object rate alone 0.722,
  0.770, 0.777 and combined 0.749, 0.773, 0.783 at k = 5, 20, all (our score
  0.721). A check of the code, not a result.

### What the score can and cannot tell apart, extra rows (written 2026-10-28, before computing)

Descriptive rows with no pass or fail. Held-out clips, with the scores of
comparisons 1 to 5: our score, Muse Spark alone, R3 and the object-only
reference.

1. AUC inside each object.
   - Main number: among pairs of held-out clips of the same object, one
     copyable and one not, the share where the copyable clip has the higher
     score (ties count half). This is the mean of the AUC per object,
     weighted by its number of pairs. 95% interval resampling the 51
     held-out sequences.
   - The object-only reference gives one score to every clip of an object,
     so it is 0.5 here by construction.
   - Also reported: the same number inside each object and person, and a
     table of AUC per object for objects with 5 or more clips of each kind
     (27 of 33 objects on held-out).
   - Pairs from the same object are about 3% of all copyable and not
     copyable pairs on held-out, so this number can be low while the pooled
     AUC is high.
2. AUC for each reason a clip fails.
   - Reason: the first condition of the v2 label that fails, in the order
     grip, reach, collision (`failed_condition` in
     `benchmark/number2/clips.csv`). Each failing clip has one reason.
   - For each reason, the AUC of copyable clips against the clips that fail
     for that reason. On held-out: 662 copyable, 80 grip, 146 reach, 538
     collision. 95% interval resampling sequences.
   - The grip row has few clips, so its interval is wide.
- Code check (EXP-31, tuning, stand-in grades): our score 0.574 inside
  objects, 0.775, 0.693, 0.723 against grip, reach and collision fails. A
  check of the code, not a result.

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
- Result (EXP-25): undistortion fails the rule (hand found -10.3 points, 2D
  error 10.8% worse, hand shape error 6.9% better). Decided: raw preview,
  fisheye reported as a limit.
- Comparison 2 in code: `number2_score.py` took the object share from the
  400 score_train clips only. This page says the tuning clips, and EXP-17
  (0.741) used all of them. Fixed 2026-10-28, before any held-out scoring:
  the final run takes it from all 2,385 tuning clips, the same source as
  comparison 6 (check mode keeps the item_train clips, so its test clips stay
  out). The unseen object row uses the base rate of the same clips.

## Soft object check (written 2026-10-28, before any call)

The October 28 proposal limits the score to rigid objects. A soft object
changes shape in the hand, so the object items do not settle whether a robot
hand can copy the grasp. One more Muse Spark question per clip takes these
clips out of the score.

### Ground truth

- Every object name is labeled rigid, squeezable or soft by the two team
  members, each on their own, from the name and one frame of the object,
  before any soft check call (pilot included).
  - The sheet is `splits/softness_labels.csv`: 33 HOT3D objects and 71
    OpenGraph target object names, with clip and segment counts per split.
  - When both are done, they settle disagreements together in the `agreed`
    column. Agreement before that (share and Cohen's kappa) is reported.
  - The sheet is committed before the first call.
- Definitions (the same words go into the prompt):
  - rigid: keeps its shape in a normal grasp (a can, a mug, a hard plastic
    box, a tool),
  - squeezable: keeps its shape, but its walls give a little under a firm
    squeeze (a plastic squeeze bottle, a paper carton, a thin plastic cup),
  - soft: changes shape in a normal grasp (a sponge, cloth, a glove, bubble
    wrap, a bag, a plant).
- HOT3D tracks every object as a rigid body, and the copyable label treats
  it so. Some objects (bottle_bbq, bottle_mustard, bottle_ranch, carton_milk,
  carton_oj, 11.9% of held-out clips) may be labeled squeezable. They stay
  in the score.

### Data

- HOT3D: all 1,426 held-out clips. Frames f1, f2 and f3 of the existing Muse
  frames (contact start, one third and two thirds into the contact).
- OpenGraph: contact segments of the cleaned glove (the number 1 ground
  truth), per hand. The object of a segment is the most common
  `annotation.target_object` over its frames (in the median segment every
  frame has the same object). Three segments whose name is a list of objects
  or "no object" are left out.
  - all 286 held-out segments,
  - plus every tuning segment whose object is labeled soft.
- Why tuning segments too: held-out OpenGraph has about a dozen segments
  with soft looking objects (sponges, bubble wrap). The soft check learns
  nothing from labels. The prompt is fixed here, and the answer is used as
  it is. So tuning segments are as new to it as held-out segments. Held-out
  results are also shown alone.
- Frames: the first frame of the segment, one third in and two thirds in,
  from the 720p episode video.

### Call

- One Muse Spark call per clip or segment, 3 frames, reasoning effort low
  (as in the other calls). The prompt, fixed now (`{hand}` is left or right):

  > These are 3 frames in time order from a first-person video. The person's
  > {hand} hand is holding or touching an object. Look at that object only.
  > How does it behave in a normal grasp? rigid: it keeps its shape (for
  > example a can, a mug, a hard plastic box, a tool). squeezable: it keeps
  > its shape, but its walls give a little under a firm squeeze (for example
  > a plastic squeeze bottle, a paper carton, a thin plastic cup). soft: it
  > changes shape in a normal grasp (for example a sponge, cloth, a glove,
  > bubble wrap, a bag, a plant). unsure: you cannot tell from these frames.
  > Answer with JSON only: {"object": "<short name>", "softness": "rigid,
  > squeezable, soft or unsure", "reason": "<one sentence>"}

- A clip is held back ("needs a touch check", no score) if the answer is
  soft or unsure. Rigid and squeezable keep their score.
- A reply that does not parse (no JSON, or softness not one of the four
  words) is sent once more. If it still does not parse, it counts as unsure,
  and the count is reported.
- Pilot before the full run, to measure cost and parsing: 10 HOT3D
  score_train clips and 10 OpenGraph tuning segments of objects labeled
  rigid or squeezable (seed 20261026). None of them is in the evaluation.
  The prompt does not change after the pilot, except the line on the JSON
  format if 3 or more of the 20 replies do not parse.
- This is a new prompt. "No further revision" in the prompt revision section
  covers the checklist and arm A, and those do not change.

### Metrics and pass

- Caught: the share of OpenGraph segments with a soft object that are held
  back. For all of them, held-out alone, tuning alone, and per object. 95%
  interval resampling objects, because segments of one object are related.
- Held back by mistake: the share of HOT3D held-out clips held back. All of
  them count, because the score covers every HOT3D object. Also by human
  label (rigid, squeezable), per object, and on OpenGraph rigid and
  squeezable segments.
- A table of the human label against the Muse Spark answer.
- Our score AUC on the HOT3D held-out clips the check keeps, with its
  interval resampling sequences, next to the AUC on all 1,426 clips.
  Comparisons 1 to 6 stay on all 1,426 clips. The check does not change them.
- Pass: 70% or more of the OpenGraph soft object segments caught, and 10% or
  less of the HOT3D held-out clips held back.
- Too few: if fewer than 20 segments or fewer than 4 objects are labeled
  soft, the caught share is reported with its counts, and no pass or fail is
  given.
- If it does not pass: the viewer still shows the touch check list, marked
  as not validated, and the summary does not claim that the check works.
- Code (2026-10-28, no call): `scripts/muse_soft.py` (`segments`, `frames`,
  `send --set pilot|hot3d|opengraph`, `score`), prompt in
  `scripts/prompts/soft_object.txt`. Sending stops until every label is
  agreed. Cost at measured token counts: HOT3D held-out $10.46, OpenGraph
  held-out $2.60, plus the tuning soft segments and the pilot (under $1).

## One prompt revision on tuning (written 2026-10-28, before sending any revised prompt)

The first real answers on tuning (EXP-29, error analysis in EXP-30) show two
systematic Muse Spark errors. Prompts may still change, because nothing has
been sent on held-out. Each prompt is revised once, tested on the same tuning
data, and the version for held-out is picked by a rule fixed here.

- Number 2 checklist, version 2: Muse Spark estimates the longest, middle and
  shortest side of the whole object in cm, using the hand (about 19 cm) and
  common objects as rulers. Our code turns the three numbers into the size,
  shape, thickness and width grades with the same anchors as before (EXP-19
  table). Handle, free space and finger occlusion stay graded as before.
  Reason: version 1 calls 72 of 103 mid-size objects small and 56 of 133 wide
  objects narrow, which looks like a scale and definition problem.
  - Sent on the same 400 score_train clips.
  - Rule: version 2 is used for P7b if its R3 (learned weights on the seven
    grades, leave one person out, mean over people) is higher than version 1
    (0.638). Otherwise version 1 stays. Ties keep version 1.
- Number 1 arm A, version 2: the prompt says that a hand which lets go of an
  object, even for a moment between two grasps, is not touching, and asks to
  check for a visible gap between fingers and object in every frame.
  Reason: version 1 marks a release of 0.2 to 1 s as "not touching" in only
  about 10% of cases, so it joins touches and finds 68 of 190 boundaries.
  - Sent on the same tuning episodes 0, 30 and 53.
  - Rule: version 2 is used for held-out if its boundary F1 at ±5 frames with
    the shift learned on the other two episodes (cleaned glove) is higher than
    version 1 (0.132). Otherwise version 1 stays. Ties keep version 1.
- Muse Spark alone (R1) keeps its prompt. It is the comparison the proposal
  defines (the same question, no checklist), and its weakness on tuning (0.52)
  is that it does not know which objects fail, not a wording problem. This
  is reported as a limit: the checklist got one revision on tuning, R1 none.
- No further revision after this one. Both versions are reported on tuning.
- Result (EXP-30): checklist version 2 R3 0.625 against 0.638, arm A version
  2 F1 0.120 against 0.132. Both prompts stay at version 1 for held-out.
  The arm A shift for held-out is +7 at the start and -7 at the end.

## Replies that never parse (written 2026-10-28, before any paid run)

A Muse Spark reply that does not parse (no JSON, a missing field, a value
outside the allowed answers) is never cached as done. The code keeps it under
`unparsed/`, and a rerun of the same send command sends it again. This
section fixes what happens to a reply that still does not parse after that.
It covers every paid Muse Spark run from now on, tuning and held-out. The
soft object check has its own rule in its section.

- Resend: after the full run of a step, the same command is run once more.
  It sends only the calls without a parsed answer. Whatever still has no
  parsed answer after this second pass gets the fallback below. No third pass.
- An API error (the call raises, no reply) is not a parse failure. Such calls
  are resent until they get a reply.
- Fallbacks, chosen so that a missing answer counts against the method that
  failed to give it:

| Call | Fallback after the second pass |
| --- | --- |
| Arm A (10 frames per call) | Every frame of the call counts as "not touching" for both hands. The episode stays in the number 1 table. |
| Arm C (one boundary per call) | The arm B boundary is kept, as if Muse Spark had answered "no change". It does not enter the arm C shift, which uses moved boundaries only. |
| Number 2 checklist or Muse alone (baseline) | The clip is dropped from every number 2 row (comparisons 1 to 6 and the extra rows), so all rows keep the same clips. |

- Limit: if more than 5% of the calls of one step (arm A, arm C, checklist,
  baseline) on held-out have no parsed answer after the second pass, the
  results of that step are not reported until the user decides how to go on.
  The 5% check for the checklist and baseline is in `number2_score.py
  --final`. For arms A and C it is the number of calls left to send that the
  send command prints on a third run with `--send` left out.
- Reported with the results: per step, the calls that needed the second pass,
  and the calls and clips that got the fallback.

## Release: the ground truth as a reusable benchmark (written 2026-10-28)

The proposal's problem is that teams filter video with their own rules and
do not publish a score others can reuse. The heaviest and most checked part
of this project is the ground truth, so it is released as a benchmark that
scores any video filter against sensors, whatever our checklist scores.

What is released, and under which license (from the license files that come
with the data: `data/ego-tactile/README.md`, `data/hot3d/assets/license.txt`.
This is our reading of those texts, not legal advice):

| Part | Content | Derived from | License |
| --- | --- | --- | --- |
| Number 1 benchmark | Per episode and hand, contact segments from the cleaned glove (frame ranges at 30 fps), raw glove segments, tuning and held-out split, scoring script (boundary F1 at ±5 and ±15 frames, contact IoU) | OpenGraph Ego-Tactile, CC BY 4.0 | CC BY 4.0, credit to OpenGraph Labs |
| Number 2 benchmark | Clip list (sequence, hand, object, start, length, contact frames), copyable v2 label with the failing condition (grip, reach, collision), depth limit curve, tuning and held-out split, scoring script (AUC per person, sequence bootstrap, top 20%) | HOT3D hand data (CC BY-NC-SA 4.0), object poses (CC BY-SA 4.0), object models (CC BY-SA with no sale) | CC BY-NC-SA 4.0, credit to Meta HOT3D, no sale of object model data |
| Code | The scripts that rebuild both benchmarks from the original downloads | | our own code license |

- No raw video, sensor or mocap file is copied. Users download the original
  datasets themselves (HOT3D needs a sign-up) and run the scripts.
- The copyable label uses the HOT3D hand data, so the number 2 benchmark
  cannot be offered for commercial use.
- Every release file names its source dataset and license.
- Done (EXP-28): `benchmark/` with both label files, license notes, example
  inputs and standalone scorers that reproduce the project numbers.

## Budget

Estimates at measured prices, $150 in total.

| Item | Cost |
| --- | --- |
| Spent so far (P1 pilot, arm A on 3 tuning episodes, P7a, one prompt revision on tuning, P2 SAM on 47 tuning episodes, P3 SAM on 22 held-out episodes) | $36.40 |
| Number 1, Muse arm A at 5 fps (held-out, plus prompt tuning on a few tuning episodes) | about $14 |
| Number 1, Muse arm C: 1,790 held-out boundaries from the applied arm B (EXP-33, $23.31) plus 200 tuning boundaries for the arm C shift ($2.61) | about $26 |
| Number 2, SAM: all held-out clips plus 400 tuning clips, overlapping clips in one request | about $27 |
| Number 2, Muse: checklist call and baseline call per clip, 1,826 clips (EXP-23 measured $0.0097 per call). P7a 400 tuning clips, then P7b 1,426 held-out clips unless the gate says stop | about $7.7 + $27.6 |
| Soft object check: 1,426 HOT3D clips at about $0.0075 and about 330 OpenGraph segments at about $0.0094 (3 frames each), plus a pilot of 20 | about $14 |
| Total (spent $36.40, still to send: arm A held-out $11.86, arm C about $26, number 2 SAM $26.48, P7b $27.6, soft check about $14) | about $142 |

Muse costs assume about 1,100 tokens per frame (measured on 720p OpenGraph
frames). Each arm is first run on one tuning episode or a few clips to
measure the real cost before the full run.
