# GraspGrade: Hackathon Proposal

Updated October 28, 2026 (first version October 21, 2026).

This version keeps the idea and the two measurements of the first version.
It adds what we learned in the first three days: what the score is for, the
scope of the score, and a step for soft objects. The exact rules for every
comparison are in [MEASUREMENT_PLAN.md](MEASUREMENT_PLAN.md).

## In one line

GraspGrade takes first-person videos of hands using objects, cuts them into
clips at the moment of contact, and gives each clip a score for how likely it
is that a robot hand can copy the grasp. It works from video alone, also on
objects it has never seen. Clips with soft objects get no score. They are
marked for a touch check.

We check the work against sensors, not against our own guesses. The contact
cut is checked against a touch-sensing glove. The robot score is checked
against motion capture from Meta HOT3D.

Theme: Push the Frontier (robotics).

## The problem

Robot hand teams want human video because there is so much of it and it is
cheap. But most of it cannot teach a robot hand. In a recent study, only 83 of
2,000 internet clips (4%) passed the quality check (Do as I Do).

Fingers get hidden, the camera shakes, and from video alone it is hard to
tell when the hand really touches the object. Each team writes its own filter.
Teams list the reasons for dropping a clip, but they rarely give a score that
others can reuse and check.

The simple option today is to ask a vision model "can a robot hand copy
this?" with no fixed rules. On our tuning data, Muse Spark asked this way is
close to chance (AUC 0.52).

## What the score is for

- **Main use: objects with no labels.** Most video a robot team can collect
  shows objects that nobody has measured. GraspGrade scores those clips from
  video alone, including objects it has never seen.
- **Add-on: objects with labels.** A few objects already have motion capture,
  a 3D model and a robot check. For them, the copyable rate of the object is
  more accurate, and it is added on top of our score. It does not replace the
  score, because it says nothing about a new object.
- **Scope: rigid objects and a geometric check.** The score says whether the
  Allegro robot hand can reach and hold the grasp without going into the
  object. It is a first filter before physics-based retargeting, which costs
  far more per clip. It does not promise that the robot succeeds.
- **Soft objects: no score.** A soft object changes shape in the hand, so its
  size and shape do not settle the question, and touch matters more.
  GraspGrade asks whether the object is soft. If it is, or if the answer is
  unsure, the clip is marked "needs a touch check".

## What we will build

1. **Cut at contact.** SAM 3.1 tracks the hands and the objects at 10 fps. A
   rule on mask overlap and moving together gives contact segments. Muse
   Spark checks the exact first and last touch frame.
2. **Score each clip.** Muse Spark answers a checklist about the object (size
   next to the hand, shape, thickness, width to wrap, handle, free space
   around it) and about hidden fingers. SAM and MediaPipe add visibility,
   contact clarity and hand shape fit. A logistic regression trained on
   tuning clips turns the answers into a 0 to 100 score.
3. **Soft object check.** One more Muse Spark question per clip: rigid,
   squeezable, soft or unsure. Soft or unsure means no score and a touch check.
4. **A simple viewer.** A web page with clips sorted by score, a contact
   timeline, a score card for each clip, and the clips that need a touch check
   in their own list.
5. **A benchmark.** Both ground truths are released as label files with
   scoring scripts, so any team can score its own filter against sensors.

## Where Meta models and data come in

| Piece | What it does in GraspGrade |
| --- | --- |
| SAM 3.1 | Tracks the hands and each object across the video. |
| Muse Spark | Checks the touch frames, answers the checklist and the soft object question. Also the baseline we try to beat. |
| HOT3D (Project Aria) | Motion capture of hands and objects. Our ground truth for whether a robot hand can copy a clip. |

We also use OpenGraph Ego-Tactile (CC BY 4.0, 72 episodes, tactile gloves on
both hands) for the contact ground truth and for soft objects.

## What we will measure

### Number 1: from video alone, when does a touch start and end?

| | |
| --- | --- |
| Ground truth | Touch glove: sum of 80 sensors over 800, gaps of 0.1 s or less joined, pieces under 0.1 s dropped |
| Data | OpenGraph Ego-Tactile, 50 tuning and 22 held-out episodes |
| What we compare | A: Muse Spark alone. B: SAM 3.1 rule. C: SAM 3.1 rule plus Muse Spark check |
| Metrics | Boundary F1 at ±5 and ±15 frames, contact IoU, per hand and pooled |

### Number 2: does our score pick clips a robot hand can copy?

How we label a clip as "copyable" (fixed before computing). We map the motion
capture hand onto the Allegro hand by fingertip position. The clip is
copyable if, for some shift of the robot hand of up to 2 cm, three things
hold: the robot thumb and one more fingertip are within 1.5 cm of the object
surface, every robot fingertip is within 2 cm of the human fingertip that
touches the object, and no part of the robot hand goes more than 5 mm into
the object. 47% of tuning clips are copyable. We also report the label at
2 mm and 10 mm.

Data: 2,385 tuning clips from 6 people and 1,426 held-out clips from 3 people.

Numbers in the first column match the measurement plan.

| # | Comparison | Why |
| --- | --- | --- |
| 1 | Our score against Muse Spark asked the same question with no checklist | The main comparison |
| 4 | The same, on objects the score has never seen. Every learned part is refit without the object. The object rate cannot score these (AUC 0.5) | The main use. Reported next to comparison 1 |
| 2 | The copyable rate of each object from tuning labels, given to held-out clips | How much the object name alone tells. It needs labels for every object, so it is a reference, not a video method |
| 3 | Muse Spark alone against Muse Spark plus our checks | GraspGrade as a layer on an existing filter |
| 5 | Muse Spark alone, checklist with equal weights, checklist with learned weights | What the checklist adds, and what learning adds |
| 6 | Our score with the object rate added when 0, 5, 20 or all labeled clips of the object exist | Our score as the default, the object rate as an add-on |
| extra | AUC inside each object, and AUC for each reason a clip fails (grip, reach, collision) | What the score can and cannot tell apart |

Metrics: AUC with a 95% interval that resamples the 51 held-out sequences,
AUC per held-out person, and the share of copyable clips in our top 20%
against the base rate.

### Soft object check

| | |
| --- | --- |
| Ground truth | Every HOT3D and OpenGraph object labeled rigid, squeezable or soft by both of us before any call |
| Data | All 1,426 HOT3D held-out clips. All 286 OpenGraph held-out contact segments, plus every tuning segment with a soft object |
| Metrics | Share of soft object segments caught, share of HOT3D clips held back, our score AUC on the clips that are kept |
| Pass | 70% or more of soft object segments caught, and 10% or less of HOT3D held-out clips held back |

HOT3D tracks every object as a rigid body, so any HOT3D clip held back counts
as a mistake. About 12% of these clips show squeeze bottles or paper cartons.
They are labeled squeezable, and squeezable objects keep their score.

All tuning uses the tuning split. Every number we report comes from held-out
data. The one exception is the soft object check. It learns nothing from
labels, and held-out OpenGraph has only about a dozen segments with soft
objects, so it also uses the soft segments of tuning episodes. Held-out
results are shown on their own as well.

## What we know so far (tuning data only)

- SAM 3.1 at 10 fps tracks hands and the grasped object as well as at 30 fps,
  so SAM costs a third.
- Visible contact starts about 0.2 s before glove pressure. Muse Spark marks
  touches about 7 frames early at the start and late at the end. A shift
  learned on tuning doubles its boundary F1 at ±5 frames (0.06 to 0.13).
- The object is the strongest signal. Seven size and shape numbers from the
  3D models predict copyable almost as well as the object name (0.70 against
  0.74) and still work on unseen objects (0.67 against 0.5).
- With real Muse Spark answers: the plain question gives 0.52, our checklist
  0.63, and 0.61 on unseen objects.
- The object is not the whole story. It explains about a quarter of the
  label variance. Every one of the 33 objects is copyable in some grasps and
  not in others, and 24 of them in 20% to 80% of their grasps. Our score
  works at the object level. Ranking two grasps of the same
  object is still open.

## Plan for the remaining days

1. SAM on the remaining tuning episodes, then fix rule B on all 50.
2. Number 1 on held-out: SAM, Muse Spark alone, Muse Spark check.
3. Number 2 on held-out: SAM on the clips, Muse Spark checklist and baseline.
4. Soft object check: object labels by both of us, a fixed prompt, one call
   per clip.
5. Final scoring, viewer, demo video, README, then submit on November 4.

Credit: about $146 of $150 planned, $19 spent so far.

## Limits we know about

- The robot check is geometric, not a physics test, and uses one robot hand,
  the Allegro.
- HOT3D tracks every object as a rigid body. The soft object check is tested
  on the soft objects in OpenGraph, and there are few of them.
- The score works at the object level. It does not rank two grasps of the
  same object.
- The copyable share depends on the collision limit (18% at 2 mm, 47% at
  5 mm, 66% at 10 mm).
- The held-out set for number 2 has three people. Everything is lab data.

## Sources

- [HOT3D dataset page](https://www.projectaria.com/datasets/hot3D/)
- [OpenGraph Ego-Tactile Manipulation](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation)
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- [Do as I Do: Dexterous Manipulation Data from Everyday Human Videos](https://arxiv.org/abs/2606.19333)
- [SPIDER: Scalable Physics-Informed Dexterous Retargeting](https://arxiv.org/abs/2511.09484)
- [ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer](https://arxiv.org/abs/2503.21860)
