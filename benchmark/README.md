# GraspGrade benchmark: score a video filter against sensors

Teams that train robots on human video filter the video with their own rules,
and they rarely publish a score that others can check. This folder holds two
ground truths built from sensors, not from people looking at the video, and
two small scripts that score any filter against them.

| | Question | Ground truth | Data it comes from | License |
| --- | --- | --- | --- | --- |
| Number 1 | From video alone, when does a touch start and end? | Contact segments from tactile gloves | OpenGraph Ego-Tactile | CC BY 4.0 |
| Number 2 | Which grasp clips can a robot hand copy? | Copyable label from motion capture and a robot hand model | HOT3D (Aria) | CC BY-NC-SA 4.0 |

No video, sensor or motion capture values are in this folder. Get the
original datasets from their owners and use the frame numbers here.

## Number 1: contact timing

File: `number1/ground_truth.json`. 72 episodes (50 tuning, 22 held-out), 30 fps.

- Source: [OpenGraph Ego-Tactile Manipulation](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation)
  by OpenGraph Labs, CC BY 4.0. A head camera and a tactile glove with 80
  sensors on each hand.
- Per episode: `split`, `n_frames`, and per hand two lists of contact
  segments `[start, end)` in the dataset `frame_index`:
  - `glove_raw`: the sum of the 80 sensors is over 800,
  - `glove_clean` (main): `glove_raw` with gaps of 3 frames or less joined
    and pieces shorter than 3 frames dropped. The raw signal flickers at the
    threshold, which would make boundaries that no camera could see.
- Score your segments:

      python score_number1.py my_segments.json                # held-out, cleaned glove
      python score_number1.py my_segments.json --split tune --truth glove_raw

  `my_segments.json` has the same shape as `examples/number1_always_in_contact.json`:
  `{"<episode>": {"left": [[start, end], ...], "right": [...]}}`.
- Metrics: boundary F1 at ±5 frames (0.17 s) and ±15 frames (0.5 s), with
  starts and ends matched one to one, each only to its own kind. Contact IoU
  over all frames. Median start and end error. Pooled and per hand.
- Visible contact usually starts before glove pressure (about 0.2 s in our
  data). A method may learn such a shift on the tuning episodes.

## Number 2: can a robot hand copy this grasp?

File: `number2/clips.csv`. 3,811 grasp clips (2,385 tuning from 6 persons,
1,426 held-out from 3 persons), 47% copyable.

- Source: [HOT3D](https://www.projectaria.com/datasets/hot3D/) by Meta, Aria
  recordings. The label uses the HOT3D hand poses (CC BY-NC-SA 4.0), object
  poses (CC BY-SA 4.0) and object models (CC BY-SA, may not be sold), so it is
  shared under CC BY-NC-SA 4.0.
- A clip is one hand grasping one object: `sequence`, `hand`, `object`,
  `clip_start` and `clip_frames` (preview video frames at 30 fps, frame k is
  the k-th RGB frame of stream 214-1), and `contact_start`, `contact_frames`
  for the touch inside it. `two_hands` is true when the other hand also
  touches the object.
- `copyable` (1 or 0): the human hand is moved onto an Allegro robot hand
  (four fingers, about 1.3 times a human hand) by fingertip position
  retargeting. The grasp is copyable if, for some shift of the robot hand of
  up to 2 cm, all three hold:
  - grip: the robot thumb tip and one more fingertip are within 1.5 cm of the
    object surface,
  - reach: for every human finger that touches the object, the matching
    robot fingertip is within 2 cm of the human one,
  - no collision: no point of the robot hand mesh is more than 5 mm inside
    the object.
- `failed_condition`: `none` for copyable clips, otherwise the first that
  fails in the order grip, reach, collision.
- `copyable_depth_2mm`, `copyable_depth_10mm`: the same label with a 2 mm or
  10 mm collision limit. The share depends strongly on this limit (18% at
  2 mm, 47% at 5 mm, 66% at 10 mm on tuning clips), so report all three if
  you can.
- `role`: how our project used the tuning clips (`score_train` for score
  weights, `item_train` for other models). You can ignore it.
- Score your ranking:

      python score_number2.py my_scores.csv                   # held-out clips
      python score_number2.py my_scores.csv --label copyable_depth_2mm

  `my_scores.csv` has the columns `clip_id,score`, as in
  `examples/number2_object_share_reference.csv`.
- Metrics: AUC with a 95% interval that resamples whole sequences, AUC per
  person and their mean, share of copyable clips in the top 20% against the
  base rate, and AUC on one-hand clips.
- Reference to beat: `examples/number2_object_share_reference.csv` gives every
  clip the copyable share of its object on the tuning clips. The same 33
  objects appear in tuning and held-out, so this simple lookup is strong. A
  video method should also be checked on objects it has not seen.

## Requirements

Python 3 with numpy (number 1), and numpy, pandas and scikit-learn (number 2).

## How the files were made

`scripts/benchmark_export.py` in the GraspGrade repository rebuilds both files
from the original downloads. The checks behind each choice are in
`docs/experiments/` (number 1: EXP-06, EXP-14. Number 2: EXP-15, EXP-16,
EXP-20). The scorers were checked against the project code (EXP-28).

## Credit

- OpenGraph Labs, Ego-Tactile Manipulation dataset, CC BY 4.0.
- Meta, HOT3D dataset. Sequence data and object models CC BY-SA 4.0 (object
  models may not be sold), hand data CC BY-NC-SA 4.0.
- Allegro hand model and retargeting: dex-retargeting.
