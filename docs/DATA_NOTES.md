# Data notes

Facts checked on 2026-10-27 against the downloaded files. Where a fact differs
from the earlier plan, the plan is wrong and this file is right.

## OpenGraph Ego-Tactile

Location: `data/ego-tactile` (5.9 GB). Per-episode cuts: `data/episodes_720p` (511 MB).

| item | value |
| --- | --- |
| episodes | 72, 138,629 frames, 1.28 h |
| fps | 30. Timestamps step exactly 1/30 s. ±5 frames = ±0.167 s |
| episode length | 1036 to 4369 frames, median 2064 |
| video | 1920x1080 **AV1**, 72 episodes packed into 23 mp4 files |
| camera | iPhone ultrawide, about 107 deg horizontal FOV, not fisheye. Intrinsics per episode in `meta/opengraph/camera_intrinsics/` (estimated from FOV, not calibrated) |
| gloves | beige, both hands, 80 taxels each |
| tactile | `observation.tactile.left/right`, 80 floats each, aligned to every frame. 90 Hz raw in `extra/raw_tactile/` |

### Video decoding

The OpenCV pip wheel cannot decode AV1. System ffmpeg can (libdav1d).
`scripts/cut_episodes.py` re-encodes each episode to 720p H.264. All 72 frame
counts match the metadata. Frame alignment was checked on episode 1, which
starts mid-file: cut frame 0 matches source frame 4369 (MSE 3.5, neighbours 37+).
So frame i of `episode_XXXXXX.mp4` is `frame_index` i in the parquet.

### Labels

Per the dataset README, the action boundaries come from contact and grip force,
not from human annotators. Columns: `annotation.l0_action` (214 unique),
`l1_subaction` (751), `anchor`, `grasp_type`, `hand`, `target_object` (89).

`annotation.anchor` has 6 values. Frame counts over the whole set:

| anchor | frames | meaning as seen in episode 0 |
| --- | --- | --- |
| hand_switch | 52,078 | the other hand takes over or joins |
| transition | 27,129 | no contact, hand moving between objects (`hand` = none) |
| load | 22,628 | grip force building |
| release | 20,603 | letting go |
| contact_on | 12,822 | first touch |
| continuation | 3,369 | ongoing hold |

So "in contact" = anchor is not `transition`. Contact onset = the step from
`transition` to `contact_on`. `hand` is none on 27,098 frames vs 27,129
`transition` frames, so the two agree on all but 31 frames.

Two ground truths are available for number 1: these anchor labels, or our own
threshold on the raw tactile columns. The labels are already derived from the
same tactile signal.

Raw tactile as ground truth needs a threshold. With "sum of 80 taxels > 0",
labels and tactile agree on 97% of frames in episodes 0 to 62 but only 72% to
74% in episodes 63 to 71, where the gloves read small pressure at rest. With
threshold 800 they agree on 98.7% (left) and 97.2% (right) of all frames. See
`docs/experiments/exp04_local_data_checks.md`.

Head IMU (`observation.imu.head`, last three values are the gyro): the camera
is almost still in episodes 0 to 47 and 63 to 71 and moves in the spray bottle
episodes 48 to 62.

`meta/opengraph/tactile_highlights/` has per episode: task sentence, object
list, contact percent per hand, and a force time series.

### MediaPipe on gloved hands

`scripts/mediapipe_check.py`, 8 episodes picked across task types, 19,363 frames.
Results in `out/mediapipe_check/`.

| ep | object | any hand | right found when right touching | left found when left touching |
| --- | --- | --- | --- | --- |
| 0 | clear plastic container | 98.3 | 88.4 | 25.0 |
| 1 | clear plastic container | 96.1 | 97.8 | 36.7 |
| 6 | silver metal object | 96.0 | 97.0 | 18.1 |
| 18 | yellow sponge | 92.1 | 51.0 | 100.0 |
| 30 | orange cardboard box | 61.0 | 34.7 | 21.4 |
| 42 | whiteboard eraser | 93.5 | 16.1 | 75.9 |
| 48 | blue spray bottle | 14.3 | 1.5 | 0.0 |
| 66 | red ketchup bottle | 100.0 | 85.4 | 89.5 |
| all | | 82.3 | 63.9 | 36.4 |

- The mirror flip is right. When both hands are found, the hand called left is
  left of the one called right in 100% of 7,101 frames. Wrist x medians: 0.37
  (left) vs 0.69 (right).
- Low left-hand numbers are misses, not label swaps. When only the left hand is
  touching and MediaPipe reports only a "right" hand, that hand is on the right
  half of the image 96% of the time. It is the idle right hand.
- Background decides it. Episode 48: beige gloves against a beige wall and
  blinds, hands large and cut by the frame edge, 14% detection. Episode 66:
  gloves over a dark mat, 100%. Spray bottle episodes are about 20% of all
  frames (blue spray bottle is the top target object at 28,226 frames).
- Impact: the retargeting check (MediaPipe + dex-retargeting) will score low on
  many OpenGraph clips for reasons that are about the glove, not the motion.
  Number 2 runs on HOT3D (bare hands), so it is not affected. Number 1 uses
  SAM 3.1, not MediaPipe.

## HOT3D (Aria)

URL JSONs: `data/hot3d/*.json`, issued 2026-10-27, valid 14 days (to about 11-10,
after the 11-04 deadline). Downloader: `scripts/hot3d_download.py`.

| item | value |
| --- | --- |
| Aria sequences | 198, 14 persons |
| with hand + object pose GT | **136 sequences, 9 persons**, 480,375 RGB frames (434,452 pass QA) |
| without GT | 62 sequences, persons P0004 P0005 P0008 P0016 P0020. Hand and object pose files are empty |
| objects | 33 in the GT sequences, 3D models in `data/hot3d/assets/*.glb` |
| full Aria size | 525 GB, of which main_vrs 329 GB |
| what we fetch | video_main_rgb + ground_truth + hand_data, about 20 GB for all 198 |

The plan assumed 19 persons and a 4 to 5 person validation set. 19 is Aria and
Quest combined. Quest video is greyscale. With Aria only, 9 persons have GT.

### One sequence checked (P0001_10a27bf7)

- `preview_rgb.mp4` is the full RGB stream: 1408x1408 H.264, 30 fps, 3725 frames,
  124 s. So the VRS file is not needed for video.
- The image is fisheye and already upright.
- Camera model in `ground_truth/camera_models.json`: label `camera-rgb`,
  `FISHEYE624`, 1408x1408, with `T_Device_Camera`.
- Checked in EXP-08: the calibration and the dataset 2D boxes
  (`box2d_objects.csv`, `box2d_hands.csv`) are in the raw Aria orientation.
  The preview is raw turned 90 degrees clockwise: raw pixel (u, v) maps to
  preview (1407 - v, u). Preview frame k is the k-th RGB timestamp of stream
  214-1. The video has one frame fewer, the missing one is at the end.
- `box2d_hands.csv` has ground-truth 2D hand boxes per stream. Stream `214-1`
  is RGB. Useful as a free check on SAM hand boxes.
- `masks/` has per-frame hand visible, object visible, QA pass.
- `mano_hand_pose_trajectory.jsonl`: per timestamp, hands "0" and "1", each with
  15 PCA pose values, wrist transform, 10 betas. 7486 rows in 124 s because RGB
  and SLAM timestamps are interleaved.
- Hands are bare with small mocap markers.
- UmeTrack is enough for fingertips, no MANO (EXP-09, `scripts/hot3d_hands.py`).
  `umetrack_hand_user_profile.json` is a kinematic model (22 joints, 21
  landmarks, landmarks 0-4 are fingertips). Hand 0 = left, uses the profile
  as is. Hand 1 = right, profile mirrored along x. Landmarks are in the same
  world frame as the objects. During grasps the closest fingertip is about
  2 mm from the object surface.

## Splits (fixed, `splits/`)

Made by `scripts/make_splits.py`, seed 20261026. Do not re-roll.

- OpenGraph: 50 tuning, 22 held-out episodes. Held-out: 7 14 18 24 27 34 36 37
  39 40 41 42 43 45 48 49 52 57 58 64 66 67.
- HOT3D Aria: tuning persons P0002 P0003 P0009 P0011 P0014 P0015 (85 seqs),
  held-out P0001 P0010 P0012 (51 seqs, 37.5%). Persons have different sequence
  counts, which is why the held-out share is above a third.
