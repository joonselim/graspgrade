# GraspGrade experiment log

Each experiment has its own page with the setup, numbers, cost, images and
what we concluded. Pages are written to be reused in the submission as they are.
Images are in `img/`. Raw outputs stay in `out/` (not committed).

What we compare, and on which data, is set in [MEASUREMENT_PLAN.md](../MEASUREMENT_PLAN.md).
Two pages have a narrower role since that plan: EXP-06 tests a logistic
regression that is now an appendix, not one of the compared arms, and
EXP-07 counts clips for pricing only. The clip set for number 2 is EXP-11.

| ID | Date | Topic | Meta API cost | Result in one line |
| --- | --- | --- | --- | --- |
| [EXP-01](exp01_meta_api_smoke_test.md) | 2026-10-27 | Meta Model API smoke test | $0.035 | Muse Spark and SAM 3.1 both work on our clips. SAM costs $0.03 per 5 s clip and finds both gloved hands in every frame. |
| [EXP-02](exp02_sam_object_prompts.md) | 2026-10-27 | SAM 3.1 object prompts, small container clip | $0.15 | Only `small object` works as a generic prompt. 2D hand and object overlap does not show contact. |
| [EXP-03](exp03_sam_object_prompts_spray.md) | 2026-10-27 | SAM 3.1 object prompts, spray bottle clip | $0.15 | `small object` also finds a large spray bottle in every frame. It is our object prompt. |
| [EXP-04](exp04_local_data_checks.md) | 2026-10-27 | Cleaning OpenGraph without any API call | $0 | Labels match tactile at threshold 800, not at 0. Head IMU gives camera motion for free. |
| [EXP-05](exp05_sam_10fps_vs_30fps.md) | 2026-10-27 | SAM 3.1 at 10 fps vs 30 fps | $0.04 | Hands and the grasped object get the same ids and masks (IoU 0.97 to 0.99) at 10 fps. SAM cost drops to a third. |
| [EXP-06](exp06_contact_pilot.md) | 2026-10-27 | Contact cutting pilot, 3 whole episodes | $1.15 | SAM keeps hand ids over whole episodes. First boundary F1 is low (0.18 at ±5 frames, 2 training episodes). Visible contact starts about 0.2 s before glove pressure. (Appendix model, see the measurement plan.) |
| [EXP-07](exp07_hot3d_clip_count.md) | 2026-10-27 | HOT3D grasp clip count and number 2 cost | $0 | 2,152 usable grasp clips (median 4.6 s) from mocap. SAM on all costs $59, so we keep all 713 held-out clips plus 400 tuning clips (about $32). (Pricing only. The clip set is EXP-11.) |
| [EXP-08](exp08_hot3d_calibration_orientation.md) | 2026-10-27 | HOT3D calibration orientation | $0 | Calibration and dataset 2D boxes are in raw Aria orientation. Preview video = raw turned 90 degrees clockwise, raw (u, v) to (1407 - v, u). Frame k = k-th RGB timestamp. |
| [EXP-09](exp09_hot3d_fingertips_umetrack.md) | 2026-10-27 | Fingertips from UmeTrack, no MANO | $0 | Our forward kinematics puts 100% of landmarks inside the dataset hand boxes. Hand 0 = left (profile as is), hand 1 = right (x mirror). During grasps the closest fingertip is 2.3 mm from the object (median). |
| [EXP-10](exp10_copyable_label_skew.md) | 2026-10-27 | Copyable labels (Allegro retargeting) and label skew | $0 | Chosen rule (near 1.5 cm, deep 1 cm, wrist shift up to 2 cm) gives 54% copyable on 1,422 tuning grasps. The share is set mostly by the near threshold. 26% to 94% by object. |
| [EXP-11](exp11_number2_clips_and_labels.md) | 2026-10-27 | Number 2 clips from hand to object distance, held-out labels | $0 | Fixed rule gives 52.6% copyable on held-out grasps. Mocap clips: 2,385 tuning and 1,426 held-out, 52% to 53% copyable. SAM for all held-out clips plus 400 tuning clips is about $27 with overlapping clips joined. |
| [EXP-12](exp12_local_checks_signal.md) | 2026-10-27 | Free checklist items and object-only reference | $0 | The three free checks measure well (camera vs headset 0.93) but do not predict copyable (AUC 0.49 to 0.53). Object identity alone reaches 0.70 (leave one person out). Pooled AUC across people is misleading. |
| [EXP-13](exp13_hand_shape_check.md) | 2026-10-27 | Hand shape during contact as the retargeting check | $0 | Video hand shape predicts copyable at 0.63 (mocap upper bound 0.67), mostly as a stand-in for object size. Object alone 0.69. Both together 0.72. |
| [EXP-14](exp14_sam_rule_pilot.md) | 2026-10-27 | Arm B (SAM rule) and reference rows, pilot | $0 | Code check on 3 tuning episodes: arm B IoU 0.75 vs 0.67 for always-in-contact, but it finds only 52 of 190 to 248 boundaries. The raw glove flickers under 0.1 s, so the cleaned glove is now the main ground truth. |
| [EXP-15](exp15_label_methods.md) | 2026-10-27 | Copyable label methods: robot or retargeting? | $0 | The EXP-10 label mostly measures a retargeting side effect (fingertips 31 mm off the human ones). Faithful retargeting calls 94% to 96% copyable. No method passes the pre-set criteria. The label is on hold. |
| [EXP-16](exp16_copyable_v2.md) | 2026-10-27 | Copyable label v2: position retargeting, mesh collision, contact reach | $0 | Passes the checks set in advance: 47% copyable, hand size correlation +0.04. Failures are mostly collisions (25.5%) and reach (10.9%). The share depends strongly on the depth limit. |

Total Meta API spend so far: about $1.53 of $150.
