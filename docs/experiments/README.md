# GraspGrade experiment log

Each experiment has its own page with the setup, numbers, cost, images and
what we concluded. Pages are written to be reused in the submission as they are.
Images are in `img/`. Raw outputs stay in `out/` (not committed).

| ID | Date | Topic | Meta API cost | Result in one line |
| --- | --- | --- | --- | --- |
| [EXP-01](exp01_meta_api_smoke_test.md) | 2026-10-27 | Meta Model API smoke test | $0.035 | Muse Spark and SAM 3.1 both work on our clips. SAM costs $0.03 per 5 s clip and finds both gloved hands in every frame. |
| [EXP-02](exp02_sam_object_prompts.md) | 2026-10-27 | SAM 3.1 object prompts, small container clip | $0.15 | Only `small object` works as a generic prompt. 2D hand and object overlap does not show contact. |
| [EXP-03](exp03_sam_object_prompts_spray.md) | 2026-10-27 | SAM 3.1 object prompts, spray bottle clip | $0.15 | `small object` also finds a large spray bottle in every frame. It is our object prompt. |
| [EXP-04](exp04_local_data_checks.md) | 2026-10-27 | Cleaning OpenGraph without any API call | $0 | Labels match tactile at threshold 800, not at 0. Head IMU gives camera motion for free. |
| [EXP-05](exp05_sam_10fps_vs_30fps.md) | 2026-10-27 | SAM 3.1 at 10 fps vs 30 fps | $0.04 | Hands and the grasped object get the same ids and masks (IoU 0.97 to 0.99) at 10 fps. SAM cost drops to a third. |
| [EXP-06](exp06_contact_pilot.md) | 2026-10-27 | Contact cutting pilot, 3 whole episodes | $1.15 | SAM keeps hand ids over whole episodes. First boundary F1 is low (0.18 at ±5 frames, 2 training episodes). Visible contact starts about 0.2 s before glove pressure. |

Total Meta API spend so far: about $1.53 of $150.
