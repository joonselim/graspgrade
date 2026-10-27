# GraspGrade experiment log

Each experiment has its own page with the setup, numbers, cost, images and
what we concluded. Pages are written to be reused in the submission as they are.
Images are in `img/`. Raw outputs stay in `out/` (not committed).

| ID | Date | Topic | Meta API cost | Result in one line |
| --- | --- | --- | --- | --- |
| [EXP-01](exp01_meta_api_smoke_test.md) | 2026-10-27 | Meta Model API smoke test | $0.035 | Muse Spark and SAM 3.1 both work on our clips. SAM costs $0.03 per 5 s clip and finds both gloved hands in every frame. |
| [EXP-02](exp02_sam_object_prompts.md) | 2026-10-27 | SAM 3.1 object prompts, small container clip | $0.15 | Only `small object` works as a generic prompt. 2D hand and object overlap does not show contact. |
| [EXP-03](exp03_sam_object_prompts_spray.md) | 2026-10-27 | SAM 3.1 object prompts, spray bottle clip | $0.15 | See page. |
| [EXP-04](exp04_local_data_checks.md) | 2026-10-27 | Cleaning OpenGraph without any API call | $0 | Labels match tactile at threshold 800, not at 0. Head IMU gives camera motion for free. |

Total Meta API spend so far: about $0.34 of $150.
