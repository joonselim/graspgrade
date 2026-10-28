# EXP-35: Arm C (SAM rule + Muse Spark boundary check), tuning shift sample (P5, part 1)

Date: 2026-10-29. Script: `scripts/muse_contact.py c --segments out/contact/rule_segments.json --sample 100 --send`.
Raw output: `out/number1/muse_c/episode_*/` (one file per boundary),
`out/number1/logs/c_tune_sample_send.log`.
Meta API cost: $3.09 (Muse Spark). Data: tuning only (200 boundaries).

## Question

Arm C shows Muse Spark 7 frames around each arm B boundary and asks which
frame is the real start or end (0 means no change). The arm C timing shift is
learned on tuning, from the boundaries Muse Spark moved. This step sends the
tuning sample for that shift: 100 starts and 100 ends drawn at random (seed
20261026) from the leave-one-episode-out arm B segments of EXP-32.

## Setup

- 200 calls, one per boundary, 7 frames each.
- Replies that never parse (plan section "Replies that never parse"): the
  same command was run once more, then without `--send` to count what is left.

## Results

| Pass | Calls sent | Without a frame number | Cost |
| --- | --- | --- | --- |
| First | 200 | 0 | $3.09 |
| Second (resend) | 0 | 0 | $0 |
| Count only | 0 | 0 | $0 |

- Cost: $3.09 against the $2.61 shown before sending (+18%). A call used
  8,559 input tokens (the estimate assumed 7,700) and 1,114 output tokens on
  average (the estimate assumed 800, median 992, 95th percentile 2,337).
- At the measured cost per call ($0.0154), the 1,790 held-out boundaries
  would cost about $27.63, not $23.31.
- Picks: 105 of 200 boundaries kept ("no change"). The other 95 moved to
  frame 1 (14), 2 (21), 3 (21), 4 (16), 5 (10), 6 (7) or 7 (6). The 7
  frames sit at -15, -10, -5, 0, +5, +10 and +15 frames from the arm B
  boundary, so frame 4 is the arm B frame itself.

![Arm C picks and cost per call on the 200 tuning boundaries](img/exp35_arm_c_tuning_sample.png)

## Conclusion

The tuning sample for the arm C shift is complete, with no fallback. The
held-out part of arm C was not sent, because its cost at the measured price
is above the range approved before the step.
