"""Environment check. Run after scripts/setup_wsl.sh, with the venv active.

    python scripts/verify_env.py

Two parts:
  A. every package we depend on imports
  B. the dex-retargeting sample video runs end to end through
     MediaPipe -> Allegro right hand, and prints three numbers

Reference run from an earlier machine: detection_rate 1.0, jitter 2.82 mm,
retarget_err median 15.7 percent. The metric definitions below are ours, so
treat those as a rough sanity range, not an exact target.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEX = ROOT / "ext" / "dex-retargeting"
SAMPLE = DEX / "example" / "vector_retargeting" / "data" / "human_hand_video.mp4"


def part_a():
    print("=== A. imports ===")
    ok = True
    for name in ("cv2", "mediapipe", "numpy", "pandas", "pyarrow", "sklearn",
                 "torch", "openai", "huggingface_hub", "dex_retargeting"):
        try:
            mod = __import__(name)
            print(f"  {name:16s} {getattr(mod, '__version__', 'n/a')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:16s} FAILED: {exc}")
            ok = False
    return ok


def find_hand_detector():
    """The example ships a MediaPipe wrapper that already handles the hand-label
    flip. Reuse it rather than redoing the wrist frame conversion."""
    hits = list(DEX.glob("example/**/single_hand_detector.py"))
    if not hits:
        raise FileNotFoundError(f"single_hand_detector.py not found under {DEX}")
    sys.path.insert(0, str(hits[0].parent))
    from single_hand_detector import SingleHandDetector  # type: ignore
    return SingleHandDetector


def part_b():
    print("\n=== B. retargeting on the sample video ===")
    import cv2
    from dex_retargeting.constants import (
        HandType, RetargetingType, RobotName, get_default_config_path,
    )
    from dex_retargeting.retargeting_config import RetargetingConfig

    if not SAMPLE.exists():
        print(f"  sample video missing: {SAMPLE}")
        print("  did the clone run with --recurse-submodules?")
        return False

    RetargetingConfig.set_default_urdf_dir(str(DEX / "assets" / "robots" / "hands"))
    cfg_path = get_default_config_path(
        RobotName.allegro, RetargetingType.vector, HandType.right
    )
    retargeting = RetargetingConfig.load_from_file(cfg_path).build()

    SingleHandDetector = find_hand_detector()
    # selfie=False: the video is not mirrored, so MediaPipe reports the right
    # hand as "Left". The wrapper inverts the label for us.
    detector = SingleHandDetector(hand_type="Right", selfie=False)

    indices = retargeting.optimizer.target_link_human_indices
    origin_idx, task_idx = indices[0, :], indices[1, :]

    cap = cv2.VideoCapture(str(SAMPLE))
    total = detected = 0
    prev_joints = None
    jitter_mm, rel_err = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num, joint_pos, _, _ = detector.detect(rgb)
        if num == 0 or joint_pos is None:
            prev_joints = None
            continue
        detected += 1

        if prev_joints is not None:
            step = np.linalg.norm(joint_pos - prev_joints, axis=1)  # metres
            jitter_mm.append(float(np.mean(step)) * 1000.0)
        prev_joints = joint_pos

        ref = joint_pos[task_idx, :] - joint_pos[origin_idx, :]
        qpos = retargeting.retarget(ref)
        achieved = robot_task_vectors(retargeting, qpos)
        if achieved is not None:
            scale = getattr(retargeting.optimizer, "scaling", 1.0)
            denom = np.linalg.norm(ref * scale, axis=1)
            good = denom > 1e-6
            rel_err.extend((np.linalg.norm(achieved - ref * scale, axis=1)[good]
                            / denom[good]).tolist())
    cap.release()

    rate = detected / total if total else 0.0
    print(f"  frames             {total}")
    print(f"  detection_rate     {rate:.3f}")
    print(f"  jitter (mm/frame)  {np.median(jitter_mm):.2f}" if jitter_mm else
          "  jitter             n/a")
    print(f"  retarget_err       {np.median(rel_err) * 100:.1f} %" if rel_err else
          "  retarget_err       n/a (forward kinematics unavailable)")
    return rate > 0.5


def robot_task_vectors(retargeting, qpos):
    """Fingertip-minus-origin vectors the robot actually reached, via FK.
    Returns None if this dex-retargeting build does not expose the pieces."""
    opt = retargeting.optimizer
    robot = getattr(opt, "robot", None)
    if robot is None or not hasattr(opt, "task_link_names"):
        return None
    try:
        full = np.zeros(robot.dof)
        full[retargeting.optimizer.idx_pin2target] = qpos[retargeting.optimizer.idx_pin2target] \
            if len(qpos) == robot.dof else qpos
        robot.compute_forward_kinematics(qpos if len(qpos) == robot.dof else full)
        pos = {n: robot.get_link_pose(robot.get_link_index(n))[:3, 3]
               for n in set(opt.task_link_names) | set(opt.origin_link_names)}
        return np.array([pos[t] - pos[o]
                         for o, t in zip(opt.origin_link_names, opt.task_link_names)])
    except Exception as exc:  # noqa: BLE001
        print(f"  (forward kinematics skipped: {exc})")
        return None


if __name__ == "__main__":
    a = part_a()
    b = part_b() if a else False
    print("\nresult:", "ok" if (a and b) else "check the failures above")
    sys.exit(0 if (a and b) else 1)
