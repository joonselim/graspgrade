"""Render the humanoid shot for the first scene of the demo video. No API call.

The first scene talks about Meta building humanoid robots, and no picture of
that robot is public. This renders a stand-in: the Apptronik Apollo model from
MuJoCo Menagerie (Apache 2.0) in a light studio, posed by hand (no physics):
it looks down, reaches forward, turns its wrist, brings the hand back, and
looks at the camera while the camera moves slowly around it.

Run in WSL from ~/graspgrade, with its own venv so the pinned project venv
stays as it is:

    python3 -m venv .venv-render && .venv-render/bin/pip install mujoco==3.3.0 numpy
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git ext/mujoco_menagerie
    (cd ext/mujoco_menagerie && git sparse-checkout set apptronik_apollo)
    MUJOCO_GL=egl .venv-render/bin/python web/tools/robot_render.py      # out/demo/robot_take.mp4

Writes out/demo/robot_take.mp4 (1152x648, 30 fps, 10 s), used by demo_video.py.
"""
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "ext" / "mujoco_menagerie" / "apptronik_apollo"
OUT = ROOT / "out" / "demo" / "robot_take.mp4"
W, H, FPS, SECONDS = 1152, 648, 30, 10.0

STUDIO = """<mujoco model="studio">
  <include file="apptronik_apollo.xml"/>
  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.32" specular="0.1 0.1 0.1"/>
    <rgba haze="0.93 0.94 0.96 1"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" zfar="60" haze="0.6"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.96 0.97" rgb2="0.82 0.85 0.89" width="512" height="3072"/>
    <material name="studio_floor" rgba="0.9 0.91 0.93 1" emission="0.55" reflectance="0.05" specular="0.1"/>
  </asset>
  <worldbody>
    <light pos="2.5 -2.0 4.0" dir="-0.5 0.4 -1" diffuse="0.75 0.75 0.75" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="-2.0 2.5 3.0" dir="0.4 -0.5 -1" diffuse="0.25 0.25 0.27" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="studio_floor"/>
  </worldbody>
</mujoco>
"""

# poses by time: joint -> angle. Joints left out keep the standing keyframe.
POSES = [
    (0.0, {}),
    (1.2, {"torso_pitch": 0.03}),
    (3.4, {"neck_yaw": -0.35, "neck_pitch": 0.38, "torso_pitch": 0.16, "torso_yaw": -0.12,
           "r_shoulder_fe": -0.95, "r_shoulder_aa": -0.12, "r_elbow_fe": -0.55, "r_wrist_pitch": -0.2}),
    (4.4, {"neck_yaw": -0.35, "neck_pitch": 0.4, "torso_pitch": 0.17, "torso_yaw": -0.12,
           "r_shoulder_fe": -1.0, "r_shoulder_aa": -0.12, "r_elbow_fe": -0.5, "r_wrist_pitch": -0.2}),
    (5.6, {"neck_yaw": -0.35, "neck_pitch": 0.4, "torso_pitch": 0.17, "torso_yaw": -0.12,
           "r_shoulder_fe": -1.0, "r_shoulder_aa": -0.12, "r_elbow_fe": -0.5, "r_wrist_pitch": -0.2,
           "r_wrist_roll": 0.9}),
    (7.4, {"neck_yaw": -0.2, "neck_pitch": 0.25, "torso_pitch": 0.06,
           "r_shoulder_fe": -0.55, "r_shoulder_aa": -0.2, "r_elbow_fe": -1.65, "r_wrist_roll": 0.3}),
    (9.0, {"neck_yaw": 0.3, "neck_pitch": 0.05, "torso_yaw": 0.12,
           "r_shoulder_fe": -0.2, "r_elbow_fe": -0.7}),
    (SECONDS, {"neck_yaw": 0.32, "neck_pitch": 0.04, "torso_yaw": 0.13,
               "r_shoulder_fe": -0.15, "r_elbow_fe": -0.6}),
]


def ease(p):
    return p * p * (3 - 2 * p)


def pose_at(t, names):
    """Each joint eases between the poses that set it, and holds its last value in between."""
    out = {}
    for name in names:
        keys = [(0.0, None)] + [(pt, pv[name]) for pt, pv in POSES if name in pv]
        prev = keys[0]
        for k in keys[1:]:
            if t <= k[0]:
                a = BASE[name] if prev[1] is None else prev[1]
                p = (t - prev[0]) / max(k[0] - prev[0], 1e-6)
                out[name] = a + (k[1] - a) * ease(min(max(p, 0.0), 1.0))
                break
            prev = k
        else:
            out[name] = prev[1] if prev[1] is not None else BASE[name]
    return out


def main():
    (MODEL / "studio.xml").write_text(STUDIO)
    m = mujoco.MjModel.from_xml_path(str(MODEL / "studio.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    key = d.qpos.copy()
    adr = {m.joint(i).name: int(m.jnt_qposadr[i]) for i in range(m.njnt) if m.joint(i).name != "floating_base"}
    global BASE
    BASE = {n: float(key[a]) for n, a in adr.items()}
    names = sorted({n for _, pv in POSES for n in pv})
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # raw frames into ffmpeg, so no extra Python video package is needed
    w = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16",
                          "-preset", "slow", "-pix_fmt", "yuv420p", str(OUT)], stdin=subprocess.PIPE)
    for f in range(int(SECONDS * FPS)):
        t = f / FPS
        d.qpos[:] = key
        for n, v in pose_at(t, names).items():
            d.qpos[adr[n]] = v
        mujoco.mj_forward(m, d)
        p = ease(t / SECONDS)
        cam.lookat[:] = [0.05, 0.0, 0.95 - 0.08 * p]
        cam.distance = 2.9 - 0.7 * p
        cam.azimuth = 165 - 35 * p
        cam.elevation = -6 - 8 * p
        r.update_scene(d, cam)
        w.stdin.write(r.render().tobytes())
    w.stdin.close()
    w.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
