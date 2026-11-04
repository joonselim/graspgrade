# GraspGrade environment

Everything runs inside WSL2 Ubuntu. The HOT3D toolkit and dex-retargeting are
built for Linux and macOS, and `pin` (pinocchio), which dex-retargeting needs,
has no Windows wheel. Native Windows is not an option.

## This machine, as found on 2026-10-27

| item | state |
| --- | --- |
| OS | Windows 11 Pro 26200 |
| WSL2 | not installed |
| virtualization | enabled (hypervisor present) |
| GPU | RTX 3070, 8 GB, driver 572.83 |
| disk | C: 170 GB free, D: 953 GB, E: 1761 GB |
| git | installed (Windows side) |
| Python | not installed |
| ffmpeg | not installed |

WSL stores its disk under C:, so the 170 GB on C: is the budget that matters.
That covers the 6.3 GB OpenGraph download, HOT3D sequences, clips and caches.

## Setup order

1. In an **administrator** PowerShell on Windows:
   `wsl --install -d Ubuntu-24.04`, then reboot and create the Ubuntu user.
   This needs elevation, so it has to be run by hand.
2. In Ubuntu: `bash /mnt/c/Users/USER/Documents/works/hackathon_meta/scripts/setup_wsl.sh`
   It installs the system packages, copies this scaffold to `~/graspgrade`,
   builds the venv, and clones dex-retargeting.
3. `cd ~/graspgrade && source .venv/bin/activate && python scripts/verify_env.py`
4. Keys: `hf auth login` (huggingface_hub 2.x removed `huggingface-cli`), and `cp .env.example .env` with the Meta
   Model API key from dev.meta.ai.

The project lives at `~/graspgrade`, on the WSL filesystem. Working under
`/mnt/c` makes disk I/O slow enough to matter on a 6 GB dataset.

## Why the versions are pinned

- `mediapipe==0.10.21`. Newer MediaPipe dropped the old `mp.solutions` API that
  the dex-retargeting examples use.
- `numpy==1.26.4`, `pandas<3`, `pyarrow<20`. MediaPipe 0.10.21 does not support
  numpy 2, and dex-retargeting 0.5.x requires numpy 2. So dex_retargeting stays
  on 0.4.x and numpy stays on 1.
- torch, CPU build, installed from the PyTorch CPU index. dex-retargeting fails
  at import without torch. A plain `pip install torch` pulls a multi-GB CUDA
  build on Linux for no benefit. Swap it if local SAM is ever needed. The 3070
  has 8 GB of VRAM and whether SAM 3 fits has not been checked.
- `--recurse-submodules` on the dex-retargeting clone. Without it the robot URDF
  folders, Allegro included, are empty.

## Traps already paid for

- MediaPipe assigns hand labels as if looking at a mirror. Egocentric video is
  not mirrored, so the right hand arrives labelled "Left". An earlier sample run
  with this set the wrong way found a hand in 0 of 300 frames.
- dex-retargeting defaults to `allegro_hand_right.yml`. Handling only the right
  hand throws away half the data.
- Check that `.env`, `data/` and `out/` are not committed before the repo goes
  public.

## Verified on 2026-10-27 (WSL2 Ubuntu 24.04)

Installed: dex_retargeting 0.4.6, mediapipe 0.10.21, numpy 1.26.4, pandas 2.3.3,
pyarrow 19.0.1, torch 2.14.1+cpu, openai 3.39.0, huggingface_hub 2.3.0, pin 2.7.0.
MediaPipe also pulls in opencv-contrib-python next to opencv-python. Both are
4.11.0.86, so cv2 resolves cleanly.

`scripts/verify_env.py` on the dex-retargeting sample (621 frames, Allegro right):

| metric | this run | earlier machine |
| --- | --- | --- |
| detection_rate | 1.000 | 1.0 |
| jitter | 1.23 mm/frame | 2.82 mm |
| retarget_err median | 17.0 % | 15.7 % |

jitter is defined here as the mean per-joint frame-to-frame move of the
MediaPipe 3D joints. The earlier script used its own definition, so the two
jitter numbers are not comparable. retarget_err is the median of
|robot vector - scaled human vector| / |scaled human vector| over the
fingertip vectors, computed with forward kinematics.

WSL renders MediaPipe on llvmpipe (CPU). The DRI3 and EGL warnings at startup
are expected and harmless.

## Running WSL commands from Windows

Use `wsl.exe -d Ubuntu-24.04 -e bash -c '...'`. Without `-e`, wsl.exe hands the
command to the login shell first, which expands `$VAR` before bash sees it, so
variables set inside the command come out empty. From Git Bash also set
`MSYS_NO_PATHCONV=1`, or `/mnt/c/...` paths get rewritten into Windows paths.

## Keys

`.env` lives in `~/graspgrade/.env` (mode 600) with `MODEL_API_KEY` and
`HUGGING_FACE_TOKEN`. The Windows copy in the scaffold folder is the one the
user edits. Re-copy with `tr -d '\r'` so CRLF line endings do not end up in
the values.
