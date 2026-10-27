#!/usr/bin/env bash
# GraspGrade environment setup. Run inside WSL2 Ubuntu, not on Windows.
#   bash scripts/setup_wsl.sh
# Safe to re-run. It skips work that is already done.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/graspgrade}"
WIN_SRC="/mnt/c/Users/USER/Documents/works/hackathon_meta"

step() { printf '\n=== %s ===\n' "$1"; }

step "1. system packages"
if ! dpkg -s python3.12-venv >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git ffmpeg build-essential unzip
else
    echo "already installed"
fi

step "2. project directory"
# The project must live on the WSL filesystem. /mnt/c is far too slow for this.
mkdir -p "$PROJECT_DIR"
for f in .gitignore requirements.txt .env.example; do
    [ -f "$PROJECT_DIR/$f" ] || cp "$WIN_SRC/$f" "$PROJECT_DIR/$f"
done
mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/data" "$PROJECT_DIR/out" "$PROJECT_DIR/ext"
cp "$WIN_SRC/scripts/"*.sh "$WIN_SRC/scripts/"*.py "$PROJECT_DIR/scripts/" 2>/dev/null || true
cd "$PROJECT_DIR"
[ -d .git ] || git init -q

step "3. virtualenv"
[ -d .venv ] || python3.12 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

step "4. torch (CPU build)"
# Plain `pip install torch` pulls several GB of CUDA on Linux. dex-retargeting
# only needs torch to import. Switch to a CUDA build if local SAM becomes necessary.
python -c "import torch" 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

step "5. python packages"
pip install -r requirements.txt

step "6. dex-retargeting source checkout"
# Needed for the robot URDF folders and the sample video. Without
# --recurse-submodules the URDF directories, Allegro included, come out empty.
if [ ! -d ext/dex-retargeting/.git ]; then
    git clone --recurse-submodules https://github.com/dexsuite/dex-retargeting ext/dex-retargeting
else
    git -C ext/dex-retargeting submodule update --init --recursive
fi

step "done"
cat <<'MSG'
Next:
  cd ~/graspgrade && source .venv/bin/activate
  python scripts/verify_env.py        # import check + retargeting sample
Then, once the keys are in hand:
  hf auth login                       # huggingface_hub 2.x has no huggingface-cli
  cp .env.example .env && edit MODEL_API_KEY
MSG
