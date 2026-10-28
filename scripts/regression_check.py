"""Regression check: run the free commands behind numbers we already reported and compare their outputs.

    python scripts/regression_check.py save        # run all, keep the outputs in out/regression/baseline/
    python scripts/regression_check.py compare     # run all again, list every output that is not byte-identical
    python scripts/regression_check.py compare --only n2_check,gate

No API call. Tuning data only, apart from the benchmark scorers, which read
labels only. `save` also reports whether the rerun equals the output files
that were already on disk. Figures that some commands redraw into
docs/experiments/img/ are put back as they were (the Windows copy is the
master of the figures anyway).
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REG = ROOT / "out" / "regression"
IMG = ROOT / "docs" / "experiments" / "img"
EPS = "0,30,53"
# arm B on the three pilot episodes only, in its own files, so the check never overwrites the
# 50-episode arm B (out/contact/rule_*.json) and stays comparable after P2
PILOT_SEGS = ["--segments", "out/contact/rule_pilot_segments.json"]

# name: (command, output files written by it)
COMMANDS = {
    "rule": (["scripts/contact_cut.py", "rule", "--episodes", EPS, "--name", "rule_pilot"],
             ["out/contact/rule_pilot_report.json", "out/contact/rule_pilot_segments.json"]),
    "selftest": (["scripts/muse_contact.py", "selftest", "--episodes", EPS] + PILOT_SEGS, ["out/number1/selftest.json"]),
    "selftest_lag6": (["scripts/muse_contact.py", "selftest", "--episodes", EPS, "--lag", "6"] + PILOT_SEGS,
                      ["out/number1/selftest_lag6.json"]),
    "score_tune": (["scripts/muse_contact.py", "score", "--episodes", EPS, "--shift-segments",
                    "out/contact/rule_pilot_segments.json"] + PILOT_SEGS, ["out/number1/arms_score.json"]),
    "n2_check": (["scripts/number2_score.py", "--check"], ["out/number2/number2_check.json"]),
    "gate_standin": (["scripts/number2_score.py", "--gate", "--standin"], ["out/number2/gate_standin.json"]),
    "gate": (["scripts/number2_score.py", "--gate"], ["out/number2/gate.json"]),
    "tuning_muse": (["scripts/tuning_muse_check.py"], ["out/tuning_muse_check.json"]),
    "prompt_revision": (["scripts/prompt_revision_check.py"], ["out/prompt_revision.json"]),
    "bench1": (["benchmark/score_number1.py", "benchmark/examples/number1_always_in_contact.json", "--split", "tune"], []),
    "bench2": (["benchmark/score_number2.py", "benchmark/examples/number2_object_share_reference.csv", "--split", "tune"],
               []),
}


def run(name):
    """Runs one command with the figures protected. Returns {file name: path of a copy} including stdout."""
    cmd, outs = COMMANDS[name]
    tmp = Path(tempfile.mkdtemp(prefix=f"reg_{name}_"))
    shutil.copytree(IMG, tmp / "img")
    try:
        r = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True)
    finally:
        for f in IMG.iterdir():
            if not (tmp / "img" / f.name).exists():
                f.unlink()
        shutil.copytree(tmp / "img", IMG, dirs_exist_ok=True)
    if r.returncode:
        raise SystemExit(f"{name} failed:\n{r.stderr[-2000:]}")
    (tmp / "stdout.txt").write_text(r.stdout)
    files = {"stdout.txt": tmp / "stdout.txt"}
    for o in outs:
        shutil.copy(ROOT / o, tmp / Path(o).name)
        files[Path(o).name] = tmp / Path(o).name
    return files


def cmd_save(names):
    for name in names:
        before = {Path(o).name: (ROOT / o).read_bytes() for o in COMMANDS[name][1] if (ROOT / o).exists()}
        files = run(name)
        dst = REG / "baseline" / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        same = []
        for f, p in files.items():
            shutil.copy(p, dst / f)
            if f in before:
                same.append(f"{f} {'same as on disk' if before[f] == p.read_bytes() else 'DIFFERS from the file on disk'}")
        print(f"{name}: saved {len(files)} files. " + "; ".join(same), flush=True)


def cmd_compare(names):
    bad = []
    for name in names:
        files = run(name)
        for f, p in files.items():
            ref = REG / "baseline" / name / f
            ok = ref.exists() and filecmp.cmp(ref, p, shallow=False)
            if not ok:
                bad.append(f"{name}/{f}")
        print(f"{name}: {'identical' if not any(b.startswith(name + '/') for b in bad) else 'DIFFERS'}", flush=True)
    print("all identical" if not bad else "differs: " + ", ".join(bad))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "compare"])
    ap.add_argument("--only", help="comma list of command names")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(COMMANDS)
    unknown = set(names) - set(COMMANDS)
    if unknown:
        raise SystemExit(f"unknown: {sorted(unknown)}")
    (cmd_save if args.cmd == "save" else cmd_compare)(names)


if __name__ == "__main__":
    main()
