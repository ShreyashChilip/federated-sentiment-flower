"""Kaggle entry point. Paste into one notebook cell (GPU on, Internet on) or
run as a script. It installs the pinned environment, records the hardware and
runs the requested stage. No research logic lives here.

Before the first run, set REPO_URL to the project's git remote, or attach the
repository as a Kaggle dataset and set REPO_DIR to its path.

Stages:
  bringup   - install, unit tests, smoke test, write environment report
  baseline  - Phase 1 (requires the tuning phase to have fixed the config)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = ""                       # e.g. https://github.com/<user>/<repo>.git
REPO_DIR = Path("/kaggle/working/fl-sentiment")
STAGE = os.environ.get("FL_STAGE", "bringup")


def sh(*cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


if not REPO_DIR.exists():
    if not REPO_URL:
        raise SystemExit("Set REPO_URL (or attach the repository and set REPO_DIR).")
    sh("git", "clone", REPO_URL, REPO_DIR)
os.chdir(REPO_DIR)

# torch is deliberately not reinstalled: the CUDA build shipped with the Kaggle
# image is used and its version is recorded in the environment report.
sh(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt")

sys.path.insert(0, str(REPO_DIR))
from src.utils.env import collect_environment  # noqa: E402

env = collect_environment(REPO_DIR)
Path("results").mkdir(exist_ok=True)
Path("results/kaggle_environment.json").write_text(json.dumps(env, indent=1))
print(json.dumps(env, indent=1))
if env["platform_kind"] != "kaggle":
    print("WARNING: not running on Kaggle; results will be marked as not official.")

if STAGE == "bringup":
    sh(sys.executable, "-m", "pytest", "tests", "-q")
    sh(sys.executable, "experiments/00_smoke_test.py")
elif STAGE == "baseline":
    sh(sys.executable, "experiments/make_partitions.py", "--config", "yelp.yaml", "--clients", "10", "--alphas", "0.1", "--no-iid")
    sh(sys.executable, "experiments/01_baseline.py", "--config", "yelp.yaml")
else:
    raise SystemExit(f"unknown stage {STAGE}")
