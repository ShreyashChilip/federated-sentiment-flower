"""Kaggle script kernel for the FL benchmark (CPU session, internet on).

The JOB block is filled in by tools/kaggle_job.py before each push; nothing
else here is edited. The kernel

1. clones the repository at the pinned commit (the run is official only if
   the tree is clean and on Kaggle; every run records the commit);
2. restores earlier outputs attached as kernel sources (results, built
   feature bundles, partitions) so finished runs are skipped;
3. runs the requested stages with a wall-clock budget, then the requested
   diagnostics and analyses;
4. leaves results, partitions and feature bundles in /kaggle/working, which
   Kaggle saves as the kernel output.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JOB = {
    "repo": "https://github.com/ShreyashChilip/federated-sentiment-flower.git",
    "commit": "REPLACE",
    "stages": [],
    "max_hours": 10.5,
    "diagnostics": [],          # e.g. [["amazon_vg", 42]]
    "analyze": [],              # stages to analyze at the end
    "keep_bundles": True,       # copy built feature bundles into the output for the next session
    "extra_args": [],
    "slug": "job",
}

T0 = time.time()
WORK = Path("/kaggle/working")
CODE = Path("/tmp/fl")
OUT_RESULTS = WORK / "results_benchmark"
OUT_BUNDLES = WORK / "bundles"
OUT_PARTITIONS = WORK / "partitions"


def sh(*cmd, check=True, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=check, **kw)


def restore() -> None:
    """Copy earlier kernel outputs (attached under /kaggle/input) into place."""
    for src in sorted(Path("/kaggle/input").glob("*")):
        for name, dest in (("results_benchmark", OUT_RESULTS), ("partitions", OUT_PARTITIONS)):
            if (src / name).is_dir():
                print(f"restore {src / name} -> {dest}", flush=True)
                shutil.copytree(src / name, dest, dirs_exist_ok=True)
        if (src / "bundles").is_dir():
            for b in (src / "bundles").iterdir():
                target = CODE / "data_cache" / "features" / b.name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(b, target)   # read-only input; bundles are only read after build
                    print(f"link bundle {b.name}", flush=True)


def main() -> None:
    if not CODE.exists():
        sh("git", "clone", "--quiet", JOB["repo"], CODE)
    sh("git", "-C", CODE, "checkout", "--quiet", JOB["commit"])
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", CODE / "requirements-kaggle.txt")
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PARTITIONS.mkdir(parents=True, exist_ok=True)
    restore()
    # results/benchmark and partitions written straight into the saved output
    (CODE / "results").mkdir(exist_ok=True)
    link = CODE / "results" / "benchmark"
    if not link.exists():
        os.symlink(OUT_RESULTS, link)
    for f in OUT_PARTITIONS.glob("*"):
        if not (CODE / "partitions" / f.name).exists():
            shutil.copy2(f, CODE / "partitions" / f.name)
    os.environ["FEDBENCH_SCRATCH"] = "/tmp/fedbench_scratch"
    os.environ.setdefault("FEDBENCH_RAM_GB", "28")
    os.chdir(CODE)
    sh("git", "status", "--short", "--untracked-files=no")
    env_report = subprocess.run([sys.executable, "-c",
                                 "import json; from pathlib import Path; from src.utils.env import collect_environment; "
                                 "print(json.dumps(collect_environment(Path('.'))))"], capture_output=True, text=True)
    (OUT_RESULTS / f"kaggle_environment_{int(T0)}.json").write_text(env_report.stdout)
    print(env_report.stdout, flush=True)

    # Model-free diagnostics first: they need the review store built in this session.
    for regime, seed in JOB["diagnostics"]:
        sh(sys.executable, "experiments/client_diagnostics.py", "--regimes", regime, "--seeds", seed, check=False)
    remaining = JOB["max_hours"] - (time.time() - T0) / 3600
    if JOB["stages"]:
        sh(sys.executable, "experiments/run_benchmark.py", "run", "--stages", *JOB["stages"],
           "--max-hours", f"{max(remaining, 0.1):.2f}", *JOB["extra_args"], check=False)
    for stage in JOB["analyze"]:
        sh(sys.executable, "experiments/analyze_benchmark.py", "--stages", stage, check=False)
    for stage in JOB["stages"]:
        sh(sys.executable, "experiments/run_benchmark.py", "status", "--stages", stage, check=False)

    for f in (CODE / "partitions").glob("*"):
        if f.is_file() and not (OUT_PARTITIONS / f.name).exists():
            shutil.copy2(f, OUT_PARTITIONS / f.name)
    if JOB["keep_bundles"]:
        for b in (CODE / "data_cache" / "features").glob("*"):
            if b.is_dir() and not b.is_symlink() and (b / "meta.json").exists() and not (OUT_BUNDLES / b.name).exists():
                size = sum(f.stat().st_size for f in b.rglob("*") if f.is_file()) / 2**30
                used = sum(f.stat().st_size for f in WORK.rglob("*") if f.is_file()) / 2**30
                if used + size > 17:   # Kaggle keeps at most ~20 GB of output; never risk losing the results
                    print(f"NOT saving bundle {b.name} ({size:.1f} GB): output would exceed 17 GB", flush=True)
                    continue
                shutil.copytree(b, OUT_BUNDLES / b.name)
                print(f"saved bundle {b.name} ({size:.1f} GB)", flush=True)
    # One compact archive to download (results + partitions; bundles stay as kernel output only).
    archive = WORK / f"fedbench_results_{JOB['slug']}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (OUT_RESULTS, OUT_PARTITIONS):
            for f in base.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(WORK))
    print(f"archive {archive} ({archive.stat().st_size / 2**20:.0f} MB)", flush=True)
    print(f"done in {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
