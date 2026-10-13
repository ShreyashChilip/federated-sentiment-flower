"""Kaggle script kernel for the EXPLORATORY DistilBERT study (GPU session, internet on).

Only the JOB block is edited per session. The kernel
1. clones the repository at the pinned commit;
2. links the official feature bundles (labels, roles, client codes: used to VERIFY the reconstructed data) and
   restores the previous session's output (token cache + results + checkpoints) from the attached inputs;
3. runs experiments/distilbert_fl.py (prepare, pilot, or tune/select/evaluate on every visible GPU) with a
   deadline, so unfinished runs are checkpointed before Kaggle's session limit;
4. leaves bert_cache/ and bert_results/ in /kaggle/working (attach this output to the next session) and a
   compact archive without checkpoints to download.

Sessions
  1. JOB["phase"] = "pilot"  (prepare + pilot); inputs: the two official bundles
  2+ JOB["phase"] = "all";    inputs: the previous session's output only (it carries bundles, cache and results)
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
    "config": "exploratory/extend_v1/bert.yaml",
    "phase": "pilot",            # "pilot" (prepare + pilot) or "all" (tune -> select -> evaluate, resumable)
    "max_hours": 11.0,           # wall-clock budget of this session (Kaggle GPU sessions end at 12 h)
    "margin_hours": 0.75,        # stop starting new rounds this long before the budget ends
    "require_bundles": ["yelp_polarity_7355dff6b72f", "amazon2023_86405fdce84c"],
    "slug": "fedbert",
}

T0 = time.time()
WORK = Path("/kaggle/working")
CODE = Path("/tmp/fl")
OUT_CACHE, OUT_RESULTS, OUT_BUNDLES = WORK / "bert_cache", WORK / "bert_results", WORK / "bundles"
BUNDLE_FILES = ("meta.json", "y_train.npy", "y_test.npy", "roles.npy", "client_codes.npy", "client_keys.json")


def sh(*cmd, check=True, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=check, **kw)


def find_dirs(name: str, root: Path = Path("/kaggle/input"), max_depth: int = 7) -> list[Path]:
    found = []
    if not root.exists():
        return found
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        if Path(dirpath).name == name:
            found.append(Path(dirpath))
            dirnames[:] = []
        elif len(Path(dirpath).relative_to(root).parts) >= max_depth:
            dirnames[:] = []
    return sorted(found)


def one(name: str) -> Path | None:
    hits = find_dirs(name)
    if len(hits) > 1:
        raise SystemExit(f"more than one input contains '{name}': {hits}. Attach only the latest session's output.")
    return hits[0] if hits else None


def restore() -> None:
    features = CODE / "data_cache" / "features"
    features.mkdir(parents=True, exist_ok=True)
    for name in JOB["require_bundles"]:
        hits = find_dirs(name)
        if not hits:
            raise SystemExit(f"PREFLIGHT FAILED: official bundle {name} not found in the inputs; nothing was run")
        src = hits[0]
        if any((h / "meta.json").read_bytes() != (src / "meta.json").read_bytes() for h in hits[1:]):
            raise SystemExit(f"bundle {name} differs between inputs")
        # Only labels/roles/codes are needed (no TF-IDF matrices): copy them so the next session needs one input.
        (OUT_BUNDLES / name).mkdir(parents=True, exist_ok=True)
        for f in BUNDLE_FILES:
            if (src / f).exists() and not (OUT_BUNDLES / name / f).exists():
                shutil.copy2(src / f, OUT_BUNDLES / name / f)
        if not (features / name).exists():
            os.symlink(OUT_BUNDLES / name, features / name)
        print(f"bundle {name}: {src}", flush=True)
    for name, dest in (("bert_cache", OUT_CACHE), ("bert_results", OUT_RESULTS)):
        src = one(name)
        if src is not None:
            print(f"restore {src} -> {dest}", flush=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
    done = len(list(OUT_RESULTS.rglob("COMPLETE"))) if OUT_RESULTS.exists() else 0
    print(f"restored {done} complete run(s)", flush=True)
    if JOB["phase"] == "all" and not any(OUT_CACHE.glob("*_train.npz")):
        raise SystemExit("PREFLIGHT FAILED: phase 'all' expects the token cache from the pilot session; attach that output")


def ensure_packages() -> None:
    """Use the image's torch/transformers/datasets (mutually compatible); install only what is missing.
    The list covers every third-party import of experiments/distilbert_fl.py and the data loaders it calls
    (src.strategies.server imports flwr; pinned to the benchmark's version from requirements-kaggle.txt)."""
    missing = []
    for mod, pkg in (("transformers", "transformers"), ("datasets", "datasets"), ("huggingface_hub", "huggingface_hub"),
                     ("yaml", "PyYAML"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("pandas", "pandas"),
                     ("psutil", "psutil"), ("flwr", "flwr==1.39.0")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        sh(sys.executable, "-m", "pip", "install", "-q", *missing)
    # fail fast (before any data work) if the experiment module still cannot be imported
    sh(sys.executable, "-c", "import sys; sys.path[:0] = ['.', 'experiments']; import distilbert_fl; "
       "import transformers, flwr, torch; print('imports OK: torch', torch.__version__, '| transformers', "
       "transformers.__version__, '| flwr', flwr.__version__)", cwd=CODE)


def main() -> None:
    if not CODE.exists():
        sh("git", "clone", "--quiet", JOB["repo"], CODE)
    sh("git", "-C", CODE, "checkout", "--quiet", JOB["commit"])
    ensure_packages()
    for d in (OUT_CACHE, OUT_RESULTS, OUT_BUNDLES):
        d.mkdir(parents=True, exist_ok=True)
    restore()
    os.chdir(CODE)
    sh("git", "status", "--short", "--untracked-files=no")
    sh("nvidia-smi", check=False)
    import torch
    gpus = max(1, torch.cuda.device_count())
    print(f"GPUs: {gpus} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})", flush=True)
    os.environ["FEDBERT_DEADLINE"] = str(T0 + 3600 * (JOB["max_hours"] - JOB["margin_hours"]))
    base = [sys.executable, "experiments/distilbert_fl.py", "--config", JOB["config"], "--out", OUT_RESULTS, "--cache", OUT_CACHE]
    prep = sh(*base, "--phase", "prepare", check=False)   # verifies data identity; a mismatch aborts here
    if prep.returncode != 0:
        print("PREPARE FAILED (data verification or loading); no training was run", flush=True)
    elif JOB["phase"] == "pilot":
        sh(*base, "--phase", "pilot", check=False)
    else:
        sh(*base, "--phase", "all", "--gpus", gpus, check=False)

    complete = sorted(str(p.parent.relative_to(OUT_RESULTS)) for p in OUT_RESULTS.rglob("COMPLETE"))
    in_progress = sorted(str(p.parent.relative_to(OUT_RESULTS)) for p in OUT_RESULTS.rglob("checkpoint.pt"))
    status = {"commit": JOB["commit"], "phase": JOB["phase"], "hours": (time.time() - T0) / 3600, "gpus": gpus,
              "complete": complete, "in_progress": in_progress}
    (WORK / f"bert_status_{int(T0)}.json").write_text(json.dumps(status, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in status.items()}, indent=1), flush=True)
    archive = WORK / f"fedbert_results_{JOB['slug']}.zip"     # results only: no checkpoints, no caches
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT_RESULTS.rglob("*"):
            if f.is_file() and f.name != "checkpoint.pt":
                z.write(f, f.relative_to(WORK))
        for f in WORK.glob("bert_status_*.json"):
            z.write(f, f.relative_to(WORK))
    used = sum(f.stat().st_size for f in WORK.rglob("*") if f.is_file()) / 2**30
    print(f"archive {archive.name} ({archive.stat().st_size / 2**20:.0f} MB); output {used:.1f} GB; "
          f"done in {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
