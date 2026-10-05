"""Prepare (and optionally push) a Kaggle script kernel for one benchmark job.

  python tools/kaggle_job.py --slug fedbench-pilot --stages pilot --diagnostics amazon_vg:42 --analyze pilot --push
  python tools/kaggle_job.py --slug fedbench-screen1 --stages tune_base tune_algorithms screening \
         --sources fedbench-pilot --push

The commit is the current HEAD; the job refuses to be prepared from a dirty
tree or from a commit that is not on the remote, because the kernel clones
that commit. Requires the Kaggle CLI and credentials for --push / --status /
--output.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--user", default=None, help="Kaggle username (default: from the CLI config)")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--diagnostics", nargs="*", default=[], help="regime:seed")
    ap.add_argument("--analyze", nargs="*", default=[])
    ap.add_argument("--sources", nargs="*", default=[], help="earlier kernel slugs whose output is restored")
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / ".kaggle_jobs")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are modified: commit first (the kernel clones a commit)")
    commit = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"commit {commit[:10]} is not on the remote: push it first")
    user = args.user or json.loads(subprocess.run(["kaggle", "config", "view", "--json"], capture_output=True,
                                                  text=True).stdout or "{}").get("username")
    if not user:
        user = input_user()
    job = {"repo": git("remote", "get-url", "origin"), "commit": commit, "stages": args.stages,
           "max_hours": args.max_hours, "diagnostics": [d.split(":") for d in args.diagnostics],
           "analyze": args.analyze, "keep_bundles": True, "extra_args": args.extra}
    src = (ROOT / "notebooks" / "kaggle_benchmark.py").read_text(encoding="utf-8")
    src = re.sub(r"JOB = \{.*?\n\}\n", "JOB = " + json.dumps(job, indent=4) + "\n", src, count=1, flags=re.S)
    out = args.out / args.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "kaggle_benchmark.py").write_text(src, encoding="utf-8")
    meta = {"id": f"{user}/{args.slug}", "title": args.slug, "code_file": "kaggle_benchmark.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [s if "/" in s else f"{user}/{s}" for s in args.sources]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    print(f"prepared {out} for commit {commit[:10]}")
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)


def input_user() -> str:
    raise SystemExit("Kaggle username unknown: pass --user or configure the Kaggle CLI")


if __name__ == "__main__":
    sys.exit(main())
