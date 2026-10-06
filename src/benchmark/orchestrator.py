"""Staged, resumable execution of the benchmark.

* Every run executes in its own Python process (``worker``), so memory, Ray
  state or torch state can never accumulate across configurations, and an OOM
  kill takes down one run, not the session.
* A run counts only if its worker finished, its artifacts validated and the
  ``COMPLETE`` marker was written. Anything else is ``failed`` (with the
  return code, a classification such as ``killed_oom_suspected`` and the log
  tail) or ``incomplete``. Failed runs are never interpolated or replaced;
  they are re-attempted only with ``--retry-failed``.
* Finished runs are skipped on resume; an unfinished run directory is set
  aside by ``claim_run_dir`` (renamed, never overwritten).
* Logs live outside the run directory (``<stage>/_logs``) so they survive a
  failed attempt.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.benchmark import plan as P
from src.benchmark import resources
from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.env import collect_environment

REQUIRED_FILES = ("run_metadata.json", "history.json", "final.json", "clients.json")


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def safe_key(key: str) -> str:
    return key.replace("/", "__")


# ---- status ---------------------------------------------------------------------
def is_complete(spec: dict) -> bool:
    return (Path(spec["run_dir"]) / "COMPLETE").exists()


def status_path(spec: dict) -> Path:
    return P.stage_dir_from_spec(spec) / "_status" / f"{safe_key(spec['key'])}.json"


def read_status(spec: dict) -> dict:
    path = status_path(spec)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"attempts": []}


def write_status(spec: dict, status: dict) -> None:
    path = status_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=1), encoding="utf-8")


def run_state(spec: dict) -> str:
    if is_complete(spec):
        return "complete"
    attempts = read_status(spec)["attempts"]
    if attempts and attempts[-1]["outcome"] != "complete":
        return "failed" if attempts[-1]["outcome"] != "running" else "incomplete"
    return "pending"


# ---- validation -----------------------------------------------------------------
def validate_run(spec: dict) -> list[str]:
    """Problems that disqualify a run; empty list = valid."""
    run_dir = Path(spec["run_dir"])
    problems = [f"missing {f}" for f in REQUIRED_FILES if not (run_dir / f).exists()]
    if spec["kind"] != "local_only" and not (run_dir / "final_weights.npz").exists():
        problems.append("missing final_weights.npz")
    if problems:
        return problems
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    if meta["seed"] != spec["seed"]:
        problems.append("seed in metadata differs from the spec")
    if meta["config_hash"] != config_hash(spec["cfg"]):
        problems.append("config hash in metadata differs from the spec (configuration drift)")
    if not meta["partition"].get("sha256"):
        problems.append("partition hash not recorded")
    if spec["kind"] == "federated":
        rounds = int(spec["cfg"]["fl"]["rounds"])
        if not final.get("diverged") and (final.get("rounds_completed") != rounds or len(history) != rounds):
            problems.append(f"expected {rounds} rounds, history has {len(history)}")
        if not final.get("diverged") and "macro_f1" not in final.get("val", {}):
            problems.append("final validation macro-F1 missing")
    if spec["cfg"]["eval"].get("test", True) is False:
        if any(k in final for k in ("unseen", "test", "client_test")):
            problems.append("a tuning run contains held-out metrics")
    elif spec["kind"] != "local_only" and not final.get("diverged"):
        held = final.get("held_out_group")
        if held not in final or not final[held].get("n"):
            problems.append(f"held-out metrics ({held}) missing")
    return problems


# ---- worker (runs inside the child process) ---------------------------------------
def worker(spec_path: Path) -> int:
    from src.benchmark.engine import RUNNERS
    from src.utils.runtime import prepare_run

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    run_dir = Path(spec["run_dir"])
    start = utc()
    t0 = time.perf_counter()
    print(f"[worker] {spec['key']} start {start} pid {os.getpid()}", flush=True)
    prepare_run(spec["cfg"], spec["seed"], run_dir, create_partition=True)
    RUNNERS[spec["kind"]](run_dir)
    problems = validate_run(spec)
    record = {"key": spec["key"], "start_utc": start, "end_utc": utc(), "wall_s": time.perf_counter() - t0,
              "validation_problems": problems, "pid": os.getpid()}
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if problems:
        print("[worker] VALIDATION FAILED: " + "; ".join(problems), flush=True)
        return 3
    (run_dir / "COMPLETE").write_text(json.dumps({"validated_utc": record["end_utc"]}), encoding="utf-8")
    print(f"[worker] {spec['key']} complete in {record['wall_s']:.0f}s", flush=True)
    return 0


def classify_failure(returncode: int, log_tail: str) -> str:
    if returncode in (-9, 137) or "Killed" in log_tail:
        return "killed_oom_suspected"
    if "MemoryError" in log_tail or "Unable to allocate" in log_tail:
        return "oom"
    if returncode == 3:
        return "validation_failed"
    if returncode == -15:
        return "terminated"
    if returncode == 124:
        return "timeout"
    return "error"


def launch(spec: dict, timeout_s: float | None = None) -> dict:
    stage_dir = P.stage_dir_from_spec(spec)
    for sub in ("_specs", "_logs"):
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)
    spec_path = stage_dir / "_specs" / f"{safe_key(spec['key'])}.json"
    spec_path.write_text(json.dumps(spec, indent=1, default=str), encoding="utf-8")
    status = read_status(spec)
    attempt = {"attempt": len(status["attempts"]) + 1, "start_utc": utc(), "outcome": "running",
               "log": str(stage_dir / "_logs" / f"{safe_key(spec['key'])}.attempt{len(status['attempts']) + 1}.log")}
    status["attempts"].append(attempt)
    write_status(spec, status)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(attempt["log"], "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments" / "run_benchmark.py"), "worker",
                                   "--spec", str(spec_path)], cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
    tail = Path(attempt["log"]).read_text(encoding="utf-8", errors="replace")[-4000:]
    attempt.update(end_utc=utc(), wall_s=time.perf_counter() - t0, returncode=code)
    if code == 0 and is_complete(spec):
        attempt["outcome"] = "complete"
    else:
        attempt["outcome"] = classify_failure(code, tail)
        attempt["log_tail"] = tail[-1500:]
    write_status(spec, status)
    return attempt


# ---- stages ---------------------------------------------------------------------
def official_problems(environment: dict) -> list[str]:
    problems = []
    if environment["platform_kind"] != "kaggle":
        problems.append("not running on Kaggle")
    if not environment["git"].get("commit"):
        problems.append("no git commit")
    if environment["git"].get("dirty"):
        problems.append("tracked files differ from the commit")
    return problems


def run_stage(bench: dict, stage: str, *, only: str | None = None, retry_failed: bool = False, jobs: int = 1,
              deadline: float | None = None, force: bool = False, allow_unofficial: bool = False,
              dry_run: bool = False, timeout_s: float | None = None) -> dict:
    kind = bench["stages"][stage]["kind"]
    environment = collect_environment(PROJECT_ROOT)
    problems = official_problems(environment)
    if problems and kind != "pilot" and not allow_unofficial:
        raise SystemExit(f"Refusing stage {stage!r} (results would not be official):\n  - " + "\n  - ".join(problems)
                         + "\nUse --allow-unofficial only for local software checks.")
    specs = P.plan_stage(bench, stage)
    if only:
        specs = [s for s in specs if only in s["key"]]
    print(f"stage {stage}: {len(specs)} run(s)", flush=True)
    summary = {"launched": 0, "complete": 0, "failed": 0, "skipped_failed": 0, "refused": 0, "deferred": 0}
    for spec in specs:
        state = run_state(spec)
        if state == "complete":
            summary["complete"] += 1
            continue
        if state == "failed" and not retry_failed:
            print(f"  FAILED earlier (not retried): {spec['key']}", flush=True)
            summary["skipped_failed"] += 1
            continue
        if deadline is not None and time.time() > deadline:
            summary["deferred"] += 1
            continue
        ok, report = resources.check(spec, jobs=jobs, **_client_facts(spec))
        est = report["estimate"]
        msg = f"est RSS {est['rss_gb']:.1f} GB, scratch {est['scratch_disk_gb']:.1f} GB" if est["known"] else est["reason"]
        for warning in report.get("warnings", []):
            print(f"  WARNING {spec['key']}: {warning}", flush=True)
        if not ok and not force:
            print(f"  REFUSED {spec['key']}: " + "; ".join(report["problems"]), flush=True)
            summary["refused"] += 1
            continue
        print(f"  run {spec['key']} ({msg})", flush=True)
        if dry_run:
            continue
        attempt = launch(spec, timeout_s)
        summary["launched"] += 1
        if attempt["outcome"] == "complete":
            summary["complete"] += 1
            print(f"    complete in {attempt['wall_s']:.0f}s", flush=True)
        else:
            summary["failed"] += 1
            print(f"    {attempt['outcome'].upper()} (return code {attempt['returncode']}); log {attempt['log']}", flush=True)
            if attempt["outcome"] in ("killed_oom_suspected", "oom"):
                print("    stopping the stage: an OOM must be investigated before anything else runs", flush=True)
                break
    write_manifest(bench, stage, specs)
    summary["total"] = len(specs)
    return summary


def _client_facts(spec: dict) -> dict:
    """Client count and largest client from a saved partition, when one exists."""
    from src.data.prepare import bundle_dir

    meta_path = bundle_dir(spec["cfg"]) / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("kind") == "natural":
        seen = meta["holdout"]["seen_clients"]
        return {"num_clients": seen}
    return {"num_clients": int(spec["cfg"]["partition"].get("num_clients") or 0)}


# ---- selections -------------------------------------------------------------------
def _tuning_rows(bench: dict, stage: str, regime: str) -> list[dict]:
    rows = []
    for spec in P.plan_stage(bench, stage):
        if spec["regime"] != regime:
            continue
        if not is_complete(spec):
            raise RuntimeError(f"tuning run not complete: {spec['key']} ({run_state(spec)})")
        run_dir = Path(spec["run_dir"])
        final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if final.get("test_evaluated") or any(k in final for k in ("unseen", "test", "client_test")):
            raise RuntimeError(f"{spec['key']} evaluated held-out data; tuning must use validation only")
        f1 = final.get("val", {}).get("macro_f1")
        rows.append({"method": spec["method"], "variant": spec["variant"], "fl": spec["cfg"]["fl"],
                     "val_macro_f1": f1 if f1 is not None and np.isfinite(f1) else None,
                     "diverged": bool(final.get("diverged")), "run_dir": str(run_dir),
                     "git_commit": meta["environment"]["git"].get("commit"),
                     "git_dirty": meta["environment"]["git"].get("dirty"),
                     "official_environment": meta["official_environment"]})
    return rows


def _rank_key(row: dict, keys: list[str], directions: list[int]):
    f1 = row["val_macro_f1"]
    return (0 if f1 is not None else 1, -(f1 or 0.0), *[d * row["fl"][k] for k, d in zip(keys, directions)])


def select_base(bench: dict, regime: str) -> dict:
    rows = _tuning_rows(bench, "tune_base", regime)
    # Frozen rule: highest validation macro-F1; ties -> smaller lr, then larger weight decay.
    best = sorted(rows, key=lambda r: _rank_key(r, ["lr", "weight_decay"], [1, -1]))[0]
    g = bench["base_grid"]
    sel = {"regime": regime, "lr": best["fl"]["lr"], "weight_decay": best["fl"]["weight_decay"],
           "val_macro_f1": best["val_macro_f1"], "criterion": "seen-client pooled validation macro-F1, last round",
           "tie_break": ["smaller_lr", "larger_weight_decay"], "tuning_seed": bench["tuning_seed"],
           "on_grid_edge": {"lr": best["fl"]["lr"] in (min(g["lr"]), max(g["lr"])),
                            "weight_decay": best["fl"]["weight_decay"] in (min(g["weight_decay"]), max(g["weight_decay"]))},
           "official": all(r["official_environment"] and not r["git_dirty"] for r in rows),
           "git_commits": sorted({str(r["git_commit"]) for r in rows}), "selected_utc": utc(),
           "grid": [{k: r[k] for k in ("variant", "val_macro_f1", "diverged")} for r in rows]}
    _write_selection(P.selection_path(bench, "base", regime), sel)
    return sel


def select_algorithms(bench: dict, regime: str) -> dict:
    rows = _tuning_rows(bench, "tune_algorithms", regime)
    out = {"regime": regime, "tuning_seed": bench["tuning_seed"], "selected_utc": utc(),
           "criterion": "seen-client pooled validation macro-F1, last round; ties -> smaller value",
           "official": all(r["official_environment"] and not r["git_dirty"] for r in rows),
           "git_commits": sorted({str(r["git_commit"]) for r in rows}), "algorithms": {}}
    for method, grid in bench["algorithm_grids"].items():
        cand = [r for r in rows if r["method"] == method]
        keys = sorted(grid)
        best = sorted(cand, key=lambda r: _rank_key(r, keys, [1] * len(keys)))[0]
        out["algorithms"][method] = {
            "hyperparameters": {k: best["fl"][k] for k in keys}, "val_macro_f1": best["val_macro_f1"],
            "on_grid_edge": {k: best["fl"][k] in (min(grid[k]), max(grid[k])) for k in keys},
            "grid": [{"variant": r["variant"], "val_macro_f1": r["val_macro_f1"], "diverged": r["diverged"]} for r in cand],
        }
    ref = out["algorithms"]["fedavg"]["val_macro_f1"]
    out["target_val_macro_f1"] = None if ref is None else float(bench["target_fraction_of_tuned_fedavg"]) * ref
    _write_selection(P.selection_path(bench, "algorithms", regime), out)
    return out


def _write_selection(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "selected_utc"} == {k: v for k, v in obj.items() if k != "selected_utc"}
        if not same:
            raise RuntimeError(f"{path} exists with a different selection; selections are never overwritten")
        return
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ---- manifest -------------------------------------------------------------------
def write_manifest(bench: dict, stage: str, specs: list[dict] | None = None) -> Path:
    specs = specs if specs is not None else P.plan_stage(bench, stage)
    rows = []
    for spec in specs:
        run_dir = Path(spec["run_dir"])
        row = {"key": spec["key"], "stage": stage, "regime": spec["regime"], "method": spec["method"],
               "variant": spec["variant"], "seed": spec["seed"], "kind": spec["kind"], "state": run_state(spec),
               "config_hash": spec["config_hash"], "run_dir": str(run_dir.relative_to(PROJECT_ROOT))
               if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
               "attempts": len(read_status(spec)["attempts"])}
        if row["state"] == "complete":
            meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            rec = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
            row.update(git_commit=meta["environment"]["git"].get("commit"), git_dirty=meta["environment"]["git"].get("dirty"),
                       official_environment=meta["official_environment"], partition_sha256=meta["partition"]["sha256"],
                       dataset_revision=meta["config"]["data"].get("revision"), vocabulary_sha256=meta.get("vocabulary_sha256"),
                       start_utc=rec["start_utc"], end_utc=rec["end_utc"], wall_s=rec["wall_s"])
        rows.append(row)
    out = P.stage_dir(bench, stage)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps({"protocol": bench["name"], "stage": stage, "written_utc": utc(),
                                "counts": {s: sum(r["state"] == s for r in rows) for s in
                                           ("complete", "failed", "incomplete", "pending")},
                                "runs": rows}, indent=1), encoding="utf-8")
    return path


# ---- verification ---------------------------------------------------------------
def verify_stage(bench: dict, stage: str) -> dict:
    """Re-validate every complete run and check consistency across runs.

    * every COMPLETE run still passes ``validate_run`` (files, seed, config hash,
      partition hash, rounds, held-out metrics);
    * per regime: one feature bundle and one vocabulary;
    * per regime: one partition hash for natural clients (seed-independent),
      one per seed for synthetic partitions (identical across methods);
    * validated client diagnostics of the regime use the same partition.
    """
    specs = P.plan_stage(bench, stage)
    problems, runs = [], []
    for spec in specs:
        if not is_complete(spec):
            continue
        for p in validate_run(spec):
            problems.append(f"{spec['key']}: {p}")
        meta = json.loads((Path(spec["run_dir"]) / "run_metadata.json").read_text(encoding="utf-8"))
        runs.append({"key": spec["key"], "regime": spec["regime"], "seed": spec["seed"],
                     "kind": meta["partition"]["type"], "partition": meta["partition"]["sha256"],
                     "vocabulary": meta.get("vocabulary_sha256"), "bundle": Path(meta["bundle_dir"]).name,
                     "commit": meta["environment"]["git"].get("commit"), "dirty": meta["environment"]["git"].get("dirty"),
                     "official": meta["official_environment"]})
    for regime in sorted({r["regime"] for r in runs}):
        rr = [r for r in runs if r["regime"] == regime]
        for field in ("bundle", "vocabulary"):
            if len({r[field] for r in rr}) > 1:
                problems.append(f"{regime}: runs use different {field}s {sorted({r[field] for r in rr})}")
        natural = rr[0]["kind"] == "natural"
        groups = {"all": rr} if natural else {s: [r for r in rr if r["seed"] == s] for s in {r["seed"] for r in rr}}
        for key, g in groups.items():
            if len({r["partition"] for r in g}) > 1:
                problems.append(f"{regime} seed {key}: runs use different partitions")
        diag_root = P.results_root(bench) / "diagnostics" / regime
        for marker in sorted(diag_root.glob("seed*/DIAG_COMPLETE")):
            seed = int(marker.parent.name[4:])
            want = {r["partition"] for r in rr if natural or r["seed"] == seed}
            got = json.loads(marker.read_text(encoding="utf-8"))["partition_sha256"]
            if want and got not in want:
                problems.append(f"{regime}: diagnostics {marker.parent.name} used partition {got[:12]}, runs use {sorted(want)}")
    report = {"stage": stage, "verified_utc": utc(), "complete_runs": len(runs),
              "not_complete": [s["key"] for s in specs if not is_complete(s)],
              "commits": sorted({str(r["commit"]) for r in runs}),
              "all_official": all(r["official"] and not r["dirty"] for r in runs),
              "problems": problems, "ok": not problems}
    out = P.stage_dir(bench, stage)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
