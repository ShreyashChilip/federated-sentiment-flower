"""Systematic FL strategy benchmark: staged, resumable, one process per run.

Stages and the experiment matrix are defined in configs/benchmark.yaml
(protocol in docs/BENCHMARK_PROTOCOL.md).

Usage
  python experiments/run_benchmark.py run --stages pilot
  python experiments/run_benchmark.py run --stages tune_base tune_algorithms screening --max-hours 11
  python experiments/run_benchmark.py run --stages screening --only amazon_vg/fedavg/seed42   # single run
  python experiments/run_benchmark.py run --stages screening --retry-failed                  # after investigating
  python experiments/run_benchmark.py status --stages screening
  python experiments/run_benchmark.py estimate --stages screening
  python experiments/run_benchmark.py select --stages tune_base
  python experiments/run_benchmark.py verify --stages pilot          # re-validate artifacts and cross-run consistency

``run`` resumes by default: complete runs are skipped, failed runs are
reported and not retried unless asked. After a tuning stage completes, its
selection (frozen rule) is written automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import orchestrator as O  # noqa: E402
from src.benchmark import plan as P  # noqa: E402
from src.benchmark import resources  # noqa: E402


def selections_for(bench: dict, stage: str) -> None:
    kind = bench["stages"][stage]["kind"]
    for regime in bench["stages"][stage].get("regimes", []):
        if kind == "tune_base":
            sel = O.select_base(bench, regime)
            print(f"  selected {regime}: lr={sel['lr']} wd={sel['weight_decay']} val_f1={sel['val_macro_f1']:.4f} "
                  f"edge={sel['on_grid_edge']}", flush=True)
        elif kind == "tune_algorithms":
            sel = O.select_algorithms(bench, regime)
            for method, entry in sel["algorithms"].items():
                print(f"  selected {regime}/{method}: {entry['hyperparameters']} val_f1={entry['val_macro_f1']} "
                      f"edge={entry['on_grid_edge']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status", "estimate", "select", "worker", "manifest", "verify"])
    parser.add_argument("--bench", default="benchmark.yaml", help="benchmark config in configs/ (or a path)")
    parser.add_argument("--stages", nargs="*", default=[])
    parser.add_argument("--only", help="run only specs whose key contains this text")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="used by the memory check; runs are sequential")
    parser.add_argument("--max-hours", type=float, help="launch no new run after this many hours")
    parser.add_argument("--timeout-hours", type=float, help="kill a single run after this many hours")
    parser.add_argument("--force", action="store_true", help="launch even if the resource check refuses")
    parser.add_argument("--allow-unofficial", action="store_true", help="local software checks only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spec", type=Path, help="worker: spec file")
    args = parser.parse_args()

    if args.command == "worker":
        raise SystemExit(O.worker(args.spec))

    bench = P.load_benchmark(args.bench)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    timeout = args.timeout_hours * 3600 if args.timeout_hours else None
    verify_failed = False
    for stage in args.stages:
        if args.command == "run":
            summary = O.run_stage(bench, stage, only=args.only, retry_failed=args.retry_failed, jobs=args.jobs,
                                  deadline=deadline, force=args.force, allow_unofficial=args.allow_unofficial,
                                  dry_run=args.dry_run, timeout_s=timeout)
            print(f"stage {stage}: {json.dumps(summary)}", flush=True)
            if bench["stages"][stage]["kind"] in ("tune_base", "tune_algorithms") and not args.only and not args.dry_run:
                if summary["complete"] == summary["total"]:
                    selections_for(bench, stage)
                else:
                    print(f"stage {stage} not complete; no selection written; later stages cannot be planned", flush=True)
                    break
            if summary["complete"] != summary["total"] and not args.dry_run:
                print(f"stage {stage} incomplete: stopping here (later stages depend on it)", flush=True)
                break
        elif args.command == "select":
            selections_for(bench, stage)
        elif args.command == "manifest":
            print(O.write_manifest(bench, stage))
        elif args.command == "verify":
            try:
                rep = O.verify_stage(bench, stage)
            except P.MissingSelection as exc:
                # Not an artifact problem: the stage depends on a selection that does not exist yet.
                print(f"verify {stage}: NOT PLANNABLE YET, nothing to verify ({exc})")
                continue
            print(f"verify {stage}: {rep['complete_runs']} complete run(s), {len(rep['not_complete'])} not complete, "
                  f"commits {[c[:8] for c in rep['commits']]}, all_official={rep['all_official']}, "
                  f"{'OK' if rep['ok'] else 'PROBLEMS'}")
            for p in rep["problems"]:
                print("  PROBLEM", p)
            verify_failed = verify_failed or not rep["ok"]
        elif args.command in ("status", "estimate"):
            try:
                specs = P.plan_stage(bench, stage)
            except P.MissingSelection as exc:
                print(f"stage {stage}: cannot be planned yet ({exc})")
                continue
            counts = {}
            for spec in specs:
                state = O.run_state(spec)
                counts[state] = counts.get(state, 0) + 1
                if args.command == "estimate":
                    ok, rep = resources.check(spec, jobs=args.jobs, **O._client_facts(spec))
                    est = rep["estimate"]
                    text = (f"RSS {est['rss_gb']:.2f} GB, scratch {est['scratch_disk_gb']:.1f} GB, "
                            f"mapped {est['mapped_matrix_gb']:.1f} GB") if est["known"] else est["reason"]
                    print(f"  {'OK ' if ok else 'NO '} {spec['key']}: {text} {rep.get('problems') or ''}")
                elif state != "complete":
                    print(f"  {state:10s} {spec['key']}")
            print(f"stage {stage}: {counts}")

    if verify_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
