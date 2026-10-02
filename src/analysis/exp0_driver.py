"""Driver shared by Experiment 0A and 0B: run the three conditions, then analyse.

Modes
-----
pilot  one seed that is NOT an official seed, a few rounds. For timing and
       plumbing. Stored under ``<experiment>_pilot``. Pilot output must not be
       used to change the protocol.
full   the official seeds. Refuses to start unless every official
       precondition holds (Kaggle, clean git commit, protocol frozen, seed
       list complete, dataset revision pinned).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.analysis import exp0
from src.runner import is_complete, run_baseline, run_federated
from src.utils.config import PROJECT_ROOT, deep_merge
from src.utils.env import collect_environment
from src.utils.schema import collect_tables


def resolve(cfg: dict, mode: str, seeds: list[int] | None) -> tuple[dict, list[int]]:
    """Return the config and seed list actually used for ``mode``."""
    if mode == "pilot":
        pilot = cfg["pilot"]
        if pilot["seed"] in (cfg["experiment"].get("seeds") or []):
            raise ValueError("the pilot seed must not be one of the official seeds")
        cfg = deep_merge(cfg, {
            "experiment": {"name": cfg["experiment"]["name"] + "_pilot"},
            "fl": {"rounds": pilot["rounds"]},
            "centralized": {"epochs": pilot["centralized_epochs"]},
        })
        return cfg, [pilot["seed"]]
    if mode != "full":
        raise ValueError(f"unknown mode {mode!r}")
    official = list(cfg["experiment"].get("seeds") or [])
    seeds = list(seeds) if seeds else official
    problems = exp0.official_preconditions(cfg, collect_environment(PROJECT_ROOT), official)
    if not set(seeds) <= set(official):
        problems.append("a requested seed is not in experiment.seeds")
    if problems:
        raise SystemExit("Refusing to start the official run:\n  - " + "\n  - ".join(problems))
    return cfg, seeds


def run_conditions(cfg: dict, seeds: list[int], create_partitions: bool) -> None:
    """Train every missing (cell, condition, seed). Finished runs are skipped."""
    central_cell = cfg["centralized_cell"]
    for seed in seeds:
        for cell in cfg["cells"]:
            cell_cfg = exp0.cell_config(cfg, cell)
            for condition in exp0.CONDITIONS:
                if condition == "centralized" and cell != central_cell:
                    continue
                run_dir = exp0.run_dir_for(cfg, cell, condition, seed)
                if is_complete(run_dir):
                    print(f"skip  {cell}/{condition}/seed{seed} (finished)", flush=True)
                    continue
                print(f"run   {cell}/{condition}/seed{seed}", flush=True)
                if condition == "centralized":
                    run_baseline(cell_cfg, seed, "centralized", run_dir, create_partitions)
                else:
                    run_federated(deep_merge(cell_cfg, {"fl": {"algorithm": condition}}), seed, run_dir, create_partitions)


def analyse(cfg: dict, seeds: list[int]) -> dict:
    """Per-seed diagnostics for every complete cell, then the across-seed summary."""
    exp_dir = exp0.experiment_dir(cfg)
    central_cell = cfg["centralized_cell"]
    for cell in cfg["cells"]:
        for seed in seeds:
            dirs = {c: exp0.run_dir_for(cfg, central_cell if c == "centralized" else cell, c, seed) for c in exp0.CONDITIONS}
            out = exp_dir / cell / "diagnostics" / f"seed{seed}"
            if (out / "diagnostic.json").exists() or not all(is_complete(d) for d in dirs.values()):
                continue
            print(f"analyse {cell}/seed{seed}", flush=True)
            exp0.analyze_cell(dirs, out, cfg["diagnostic"], seed)
    summary = exp0.aggregate(exp_dir, list(cfg["cells"]))
    skew = [c for c, part in cfg["cells"].items() if part["scheme"].startswith("dirichlet")]
    if skew and "iid" in cfg["cells"]:
        summary["decision_rule_0a"] = exp0.decision_rule_0a(summary, skew[0])
    summary["schema"] = collect_tables(exp_dir)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    write_summary_md(exp_dir, cfg, summary)
    return summary


def write_summary_md(exp_dir: Path, cfg: dict, summary: dict) -> None:
    """Human-readable view of summary.json. Every number is read from the files."""
    runs = []
    if (exp_dir / "runs.csv").exists():
        import csv

        with open(exp_dir / "runs.csv", "r", encoding="utf-8") as f:
            runs = list(csv.DictReader(f))
    official = bool(runs) and all(r["official_environment"] == "True" for r in runs)
    pilot = cfg["experiment"]["name"].endswith("_pilot")
    lines = [f"# {cfg['experiment']['name']}", ""]
    if pilot:
        lines += ["> **PILOT (engineering run).** Not a scientific result. Must not be used to change the protocol.", ""]
    if not official:
        lines += ["> **NOT AN OFFICIAL RESULT.** At least one run was produced outside Kaggle.", ""]
    if "iid" in cfg["cells"]:
        lines += ["Scope: this experiment describes ordinary label heterogeneity under a synthetic partition. "
                  "It is not evidence for or against natural-client lexical confounding.", ""]
    lines += ["## Runs", "", "| cell | algorithm | seed | test macro-F1 | test accuracy | worst-client F1 | git | official |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["run_dir"], int(r["seed"]))):
        cell = Path(r["run_dir"]).parts[-3]
        lines.append(f"| {cell} | {r['algorithm']} | {r['seed']} | {r['test_macro_f1']} | {r['test_accuracy']} | "
                     f"{r['worst_client_macro_f1']} | {(r['git_commit'] or '')[:8]} | {r['official_environment']} |")
    lines += ["", "## Coefficient bias vs client label exposure (supported features)", "",
              "| cell | condition | statistic | per seed | mean | same sign in all seeds |", "|---|---|---|---|---|---|"]
    for cell, entry in summary["cells"].items():
        for cond, stats in entry["bias_vs_exposure"].items():
            for stat, v in stats.items():
                per_seed = ", ".join(f"{x:.4f}" for x in v["per_seed"])
                mean = "n/a" if v.get("mean") is None else f"{v['mean']:.4f}"
                lines.append(f"| {cell} | {cond} | {stat} | {per_seed} | {mean} | {v['same_sign_all_seeds']} |")
    if "decision_rule_0a" in summary:
        lines += ["", "## Frozen decision rule (EXPERIMENT_PLAN.md section 5), evaluated mechanically", "",
                  "```", json.dumps(summary["decision_rule_0a"], indent=1), "```"]
    problems = summary["schema"]["problems"]
    lines += ["", "## Schema validation", "", f"{summary['schema']['runs']} runs; {len(problems)} with missing fields."]
    lines += [f"- `{k}`: {v}" for k, v in problems.items()]
    lines += ["", "## Interpretation", "", "_To be written by the authors after inspecting the results._", ""]
    (exp_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
