"""Collect finished runs of an experiment and write ``summary.md``.

Everything in the summary is computed from the ``final.json``,
``history.json`` and ``run_metadata.json`` files of the runs; nothing is typed
in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.statistics.stats import holm_correction, paired_comparison, summarize


def _read(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect(experiment_dir) -> list[dict]:
    """One record per finished run, grouped later by condition directory."""
    runs = []
    for final_path in sorted(Path(experiment_dir).glob("*/seed*/final.json")):
        run_dir = final_path.parent
        meta, final = _read(run_dir / "run_metadata.json"), _read(final_path)
        cfg = meta["config"]
        kind = final["kind"]
        label = kind if kind != "federated" else final["algorithm"]
        if label == "fedprox":
            label += f"(mu={cfg['fl']['mu']})"
        runs.append({
            "condition": run_dir.parent.name, "label": label, "kind": kind, "seed": meta["seed"],
            "final": final, "meta": meta, "run_dir": str(run_dir),
        })
    return runs


def headline(run: dict) -> dict:
    """The scalar metrics that are tabulated for a run."""
    f = run["final"]
    if run["kind"] == "local_only":
        return {
            "test_macro_f1": f["global_test_macro_f1"]["mean"],
            "test_accuracy": f["global_test_accuracy"]["mean"],
            "worst_client_f1": f["client_test_macro_f1"].get("worst"),
        }
    out = {"test_macro_f1": f["test"]["macro_f1"], "test_accuracy": f["test"]["accuracy"],
           "val_macro_f1": f["val"]["macro_f1"]}
    if run["kind"] == "federated":
        out["worst_client_f1"] = f["client_test_macro_f1"].get("worst")
        out["client_f1_std"] = f["client_test_macro_f1"].get("std")
        out["total_comm_mb"] = f["communication"]["training_total_bytes"] / 1e6
        out["rounds_to_target"] = f["rounds_to_target"]
        out["client_cpu_s"] = f["resources"]["client_cpu_s"]
    return out


def _fmt(s: dict, digits: int = 4) -> str:
    if s["mean"] is None:
        return "n/a"
    if s["std"] is None:
        return f"{s['mean']:.{digits}f} (n=1)"
    return f"{s['mean']:.{digits}f} +/- {s['std']:.{digits}f} [{s['ci95_low']:.{digits}f}, {s['ci95_high']:.{digits}f}]"


def write_summary(experiment_dir, interpretation: str = "", next_step: str = "") -> Path:
    experiment_dir = Path(experiment_dir)
    runs = collect(experiment_dir)
    if not runs:
        raise RuntimeError(f"no finished runs under {experiment_dir}")
    conditions: dict[str, list[dict]] = {}
    for run in runs:
        conditions.setdefault(run["condition"], []).append(run)

    first = runs[0]["meta"]
    cfg, env = first["config"], first["environment"]
    platforms = sorted({r["meta"]["environment"]["platform_kind"] for r in runs})
    official = platforms == ["kaggle"]
    lines = [f"# {experiment_dir.name}", ""]
    if not official:
        lines += ["> **NOT AN OFFICIAL RESULT.** At least one run was produced outside Kaggle "
                  f"(platforms: {', '.join(platforms)}). Do not report these numbers.", ""]
    lines += [
        "## Configuration", "",
        f"- Dataset: {first['dataset']['dataset']} (train rows used {first['dataset']['train_size']}, "
        f"test rows used {first['dataset']['test_size']}, features {first['dataset']['num_features']})",
        f"- Model: {cfg['model']['name']}",
        f"- Partition: {cfg['partition']['scheme']}, N = {cfg['partition']['num_clients']}"
        + (f", alpha = {cfg['partition'].get('alpha')}" if cfg['partition']['scheme'] == 'dirichlet' else ""),
        f"- Seeds: {sorted({r['seed'] for r in runs})}",
        f"- FL: rounds = {cfg['fl']['rounds']}, C = {cfg['fl']['fraction_fit']}, E = {cfg['fl']['local_epochs']}, "
        f"batch = {cfg['fl']['batch_size']}, lr = {cfg['fl']['lr']}, gamma = {cfg['fl']['lr_decay']}",
        f"- Environment: {env['platform_kind']}, Python {env['hardware']['python']}, flwr {env['packages']['flwr']}, "
        f"torch {env['packages']['torch']}, git {env['git'].get('commit')}",
        "", "## Final metrics (mean +/- std [95% CI] across seeds)", "",
    ]
    keys = ["test_macro_f1", "test_accuracy", "worst_client_f1", "client_f1_std", "total_comm_mb", "rounds_to_target", "client_cpu_s"]
    lines += ["| Condition | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    table: dict[str, dict] = {}
    for name, group in conditions.items():
        heads = {r["seed"]: headline(r) for r in group}
        table[name] = {"label": group[0]["label"], "by_seed": heads}
        cells = []
        for key in keys:
            vals = [h[key] for h in heads.values() if h.get(key) is not None]
            cells.append(_fmt(summarize(vals)) if vals else "n/a")
        lines.append(f"| {group[0]['label']} ({name}) | {len(group)} | " + " | ".join(cells) + " |")

    # paired tests between federated conditions on shared seeds
    fed = [n for n, g in conditions.items() if g[0]["kind"] == "federated"]
    tests = []
    for i, a in enumerate(fed):
        for b in fed[i + 1:]:
            seeds = sorted(set(table[a]["by_seed"]) & set(table[b]["by_seed"]))
            if len(seeds) < 2:
                continue
            res = paired_comparison([table[a]["by_seed"][s]["test_macro_f1"] for s in seeds],
                                    [table[b]["by_seed"][s]["test_macro_f1"] for s in seeds])
            tests.append((table[a]["label"], table[b]["label"], res))
    lines += ["", "## Paired tests on test macro-F1 (A minus B, same seeds and partitions)", ""]
    if tests:
        raw = [t[2]["wilcoxon_p"] for t in tests]
        adj = holm_correction([p if p is not None else 1.0 for p in raw])
        lines += ["| A | B | n | mean diff | 95% CI | Cohen d_z | paired t p | Wilcoxon p | Holm-adjusted | min attainable Wilcoxon p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for (a, b, r), p_adj in zip(tests, adj):
            f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
            lines.append(f"| {a} | {b} | {r['n']} | {f(r['mean_diff'])} | [{f(r['diff_ci95_low'])}, {f(r['diff_ci95_high'])}] | "
                         f"{f(r['cohens_dz'])} | {f(r['t_p'])} | {f(r['wilcoxon_p'])} | {f(p_adj)} | {f(r['wilcoxon_min_p'])} |")
    else:
        lines.append("Fewer than two federated conditions with at least two shared seeds; no test run.")

    anomalies = []
    for run in runs:
        if run["kind"] == "federated":
            hist = _read(Path(run["run_dir"]) / "history.json")
            if any(h["reported"] == 0 for h in hist):
                anomalies.append(f"{run['label']} seed {run['seed']}: at least one round with no reporting client")
            cm = run["final"]["test"]["confusion_matrix"]
            if any(sum(row[j] for row in cm) == 0 for j in range(len(cm))):
                anomalies.append(f"{run['label']} seed {run['seed']}: final model never predicts at least one class")
    lines += ["", "## Anomalies", ""] + ([f"- {a}" for a in anomalies] or ["None detected automatically."])
    lines += ["", "## Interpretation", "", interpretation or "_To be written after inspecting the results._"]
    lines += ["", "## Next recommended experiment", "", next_step or "_To be decided._", ""]

    out = experiment_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    with open(experiment_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"official": official, "conditions": table,
                   "tests": [{"a": a, "b": b, **r} for a, b, r in tests], "anomalies": anomalies}, f, indent=1)
    return out
