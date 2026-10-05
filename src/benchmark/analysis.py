"""Tables, statistics and figures from completed benchmark runs.

Only runs with a COMPLETE marker are read. Seeds are the replication unit:
every across-run statistic is computed over per-seed values; client-level
observations of one run are summarized inside that run first and never
treated as independent replicates.

Per-client metrics are noisy for clients with few evaluation records. The
primary per-client statistics therefore use clients with at least
``MIN_EVAL_N`` evaluated records (fixed before any result; seen clients'
client-test split holds ~10% of their records). Every client is still written
to the per-client table.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from src.benchmark import plan as P
from src.benchmark.orchestrator import is_complete
from src.metrics.convergence import bytes_to_target, rounds_to_target
from src.metrics.fairness import client_dispersion
from src.statistics.stats import holm_correction, paired_comparison, summarize

MIN_EVAL_N = 5
FL_METHODS = ["fedavg", "fedprox", "fednova", "scaffold", "fedadam", "fedadagrad", "fedyogi"]
BASELINES = ["centralized", "local_only"]
ORDER = BASELINES + FL_METHODS
COLORS = {"fedavg": "#2a78d6", "fedprox": "#eb6834", "fednova": "#1baf7a", "scaffold": "#eda100",
          "fedadam": "#e87ba4", "fedadagrad": "#008300", "fedyogi": "#4a3aa7",
          "centralized": "#444444", "local_only": "#999999"}
LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova", "scaffold": "SCAFFOLD",
          "fedadam": "FedAdam", "fedadagrad": "FedAdagrad", "fedyogi": "FedYogi",
          "centralized": "Centralized", "local_only": "Local-only"}
DISP_KEYS = ("mean", "median", "std", "p10", "p25", "p75", "p90", "worst", "best", "iqr", "spread")


def _load(run_dir: Path, name: str):
    path = run_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def complete_specs(bench: dict, stage: str) -> list[dict]:
    return [s for s in P.plan_stage(bench, stage) if is_complete(s)]


def _targets(bench: dict) -> dict:
    out = {}
    for regime in bench["regimes"]:
        try:
            out[regime] = P.read_selection(bench, "algorithms", regime).get("target_val_macro_f1")
        except P.MissingSelection:
            out[regime] = None
    return out


def run_row(spec: dict, target: float | None) -> tuple[dict, list[dict], list[dict]]:
    run_dir = Path(spec["run_dir"])
    final, history = _load(run_dir, "final"), _load(run_dir, "history") or []
    clients, unseen = _load(run_dir, "clients") or [], _load(run_dir, "unseen_clients") or []
    meta = _load(run_dir, "run_metadata")
    row = {"regime": spec["regime"], "method": spec["method"], "variant": spec["variant"], "seed": spec["seed"],
           "kind": spec["kind"], "diverged": bool(final.get("diverged")),
           "git_commit": meta["environment"]["git"].get("commit"), "official": meta["official_environment"],
           "partition_sha256": meta["partition"]["sha256"]}
    held = final.get("held_out_group")
    for group, prefix in (("val", "val"), ("client_test", "ct"), (held, "held")):
        m = final.get(group) if group else None
        if not m or not m.get("n"):
            continue
        row[f"{prefix}_accuracy"], row[f"{prefix}_macro_f1"] = m["accuracy"], m["macro_f1"]
        for k, v in enumerate(m.get("per_class_f1", [])):
            row[f"{prefix}_f1_class{k}"] = v
        if prefix == "held":
            for k, (p, r) in enumerate(zip(m.get("per_class_precision", []), m.get("per_class_recall", []))):
                row[f"held_precision_class{k}"], row[f"held_recall_class{k}"] = p, r
    # per-client distributions (all clients, and clients with >= MIN_EVAL_N records)
    seen_vals = [(c.get("client_test_macro_f1"), c.get("client_test_n", 0)) for c in clients if c.get("client_test_macro_f1") is not None]
    groups = {"ct_client": seen_vals, "held_client": [(u["macro_f1"], u["n"]) for u in unseen]}
    for prefix, vals in groups.items():
        for suffix, minimum in (("", 0), ("_min5", MIN_EVAL_N)):
            disp = client_dispersion([f for f, n in vals if n >= minimum])
            for key in DISP_KEYS:
                if key in disp:
                    row[f"{prefix}{suffix}_f1_{key}"] = disp[key]
            row[f"{prefix}{suffix}_count"] = disp.get("num_clients", 0)
    if "ct_client_min5_f1_mean" in row and "held_client_min5_f1_mean" in row:
        row["gap_mean_client_f1"] = row["ct_client_min5_f1_mean"] - row["held_client_min5_f1_mean"]
        row["gap_p10_client_f1"] = row["ct_client_min5_f1_p10"] - row["held_client_min5_f1_p10"]
    if "ct_macro_f1" in row and "held_macro_f1" in row and held == "unseen":
        row["gap_pooled_macro_f1"] = row["ct_macro_f1"] - row["held_macro_f1"]
    comm = final.get("communication", {})
    row["total_bytes"] = comm.get("training_total_bytes", 0)
    row["bytes_per_round"] = comm.get("mean_total_bytes_per_round")
    res = final.get("resources", {})
    row.update(wall_s=res.get("wall_s"), mean_round_wall_s=res.get("mean_round_wall_s"), peak_rss_mb=res.get("peak_rss_mb"))
    if history and spec["kind"] == "federated":
        vals = [h.get("val_macro_f1") for h in history if h.get("val_macro_f1") is not None]
        if len(vals) >= 10:
            row["val_f1_last10_std"] = float(np.std(vals[-10:]))
            row["val_f1_best_minus_final"] = float(max(vals) - vals[-1])
        if target:
            row["target_val_f1"] = target
            row["rounds_to_target"] = rounds_to_target(history, "val_macro_f1", target)
            row["bytes_to_target"] = bytes_to_target(history, "val_macro_f1", target)
    rounds = [{"regime": spec["regime"], "method": spec["method"], "seed": spec["seed"], **h} for h in history]
    per_client = [{"regime": spec["regime"], "method": spec["method"], "seed": spec["seed"], "population": "seen",
                   "client": c["client_id"], "n_train": c.get("n_train"), "n_eval": c.get("client_test_n"),
                   "macro_f1": c.get("client_test_macro_f1"), "accuracy": c.get("client_test_accuracy"),
                   "val_macro_f1": c.get("val_macro_f1"), "participation": c.get("participation")} for c in clients]
    per_client += [{"regime": spec["regime"], "method": spec["method"], "seed": spec["seed"], "population": "unseen",
                    "client": u["client_code"], "n_train": 0, "n_eval": u["n"], "macro_f1": u["macro_f1"],
                    "accuracy": u["accuracy"]} for u in unseen]
    return row, rounds, per_client


def _write_csv(path: Path, rows: list[dict], gz: bool = False) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if gz else open
    with opener(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


PRIMARY_METRICS = ["held_macro_f1", "held_accuracy", "ct_macro_f1", "val_macro_f1",
                   "ct_client_min5_f1_mean", "ct_client_min5_f1_p10", "ct_client_min5_f1_worst",
                   "ct_client_min5_f1_std", "held_client_min5_f1_mean", "held_client_min5_f1_median",
                   "held_client_min5_f1_p10", "held_client_min5_f1_p25", "held_client_min5_f1_worst",
                   "held_client_min5_f1_std", "held_client_min5_f1_iqr", "gap_pooled_macro_f1",
                   "gap_mean_client_f1", "gap_p10_client_f1", "rounds_to_target", "total_bytes", "wall_s",
                   "val_f1_last10_std"]


def summarize_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per (regime, method, metric): across-seed summary. Paired comparisons vs FedAvg."""
    summary, paired = [], []
    for regime in sorted({r["regime"] for r in runs}):
        rr = [r for r in runs if r["regime"] == regime]
        methods = [m for m in ORDER if any(r["method"] == m for r in rr)]
        for metric in PRIMARY_METRICS:
            by = {m: {r["seed"]: r.get(metric) for r in rr if r["method"] == m} for m in methods}
            for m in methods:
                vals = [v for v in by[m].values() if v is not None and np.isfinite(v)]
                if not vals:
                    continue
                s = summarize(vals)
                summary.append({"regime": regime, "metric": metric, "method": m, **s,
                                "per_seed": json.dumps({str(k): v for k, v in sorted(by[m].items())})})
            family = []
            for m in methods:
                if m == "fedavg" or "fedavg" not in by:
                    continue
                seeds = sorted(k for k in by[m] if k in by["fedavg"] and by[m][k] is not None
                               and by["fedavg"][k] is not None and np.isfinite(by[m][k]) and np.isfinite(by["fedavg"][k]))
                if len(seeds) < 2:
                    continue
                a = [by[m][k] for k in seeds]
                b = [by["fedavg"][k] for k in seeds]
                comp = paired_comparison(a, b)
                d = np.asarray(a) - np.asarray(b)
                family.append({"regime": regime, "metric": metric, "method": m, "reference": "fedavg", "seeds": len(seeds),
                               "positive_diffs": int((d > 0).sum()), "negative_diffs": int((d < 0).sum()), **comp})
            # Holm within (regime, metric) over the comparisons that have a p-value;
            # a comparison with all-zero differences has none and is reported as such.
            tested = [f for f in family if f["wilcoxon_p"] is not None]
            for f, adj in zip(tested, holm_correction([f["wilcoxon_p"] for f in tested]) if tested else []):
                f["wilcoxon_p_holm"] = adj
            paired += family
    return summary, paired


# ---- figures ------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#888888", "legend.frameon": False, "figure.dpi": 150})
    return plt


def _style(m):
    return {"color": COLORS[m], "lw": 1.6 if m in FL_METHODS else 1.2, "ls": "-" if m in FL_METHODS else "--"}


def plot_convergence(rounds: list[dict], out: Path) -> None:
    plt = _plt()
    for regime in sorted({r["regime"] for r in rounds}):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for m in FL_METHODS:
            rows = [r for r in rounds if r["regime"] == regime and r["method"] == m and r.get("val_macro_f1") is not None]
            if not rows:
                continue
            rs = sorted({r["round"] for r in rows})
            mat = [[r["val_macro_f1"] for r in rows if r["round"] == t] for t in rs]
            mean = [np.mean(v) for v in mat]
            ax.plot(rs, mean, label=LABELS[m], **_style(m))
            ax.fill_between(rs, [min(v) for v in mat], [max(v) for v in mat], color=COLORS[m], alpha=0.12, lw=0)
        ax.set_xlabel("Round")
        ax.set_ylabel("Seen-client validation macro-F1")
        ax.set_title(f"{regime}: convergence (mean, band = min-max over seeds)", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"convergence_{regime}.png")
        plt.close(fig)


def plot_metric_dots(runs: list[dict], out: Path, metrics: list[tuple[str, str]]) -> None:
    plt = _plt()
    for regime in sorted({r["regime"] for r in runs}):
        rr = [r for r in runs if r["regime"] == regime]
        present = [(k, lab) for k, lab in metrics if any(r.get(k) is not None for r in rr)]
        if not present:
            continue
        fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        methods = [m for m in ORDER if any(r["method"] == m for r in rr)]
        for ax, (key, label) in zip(axes, present):
            for i, m in enumerate(methods):
                vals = [r[key] for r in rr if r["method"] == m and r.get(key) is not None]
                if vals:
                    ax.scatter(vals, [i] * len(vals), s=14, color=COLORS[m], zorder=3, edgecolors="white", linewidths=0.5)
                    ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color="#222222", lw=1.2)
            ax.set_title(label, fontsize=8)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([LABELS[m] for m in methods])
        fig.suptitle(f"{regime} (dots = seeds, bar = mean)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"metrics_{regime}.png")
        plt.close(fig)


def plot_client_ecdf(clients: list[dict], out: Path, seed: int) -> None:
    plt = _plt()
    for regime in sorted({c["regime"] for c in clients}):
        pops = sorted({c["population"] for c in clients if c["regime"] == regime})
        fig, axes = plt.subplots(1, len(pops), figsize=(4.6 * len(pops), 3.3), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, pop in zip(axes, pops):
            for m in ORDER:
                v = np.sort([c["macro_f1"] for c in clients if c["regime"] == regime and c["population"] == pop
                             and c["method"] == m and c["seed"] == seed and c["macro_f1"] is not None
                             and (c["n_eval"] or 0) >= MIN_EVAL_N])
                if len(v):
                    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=LABELS[m], **_style(m))
            ax.set_xlabel(f"Per-client macro-F1 ({pop} clients, >= {MIN_EVAL_N} records)")
            ax.set_title(f"{regime}, seed {seed}", fontsize=9)
        axes[0].set_ylabel("Fraction of clients")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"client_ecdf_{regime}.png")
        plt.close(fig)


def analyze_stage(bench: dict, stage: str, out_dir: Path | None = None) -> dict:
    out = out_dir or (P.stage_dir(bench, stage) / "analysis")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    targets = _targets(bench)
    runs, rounds, clients = [], [], []
    for spec in complete_specs(bench, stage):
        row, r, c = run_row(spec, targets.get(spec["regime"]))
        runs.append(row)
        rounds += r
        clients += c
    _write_csv(out / "runs.csv", runs)
    _write_csv(out / "rounds.csv", rounds)
    _write_csv(out / "clients.csv.gz", clients, gz=True)
    summary, paired = summarize_runs(runs)
    _write_csv(out / "summary_by_method.csv", summary)
    _write_csv(out / "paired_vs_fedavg.csv", paired)
    if runs:
        plot_convergence(rounds, out / "figures")
        plot_metric_dots(runs, out / "figures", [
            ("held_macro_f1", "Held-out pooled macro-F1"), ("held_client_min5_f1_mean", "Unseen: mean client F1"),
            ("held_client_min5_f1_p10", "Unseen: P10 client F1"), ("ct_client_min5_f1_mean", "Seen: mean client F1"),
            ("ct_client_min5_f1_p10", "Seen: P10 client F1"), ("ct_client_min5_f1_worst", "Seen: worst client F1")])
        seeds = sorted({r["seed"] for r in runs})
        plot_client_ecdf(clients, out / "figures", seeds[0])
    from src.benchmark.screening import screen

    screening = screen(runs)
    (out / "screening_rules.json").write_text(json.dumps(screening, indent=1, default=str), encoding="utf-8")
    info = {"stage": stage, "runs": len(runs), "screening_outcome": screening["outcome"], "regimes": sorted({r["regime"] for r in runs}),
            "commits": sorted({str(r["git_commit"]) for r in runs}), "all_official": all(r["official"] for r in runs),
            "min_eval_n": MIN_EVAL_N, "targets": targets}
    (out / "analysis_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
