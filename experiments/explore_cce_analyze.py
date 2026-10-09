"""EXPLORATORY analysis of explore_cce_v1 (docs/EXPLORATORY_CCE_PROTOCOL.md).

Applies the pre-registered rules (src/benchmark/exploratory_decision.py) to the
completed exploratory runs and writes results/exploratory/cce_v1/analysis/.
Reads the official archive read-only for the centralized seed-42 reference.

Usage
  python experiments/explore_cce_analyze.py --official <path to official results_benchmark>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import exploratory_decision as D  # noqa: E402
from src.benchmark import plan as P  # noqa: E402
from src.benchmark.orchestrator import is_complete  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def centred(path):
    w = np.load(path)["arr_0"].astype(np.float64)
    return w - w.mean(0, keepdims=True)


def weight_diagnostics(run_dir: Path, central_w: Path) -> dict:
    c, a = centred(central_w), centred(run_dir / "final_weights.npz")
    q = np.digitize(np.abs(c).ravel(), np.percentile(np.abs(c), [20, 40, 60, 80]))
    ratio = np.abs(a).ravel() / (np.abs(c).ravel() + 1e-12)
    return {"norm_W": float(np.linalg.norm(np.load(run_dir / "final_weights.npz")["arr_0"])),
            "cos_to_centralized": float((a * c).sum() / np.linalg.norm(a) / np.linalg.norm(c)),
            "ratio_by_quintile_of_central": [float(np.median(ratio[q == k])) for k in range(5)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="exploratory/cce_v1.yaml")
    ap.add_argument("--official", type=Path, required=True, help="official results_benchmark folder (read-only)")
    args = ap.parse_args()
    bench = P.load_benchmark(args.bench)
    specs = P.plan_stage(bench, "explore")
    missing = [s["key"] for s in specs if not is_complete(s)]
    if missing:
        raise SystemExit(f"{len(missing)} exploratory run(s) not complete; no decision is made on partial data:\n  " + "\n  ".join(missing))
    runs = {}
    for s in specs:
        d = Path(s["run_dir"])
        f, h = load(d / "final.json"), load(d / "history.json")
        v = [r["val_macro_f1"] for r in h if r.get("val_macro_f1") is not None]
        runs[(s["regime"], s["method"], s["variant"])] = {
            "dir": d, "V": f["val"]["macro_f1"], "sha": f["weights_sha256"], "diverged": f["diverged"],
            "last10_std": float(np.std(v[-10:])) if len(v) >= 10 else None, "curve": v}
    out = {"protocol": "explore_cce_v1", "exploratory": True, "regimes": {}}
    for reg in ("amazon_vg", "yelp_dir01"):
        r = {k[1:]: v for k, v in runs.items() if k[0] == reg}
        get = lambda m, var="reference": r[(m, var)]
        slr = {var: x["V"] for (m, var), x in r.items() if m == "fedavg"}
        exp = {var: x["V"] for (m, var), x in r.items() if m == "fedexp"}
        central = args.official / "screening" / reg / "centralized" / "seed42"
        c_val = load(central / "final.json")["val"]["macro_f1"]
        target = 0.95 * c_val
        reg_out = {
            "integrity": D.integrity(reg, get("fedavg", "server_lr1")["sha"], get("fedavg", "server_lr1")["V"], get("cce_unit")["V"]),
            "V": {"fedavg_server_lr": slr, "fedexp": exp, "cav": get("cav")["V"], "cce": get("cce")["V"],
                  "cce_unit": get("cce_unit")["V"], "fedyogi_lr3": get("fedyogi", "server_lr3")["V"]},
            "rounds_to_0.95x_centralized_val": {f"{m}/{var}": next((i + 1 for i, y in enumerate(x["curve"]) if y >= target), None)
                                                for (m, var), x in r.items()},
            "last10_std": {f"{m}/{var}": x["last10_std"] for (m, var), x in r.items()},
            "weights": {m: weight_diagnostics(get(m)["dir"], central / "final_weights.npz") for m in ("cav", "cce")},
        }
        if reg == "amazon_vg":
            stable = {m: (not get(m)["diverged"]) and (get(m)["last10_std"] or 1) <= D.STABILITY_STD for m in ("cav", "cce")}
            reg_out["decision"] = D.amazon_decision({
                "fedavg_slr": [v for k, v in slr.items() if k != "server_lr1"], "fedexp": list(exp.values()),
                "cav": get("cav")["V"], "cce": get("cce")["V"], "fedyogi_lr3": get("fedyogi", "server_lr3")["V"]}, stable)
        else:
            reg_out["specificity"] = D.yelp_specificity(get("cav")["V"], get("cce")["V"])
        out["regimes"][reg] = reg_out
    dest = P.stage_dir(bench, "explore").parent / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "decision.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v.get("decision", v.get("specificity")) | {"integrity_ok": v["integrity"]["ok"]}
                      for k, v in out["regimes"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
