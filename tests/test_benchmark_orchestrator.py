"""Orchestrator: resume, failure recording, validation, selection immutability (SYNTHETIC data)."""
import json
from pathlib import Path

import pytest

from src.benchmark import orchestrator as O
from src.benchmark import plan as P
from src.utils.config import deep_merge


@pytest.fixture()
def bench(tmp_path):
    from tests.synthetic_reviews import write_reviews

    Path("results/_smoke_bench_data").mkdir(parents=True, exist_ok=True)
    if not Path("results/_smoke_bench_data/Synthetic.jsonl").exists():
        write_reviews(Path("results/_smoke_bench_data/Synthetic.jsonl"), num_users=120, num_items=20)
    b = P.load_benchmark("smoke_bench/benchmark.yaml")
    return deep_merge(b, {"results_dir": str(tmp_path / "bench")})


def test_pilot_runs_resume_and_manifest(bench):
    s1 = O.run_stage(bench, "pilot", only="amazon_vg/fedavg/")
    assert s1["launched"] == 1 and s1["complete"] == 1
    s2 = O.run_stage(bench, "pilot", only="amazon_vg/fedavg/")
    assert s2["launched"] == 0 and s2["complete"] == 1          # finished runs are never re-run
    manifest = json.loads((P.stage_dir(bench, "pilot") / "manifest.json").read_text())
    assert manifest["counts"]["complete"] == 1 and manifest["runs"][0]["partition_sha256"]


def test_failed_run_is_recorded_not_retried(bench, monkeypatch):
    spec = [s for s in P.plan_stage(bench, "pilot") if "amazon_vg/scaffold/" in s["key"]][0]
    bad = deep_merge(spec, {"cfg": {"fl": {"algorithm": "no_such_algorithm"}}})
    attempt = O.launch(bad)
    assert attempt["outcome"] == "error" and attempt["returncode"] != 0 and "log_tail" in attempt
    assert O.run_state(bad) == "failed" and not O.is_complete(bad)
    summary = O.run_stage(bench, "pilot", only="amazon_vg/scaffold/")
    assert summary["skipped_failed"] == 1 and summary["launched"] == 0


def test_validation_detects_configuration_drift(bench):
    O.run_stage(bench, "pilot", only="amazon_vg/fedavg/")
    spec = [s for s in P.plan_stage(bench, "pilot") if "amazon_vg/fedavg/" in s["key"]][0]
    assert O.validate_run(spec) == []
    drifted = deep_merge(spec, {"cfg": {"fl": {"lr": 0.123}}})
    assert any("drift" in p for p in O.validate_run(drifted))


def test_official_stages_refuse_on_the_laptop(bench):
    with pytest.raises(SystemExit, match="not running on Kaggle"):
        O.run_stage(bench, "screening")


def test_failure_classification():
    assert O.classify_failure(-9, "") == "killed_oom_suspected"
    assert O.classify_failure(1, "numpy.core._exceptions: Unable to allocate 3 GiB") == "oom"
    assert O.classify_failure(3, "") == "validation_failed"


def test_selection_is_never_overwritten(tmp_path):
    path = tmp_path / "selected.json"
    O._write_selection(path, {"lr": 0.1, "selected_utc": "a"})
    O._write_selection(path, {"lr": 0.1, "selected_utc": "b"})       # identical content: fine
    with pytest.raises(RuntimeError, match="never overwritten"):
        O._write_selection(path, {"lr": 0.3, "selected_utc": "c"})


def test_pilot_and_base_tuning_specs_do_not_depend_on_later_selections(bench):
    """A pilot planned before tuning must equal the same pilot planned after it,
    otherwise completed pilot runs would fail validation (config drift)."""
    before = {s["key"]: s["config_hash"] for st in ("pilot", "tune_base") for s in P.plan_stage(bench, st)}
    O._write_selection(P.selection_path(bench, "base", "amazon_vg"),
                       {"regime": "amazon_vg", "lr": 0.3, "weight_decay": 1e-5, "selected_utc": "x"})
    after = {s["key"]: s["config_hash"] for st in ("pilot", "tune_base") for s in P.plan_stage(bench, st)}
    assert before == after
    tuned = P.regime_config(bench, "amazon_vg")                     # evaluation stages DO use it
    assert tuned["fl"]["lr"] == 0.3 and tuned["benchmark"]["base_tuning"]["source"] == "tune_base"


def test_verify_detects_partition_mismatch_and_passes_clean_runs(bench):
    O.run_stage(bench, "pilot", only="amazon_vg/fedavg/")
    O.run_stage(bench, "pilot", only="amazon_vg/fedadam/")
    assert O.verify_stage(bench, "pilot")["ok"]
    spec = [s for s in P.plan_stage(bench, "pilot") if "amazon_vg/fedadam/" in s["key"]][0]
    meta_path = Path(spec["run_dir"]) / "run_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["partition"]["sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    report = O.verify_stage(bench, "pilot")
    assert not report["ok"] and any("different partitions" in p for p in report["problems"])


def test_tuning_seed_must_not_be_an_evaluation_seed(bench):
    assert bench["tuning_seed"] not in P.evaluation_seeds(bench)
    assert len(P.plan_stage(bench, "tune_base")) > 0                    # seed 7 is accepted
    for clash in (42, 271828):                                          # official / screening seeds
        bad = deep_merge(bench, {"tuning_seed": clash})
        for stage in ("tune_base", "tune_algorithms"):
            with pytest.raises(ValueError, match="evaluation seed"):
                P.plan_stage(bad, stage)


def test_real_benchmark_config_keeps_seed_7_and_the_official_seeds():
    b = P.load_benchmark("benchmark.yaml")
    assert b["tuning_seed"] == 7
    assert P.evaluation_seeds(b) == {42, 123, 456, 789, 1001, 2024, 31415, 271828}
    assert b["stages"]["screening"]["seeds"] == [42, 123, 456]


def test_verify_reports_an_unplannable_stage_without_a_traceback(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(f"inherits: {(root / 'configs' / 'smoke_bench' / 'benchmark.yaml').as_posix()}\n"
                   f"benchmark:\n  results_dir: {(tmp_path / 'r').as_posix()}\n")
    out = subprocess.run([sys.executable, str(root / "experiments" / "run_benchmark.py"), "verify", "--bench", str(cfg),
                          "--stages", "tune_algorithms", "screening"], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert "Traceback" not in out.stdout + out.stderr
    assert out.stdout.count("NOT PLANNABLE YET") == 2
