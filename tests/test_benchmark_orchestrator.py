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
