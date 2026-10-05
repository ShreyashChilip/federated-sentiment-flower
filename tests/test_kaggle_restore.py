"""Restoring earlier Kaggle outputs must never lose or silently replace artifacts."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_kernel(tmp_path):
    spec = importlib.util.spec_from_file_location("kernel", ROOT / "notebooks" / "kaggle_benchmark.py")
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)             # main() is not run on import
    k.OUT_RESULTS, k.OUT_PARTITIONS, k.CODE = tmp_path / "out_r", tmp_path / "out_p", tmp_path / "code"
    return k


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def status(*attempts):
    return json.dumps({"attempts": [{"attempt": 1, "start_utc": s, "outcome": o, "log": "x/attempt1.log"} for s, o in attempts]})


def two_sessions(tmp_path):
    inp = tmp_path / "input"
    key = "pilot/_status/pilot__amazon_vg__fedavg__seed0.json"
    # job 1: killed attempt; separate OOM-check session (job 1 not attached): complete attempt
    write(inp / "job1" / "results_benchmark" / key, status(("2026-10-05T10:00:00+00:00", "killed_oom_suspected")))
    write(inp / "job1" / "results_benchmark" / "pilot/_logs/a.attempt1.log", "Killed")
    write(inp / "job1" / "results_benchmark" / "pilot/yelp_iid/fedavg/seed0/COMPLETE", "{}")
    write(inp / "oom" / "results_benchmark" / key, status(("2026-10-06T08:00:00+00:00", "complete")))
    write(inp / "oom" / "results_benchmark" / "pilot/_logs/a.attempt1.log", "complete")
    write(inp / "oom" / "results_benchmark" / "pilot/amazon_vg/fedavg/seed0/COMPLETE", "{}")
    write(inp / "oom" / "bundles" / "amazon2023_x" / "meta.json", '{"v": 1}')
    write(inp / "job1" / "bundles" / "amazon2023_x" / "meta.json", '{"v": 1}')
    return inp, key


def test_two_sessions_merge_without_losing_attempts_or_logs(tmp_path):
    k = load_kernel(tmp_path)
    inp, key = two_sessions(tmp_path)
    k.restore(inp)
    merged = json.loads((k.OUT_RESULTS / key).read_text())
    assert [a["outcome"] for a in merged["attempts"]] == ["killed_oom_suspected", "complete"]
    logs = sorted(p.name for p in (k.OUT_RESULTS / "pilot/_logs").iterdir())
    assert len(logs) == 2 and {(k.OUT_RESULTS / "pilot/_logs" / n).read_text() for n in logs} == {"Killed", "complete"}
    assert (k.OUT_RESULTS / "pilot/yelp_iid/fedavg/seed0/COMPLETE").exists()
    assert (k.OUT_RESULTS / "pilot/amazon_vg/fedavg/seed0/COMPLETE").exists()
    assert (k.CODE / "data_cache/features/amazon2023_x/meta.json").exists()


def test_conflicting_run_artifact_or_bundle_aborts(tmp_path):
    k = load_kernel(tmp_path)
    inp, _ = two_sessions(tmp_path)
    write(inp / "job1" / "results_benchmark" / "pilot/amazon_vg/fedavg/seed0/final.json", '{"a": 1}')
    write(inp / "oom" / "results_benchmark" / "pilot/amazon_vg/fedavg/seed0/final.json", '{"a": 2}')
    with pytest.raises(SystemExit, match="conflicting"):
        k.restore(inp)
    k2 = load_kernel(tmp_path / "second")
    inp2, _ = two_sessions(tmp_path / "second")
    write(inp2 / "oom" / "bundles" / "amazon2023_x" / "meta.json", '{"v": 2}')
    with pytest.raises(SystemExit, match="bundle amazon2023_x differs"):
        k2.restore(inp2)
