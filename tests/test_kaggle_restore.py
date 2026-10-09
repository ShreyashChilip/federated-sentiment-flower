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
    k.OUT_EXPLORATORY = tmp_path / "out_x"
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


def make_job3a_output(root: Path, completed: int = 3) -> None:
    for i in range(completed):
        write(root / "results_benchmark" / f"tune_algorithms/yelp_iid/m{i}/seed7/COMPLETE", "{}")
    write(root / "results_benchmark" / "tune_base/amazon_vg/selected.json", '{"lr": 0.3}')
    write(root / "partitions" / "p.json", "{}")


def test_dataset_nested_deeply_is_found(tmp_path):
    """Kaggle may mount a dataset as /kaggle/input/datasets/<owner>/<slug>/<zip name>/..."""
    k = load_kernel(tmp_path)
    inp = tmp_path / "input"
    make_job3a_output(inp / "datasets" / "owner" / "fedbench-results" / "fedbench_results_fedbench-job3")
    write(inp / "notebooks" / "owner" / "fedbench-job2" / "bundles" / "amazon2023_x" / "meta.json", '{"v": 1}')
    k.restore(inp, tmp_path / "unz")
    assert len(list((k.OUT_RESULTS / "tune_algorithms").glob("*/*/seed7/COMPLETE"))) == 3
    assert (k.OUT_RESULTS / "tune_base/amazon_vg/selected.json").exists()
    assert (k.CODE / "data_cache/features/amazon2023_x/meta.json").exists()


def test_result_archives_kept_as_zip_are_extracted_and_restored(tmp_path):
    import zipfile

    k = load_kernel(tmp_path)
    staging = tmp_path / "staging"
    make_job3a_output(staging, completed=4)
    inp = tmp_path / "input" / "fedbench-results"
    inp.mkdir(parents=True)
    with zipfile.ZipFile(inp / "fedbench_results_fedbench-job3.zip", "w") as z:
        for f in staging.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(staging))
    k.restore(tmp_path / "input", tmp_path / "unz")
    assert len(list((k.OUT_RESULTS / "tune_algorithms").glob("*/*/seed7/COMPLETE"))) == 4


def test_preflight_refuses_to_redo_work_or_rebuild_bundles(tmp_path):
    k = load_kernel(tmp_path)
    make_job3a_output(tmp_path / "input" / "job3a", completed=2)
    k.restore(tmp_path / "input", tmp_path / "unz")
    k.JOB.update(expect_complete={"tune_algorithms": 33}, require_bundles=[])
    with pytest.raises(SystemExit, match="(?s)PREFLIGHT FAILED.*2 completed runs restored, expected at least 33"):
        k.preflight()
    k.JOB.update(expect_complete={"tune_algorithms": 2}, require_bundles=["amazon2023_x"])
    with pytest.raises(SystemExit, match="bundle amazon2023_x not found"):
        k.preflight()
    write(k.CODE / "data_cache/features/amazon2023_x/meta.json", "{}")
    k.preflight()                                           # everything expected is present


def test_code_commit_pin_requires_identical_experiment_code():
    spec = importlib.util.spec_from_file_location("kaggle_job", ROOT / "tools" / "kaggle_job.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    head = tool.git("rev-parse", "HEAD")
    # 10f519b predates the partition-loading fix in src/: must be refused
    with pytest.raises(SystemExit, match="experiment code changed"):
        tool.check_code_commit("10f519b", head)
    # the experiment code of 3415496 (Job 3a) is unchanged at HEAD
    assert tool.check_code_commit("3415496", head).startswith("3415496")


def natural_summary(seed: int, sha: str = "a" * 64, clients: int = 3) -> str:
    return json.dumps({"name": "amazon_natural", "params": {"scheme": "natural", "num_clients": None},
                       "experiment_seed": seed, "partition_seed": 1000 + seed, "sha256": sha,
                       "num_clients": clients, "clients": [{"client_id": i} for i in range(clients)]})


def test_natural_partition_created_under_different_seeds_is_not_a_conflict(tmp_path):
    """Job 3b restore (2026-10-07): Job 2 created the Amazon partition with seed 42,
    Job 3a re-created the identical partition with seed 7; only the recorded seed differs."""
    k = load_kernel(tmp_path)
    inp = tmp_path / "input"
    write(inp / "job2" / "partitions" / "amazon_natural.json", natural_summary(42))
    write(inp / "job3a" / "partitions" / "amazon_natural.json", natural_summary(7))
    write(inp / "job2" / "partitions" / "amazon_natural.npz", "same bytes")
    write(inp / "job3a" / "partitions" / "amazon_natural.npz", "same bytes")
    k.restore(inp, tmp_path / "unz")
    kept = json.loads((k.OUT_PARTITIONS / "amazon_natural.json").read_text())
    assert kept["experiment_seed"] == 42                                   # first one kept
    assert (k.OUT_PARTITIONS / "amazon_natural.json.from-job3a").exists()  # the other preserved


@pytest.mark.parametrize("other", [natural_summary(7, sha="b" * 64), natural_summary(7, clients=4)])
def test_partition_summaries_with_real_differences_still_conflict(tmp_path, other):
    k = load_kernel(tmp_path)
    inp = tmp_path / "input"
    write(inp / "job2" / "partitions" / "amazon_natural.json", natural_summary(42))
    write(inp / "job3a" / "partitions" / "amazon_natural.json", other)
    with pytest.raises(SystemExit, match="conflicting"):
        k.restore(inp, tmp_path / "unz")


def test_exploratory_results_are_restored_separately_from_official_ones(tmp_path):
    k = load_kernel(tmp_path)
    inp = tmp_path / "input"
    write(inp / "a" / "results_benchmark" / "tune_base/amazon_vg/selected.json", '{"lr": 0.3}')
    write(inp / "b" / "results_exploratory" / "cce_v1/explore/amazon_vg/cav__reference/seed7/COMPLETE", "{}")
    k.restore(inp, tmp_path / "unz")
    assert (k.OUT_EXPLORATORY / "cce_v1/explore/amazon_vg/cav__reference/seed7/COMPLETE").exists()
    assert not (k.OUT_RESULTS / "cce_v1").exists()
    assert (k.OUT_RESULTS / "tune_base/amazon_vg/selected.json").exists()
