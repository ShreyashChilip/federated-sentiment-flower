"""Results schema, immutable run directories, pre-run guards and consistency checks.

Runs are produced from the SYNTHETIC review file; no number here is a result.
"""
import csv
import json

import pytest

from src.analysis import exp0, exp0_driver
from src.analysis.profile import profile_and_save
from src.runner import run_baseline
from src.utils.config import deep_merge, load_config
from src.utils.runtime import prepare_run
from src.utils.schema import REQUIRED_KEYS, collect_tables, run_record, validate_run
from tests.synthetic_reviews import write_reviews
from tests.test_natural_clients import make_cfg


@pytest.fixture(scope="module")
def review_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("amazon") / "Synthetic.jsonl"
    return path, write_reviews(path)


@pytest.fixture(scope="module")
def finished_run(review_file, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = make_cfg(review_file, tmp, **{"centralized.epochs": 1})
    cell_cfg = exp0.cell_config(cfg, "natural")
    run_dir = exp0.run_dir_for(cfg, "natural", "centralized", 42)
    run_baseline(cell_cfg, 42, "centralized", run_dir, create_partition=True)
    return cfg, cell_cfg, run_dir, tmp


def test_run_record_has_every_schema_field(finished_run):
    _, _, run_dir, _ = finished_run
    rec = run_record(run_dir)
    assert not [k for k in REQUIRED_KEYS if k not in rec]
    assert validate_run(run_dir) == []
    assert rec["experiment_id"] == "exp0b_user" and rec["algorithm"] == "centralized" and rec["model"] == "lr"
    assert rec["partition_type"] == "natural" and rec["client_definition"] == "user_id"
    assert json.loads(rec["client_filter_rule"])["min_reviews"] == 10
    assert rec["official_environment"] is False            # the laptop is never official
    assert len(rec["config_hash"]) == 12 and len(rec["vocabulary_sha256"]) == 64 and len(rec["run_id"]) == 32
    assert rec["seed"] == 42 and rec["test_macro_f1"] is not None and rec["cpu_time_s"] is not None


def test_metadata_records_realized_client_class_counts(finished_run):
    _, _, run_dir, _ = finished_run
    part = json.loads((run_dir / "run_metadata.json").read_text())["partition"]
    assert len(part["client_class_counts"]) == part["client_count"] == len(part["client_sizes"])
    assert all(sum(c) == n for c, n in zip(part["client_class_counts"], part["client_sizes"]))
    assert part["partition_seed"] is not None and len(part["sha256"]) == 64


def test_validation_reports_missing_information(finished_run, tmp_path):
    _, _, run_dir, _ = finished_run
    broken = tmp_path / "seed1"
    broken.mkdir()
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    meta["partition"]["client_class_counts"] = []
    meta["config_hash"] = None
    (broken / "run_metadata.json").write_text(json.dumps(meta))
    (broken / "final.json").write_text((run_dir / "final.json").read_text())
    problems = validate_run(broken)
    assert "field 'config_hash' is empty" in problems and "realized client class counts not saved" in problems
    assert validate_run(tmp_path / "nothing") == ["missing file run_metadata.json", "missing file final.json"]


def test_finished_runs_are_never_overwritten(finished_run):
    _, cell_cfg, run_dir, _ = finished_run
    before = (run_dir / "final.json").read_text()
    with pytest.raises(FileExistsError, match="never overwritten"):
        run_baseline(cell_cfg, 42, "centralized", run_dir)
    assert (run_dir / "final.json").read_text() == before


def test_unfinished_run_is_set_aside_not_overwritten(finished_run):
    cfg, cell_cfg, _, _ = finished_run
    run_dir = exp0.run_dir_for(cfg, "natural", "fedavg", 7)
    prepare_run(cell_cfg, 7, run_dir)                      # metadata written, run never finished
    first_id = json.loads((run_dir / "run_metadata.json").read_text())["run_id"]
    prepare_run(cell_cfg, 7, run_dir)
    kept = list(run_dir.parent.glob("seed7.incomplete-*"))
    assert len(kept) == 1 and json.loads((kept[0] / "run_metadata.json").read_text())["run_id"] == first_id
    assert json.loads((run_dir / "run_metadata.json").read_text())["run_id"] != first_id


def test_consistency_checks_pass_for_comparable_conditions_and_catch_differences(finished_run, tmp_path):
    cfg, cell_cfg, central_dir, _ = finished_run
    dirs = {"centralized": central_dir}
    for cond in exp0.FEDERATED:
        dirs[cond] = prepare_run(deep_merge(cell_cfg, {"fl": {"algorithm": cond}}), 42, tmp_path / cond / "seed42")
    checks = exp0.consistency_checks(dirs)
    names = {c["check"] for c in checks}
    for required in ("same vocabulary (feature space) in every condition", "same training split", "same evaluation test set",
                     "partition seed recorded", "realized client class counts saved", "alpha recorded", "client count recorded",
                     "seed recorded", "configuration hash recorded", "git commit recorded", "official/non-official status recorded",
                     "identical fl.weight_decay"):
        assert required in names
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]

    other = prepare_run(deep_merge(cell_cfg, {"fl": {"algorithm": "fedavg_la", "weight_decay": 0.01}}), 42, tmp_path / "wd" / "seed42")
    failed = {c["check"] for c in exp0.consistency_checks({**dirs, "fedavg_la": other}) if not c["passed"]}
    assert failed == {"identical fl.weight_decay"}

    meta_path = dirs["fedavg"] / "run_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["vocabulary_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    failed = {c["check"] for c in exp0.consistency_checks(dirs) if not c["passed"]}
    assert "same vocabulary (feature space) in every condition" in failed


def test_official_run_preconditions():
    cfg = load_config("exp0a.yaml")
    kaggle = {"platform_kind": "kaggle", "git": {"commit": "abc123", "dirty": False}}
    unready = deep_merge(cfg, {"protocol": {"frozen": False}, "data": {"revision": None},
                               "experiment": {"seeds": [1, 2, 3, 4, 5]}})
    problems = exp0.official_preconditions(unready, kaggle, unready["experiment"]["seeds"])
    assert any("protocol.frozen is false" in p for p in problems)
    assert any("requires 8" in p for p in problems)        # a 5-seed list is not accepted for the 8-seed protocol
    assert any("data.revision is not pinned" in p for p in problems)

    ready = deep_merge(cfg, {"protocol": {"frozen": True}, "data": {"revision": "deadbeef"},
                             "experiment": {"seeds": [1, 2, 3, 4, 5, 6, 7, 8]}})
    assert exp0.official_preconditions(ready, kaggle, ready["experiment"]["seeds"]) == []
    laptop = {"platform_kind": "local", "git": {"commit": None, "dirty": True}}
    assert set(exp0.official_preconditions(ready, laptop, ready["experiment"]["seeds"])) == {
        "not running on Kaggle", "no git commit", "git working tree has uncommitted changes"}


def test_full_mode_refuses_on_the_laptop_and_pilot_uses_a_non_official_seed():
    cfg = load_config("exp0a.yaml")
    with pytest.raises(SystemExit, match="Refusing to start the official run"):
        exp0_driver.resolve(cfg, "full", None)
    pilot_cfg, seeds = exp0_driver.resolve(cfg, "pilot", None)
    assert seeds == [cfg["pilot"]["seed"]] and seeds[0] not in cfg["experiment"]["seeds"]
    assert pilot_cfg["experiment"]["name"] == "exp0a_pilot" and pilot_cfg["fl"]["rounds"] == cfg["pilot"]["rounds"]
    with pytest.raises(ValueError, match="pilot seed"):
        exp0_driver.resolve(deep_merge(cfg, {"pilot": {"seed": 42}}), "pilot", None)


def test_experiment_0a_config_matches_the_frozen_design():
    cfg = load_config("exp0a.yaml")
    assert cfg["data"]["dataset"] == "yelp_polarity" and cfg["data"]["max_train"] is None and cfg["data"]["max_test"] is None
    assert set(cfg["cells"]) == {"iid", "label_skew"}
    assert cfg["cells"]["iid"] == {"scheme": "iid", "num_clients": 100}
    assert cfg["cells"]["label_skew"]["alpha"] == 0.1 and cfg["cells"]["label_skew"]["num_clients"] == 100
    assert exp0.CONDITIONS == ("centralized", "fedavg", "fedavg_la")
    assert cfg["diagnostic"]["min_document_frequency"] == 50
    iid = exp0.cell_config(cfg, "iid")["partition"]
    assert "alpha" not in iid                               # no Dirichlet parameter leaks into the IID control


def test_the_two_client_definitions_are_separate_configs():
    user, product = load_config("exp0b_user.yaml"), load_config("exp0b_product.yaml")
    assert user["data"]["client_definition"] == "user_id" and product["data"]["client_definition"] == "parent_asin"
    assert user["experiment"]["name"] != product["experiment"]["name"]
    for cfg in (user, product):
        assert cfg["data"]["dataset"] == "amazon2023" and cfg["data"]["revision"]
        assert cfg["partition"]["scheme"] == "natural" and set(cfg["cells"]) == {"natural"}


def test_profile_is_written_once_and_is_machine_readable(review_file, tmp_path):
    cfg = make_cfg(review_file, tmp_path, "product")
    out, report = profile_and_save(cfg, tmp_path / "profile")
    saved = json.loads((out / "profile.json").read_text())
    for key in ("num_raw_clients", "client_size", "global_class_distribution", "retention_by_min_reviews",
                "per_client_majority_class_share", "official_environment", "config_hash", "source"):
        assert key in saved
    assert saved["client_definition"] == "parent_asin" and saved["official_environment"] is False
    assert {"median", "min", "max", "p10", "p90", "p99"} <= set(saved["client_size"])
    with open(out / "clients.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == saved["num_raw_clients"]
    assert sum(int(r["num_reviews"]) for r in rows) == saved["num_records"]
    assert all(int(r["num_reviews"]) == sum(int(r[f"count_{s}_star"]) for s in range(1, 6)) for r in rows)
    with pytest.raises(FileExistsError):
        profile_and_save(cfg, tmp_path / "profile")


def test_collect_tables(finished_run):
    cfg, _, run_dir, _ = finished_run
    out = collect_tables(exp0.experiment_dir(cfg))
    assert out["runs"] >= 1 and str(run_dir) not in out["problems"]
    with open(exp0.experiment_dir(cfg) / "runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(r["algorithm"] == "centralized" and r["seed"] == "42" for r in rows)
