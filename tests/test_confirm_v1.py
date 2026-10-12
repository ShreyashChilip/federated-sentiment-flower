"""EXPLORATORY new-seed confirmation (docs/CONFIRM_V1_PROTOCOL.md): planning and isolation."""
import hashlib
from pathlib import Path

from src.benchmark import plan as P
from src.utils.config import PROJECT_ROOT

SEL = PROJECT_ROOT / "configs" / "exploratory" / "confirm_v1_selections"


def test_plan_is_ten_runs_on_new_seeds_in_the_exploratory_folder():
    b = P.load_benchmark("exploratory/confirm_v1.yaml")
    specs = [s for st in ("confirm_yelp_iid", "confirm_yelp_dir", "confirm_amazon") for s in P.plan_stage(b, st)]
    assert len(specs) == 10
    assert {s["seed"] for s in specs} == {789, 1001}
    assert not {s["seed"] for s in specs} & {7, 42, 123, 456}
    assert all("exploratory" in s["run_dir"] and "confirm_v1" in s["run_dir"] for s in specs)
    assert {(s["regime"], s["method"]) for s in specs} == {
        ("yelp_iid", "fedadam"), ("yelp_iid", "fedadagrad"), ("yelp_dir01", "fedadagrad"),
        ("amazon_vg", "fedadam"), ("amazon_vg", "fedyogi")}


def test_selection_copies_match_their_recorded_hashes():
    for line in (SEL / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(" *")
        assert hashlib.sha256((SEL / name).read_bytes()).hexdigest() == digest, name


def test_frozen_benchmark_does_not_use_selection_dir():
    assert "selection_dir" not in P.load_benchmark("benchmark.yaml")
