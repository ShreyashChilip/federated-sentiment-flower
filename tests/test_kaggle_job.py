"""The generated Kaggle script must be valid Python with the exact JOB values."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("kaggle_job", ROOT / "tools" / "kaggle_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_script_uses_python_literals_and_round_trips():
    job = {"repo": "https://example.invalid/r.git", "commit": "a" * 40, "stages": ["pilot", "tune_base"],
           "max_hours": 10.0, "diagnostics": [["amazon_vg", 42], ["yelp_iid", 42]], "analyze": ["pilot"],
           "keep_bundles": True, "extra_args": [], "slug": "t", "flag_false": False, "nothing": None}
    src = load_tool().render_script(job)
    compile(src, "kaggle_benchmark.py", "exec")
    block = src[src.index("JOB = "):src.index("\nT0 = ")]
    for token in ("true", "false", "null"):
        assert not any(tok == token for tok in block.replace(",", " ").replace(":", " ").split())
    tree = ast.parse(src)
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "JOB")
    assert ast.literal_eval(value) == job


def test_generated_script_carries_the_benchmark_config():
    job = {"repo": "r", "commit": "a" * 40, "stages": ["explore"], "max_hours": 10.0, "diagnostics": [], "analyze": [],
           "keep_bundles": True, "extra_args": [], "verify": ["explore"], "slug": "x", "bench": "exploratory/cce_v1.yaml",
           "expect_complete": {}, "require_bundles": ["b"], "wrapper_commit": "a" * 40}
    src = load_tool().render_script(job)
    tree = ast.parse(src)
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "JOB")
    assert ast.literal_eval(value)["bench"] == "exploratory/cce_v1.yaml"
    assert '"--bench", JOB.get("bench", "benchmark.yaml")' in src
