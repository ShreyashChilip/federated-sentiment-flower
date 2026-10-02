# Environment

There are two environments. Only the Kaggle one may produce reported results.
Every run writes the environment it actually ran in to `run_metadata.json`
(`src/utils/env.py`), and summaries flag any run not made on Kaggle.

## 1. Official environment: Kaggle GPU notebook

**Not recorded yet.** Phase 0K (`notebooks/kaggle_runner.py`, stage `bringup`)
writes `results/kaggle_environment.json`; copy its values into this table and
commit. Nothing below may be filled in from memory or documentation.

| Item | Value |
|---|---|
| Kaggle image / "pin to original environment" date | to record |
| OS | to record |
| Python | to record |
| CPU, logical cores | to record |
| RAM | to record |
| GPU model, count, VRAM | to record |
| CUDA | to record |
| PyTorch (image build) | to record |
| transformers, peft | to record when Phase 4 starts |
| Declared CPU watts / GPU watts for the energy proxy | to decide and record, with source |
| Dataset revision (Hugging Face commit of `fancyzhx/yelp_polarity`) | to record and pin in `configs/base.yaml` |

Packages installed on top of the image are pinned in `requirements-kaggle.txt`.
PyTorch is taken from the image (CUDA build) and not reinstalled. If the
pinned NumPy conflicts with the image's PyTorch build, resolve it at bring-up,
update the pin and note it in `CHANGELOG.md`.

In the notebook settings choose "Pin to original environment" so that the
image does not change between sessions.

## 2. Development environment: laptop (debugging only)

Recorded 2026-10-02 by `src/utils/env.py`.

| Item | Value |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| Python | 3.11.9 (virtualenv at `C:\Users\Admin\.venvs\flsent`) |
| CPU | AMD64 Family 25 Model 80 (6 physical / 12 logical cores); the old manuscript names it Ryzen 5 5600H |
| RAM | 23.3 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB (not used: CPU-only PyTorch build) |
| CUDA | none in this virtualenv |
| flwr | 1.39.0 |
| ray | 2.55.1 |
| torch | 2.14.1+cpu |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.6 |
| datasets | 5.0.1 |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |

Full `pip freeze`: `docs/laptop_pip_freeze.txt`. Install with
`pip install -r requirements.txt`.

## 3. Deployment type (reviewer 1.10)

Single-machine simulation with the Flower Simulation Runtime (Ray backend).
Clients are virtual; no physical devices, VMs or network emulation are used,
and no latency or bandwidth figures are claimed. Communication is reported as
serialized payload bytes.

## 4. Known environment notes

* Flower 1.39.0 marks `flwr.simulation.run_simulation` as deprecated in favour
  of the `flwr run` CLI. It works in the pinned version; the pin must not be
  raised without porting the entry point (`src/runner.py`).
* Flower warns that Ray on Windows is experimental. On the laptop this shows
  up as harmless access-violation traces from `raylet` at shutdown. Kaggle is
  Linux.
* The laptop has a folder `D:\kaggle\working`. Kaggle detection therefore
  requires Linux and the `KAGGLE_KERNEL_RUN_TYPE` variable (regression-tested).
