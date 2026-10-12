"""EXPLORATORY DistilBERT study (docs/EXTEND_V1_PROTOCOL.md): centralized and federated (FedAvg, FedProx)
fine-tuning at FULL SCALE on exactly the data of the TF-IDF+LR benchmark (fedbench_v1):

* Yelp IID and Yelp Dirichlet-0.1: all 560,000 official training reviews with the benchmark's role split
  (448,000 TRAIN), 100 clients from the benchmark's partition function and seeds; official test set held out.
  Roles and labels are verified against the official bundle when it is present.
* Amazon VG: all retained reviews of the products with >= 20 reviews (seen and unseen clients as in the
  benchmark); texts are read from the cleaned review store in the bundle's row order and VERIFIED against the
  official bundle's labels and client codes before anything is cached or trained.

Regime settings (rounds, client fraction, local epochs, batch size, client sampling, init and batch-order seeds)
are those of the benchmark regime configs. The local optimizer is AdamW (standard for transformers), fixed a
priori. Not part of fedbench_v1.

Phases (one GPU per worker; workers share the run list through atomic claim files; every run resumes from
round/step checkpoints; with --deadline-hours, workers checkpoint and stop cleanly before a session limit):
  prepare   load, verify and tokenize all data once (cached; later sessions need only the cache)
  pilot     one full-scale FedAvg round per regime with one validation and one held-out pass; projects run times
  tune      FedProx mu grid per regime, tuning seed, validation only
  select    pick mu per regime (validation macro-F1; ties -> smaller mu)
  evaluate  FedAvg and FedProx (selected mu) per regime and seed; centralized per dataset and seed
  all       tune -> select -> evaluate
Usage
  python experiments/distilbert_fl.py --phase prepare
  python experiments/distilbert_fl.py --phase pilot
  python experiments/distilbert_fl.py --phase all --gpus 2 --deadline-hours 11
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.exp0 import cell_config  # noqa: E402
from src.data.roles import CLIENT_TEST, TRAIN, VAL, assign_roles  # noqa: E402
from src.metrics.classification import confusion_matrix, metrics_from_confusion  # noqa: E402
from src.partitioning.partition import make_partition  # noqa: E402
from src.partitioning.store import partition_hash  # noqa: E402
from src.strategies.server import select_clients  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeding import derive_seed  # noqa: E402

UNSEEN = 3   # src.data.natural.UNSEEN


class Paused(Exception):
    """Raised when the session deadline is reached; the run has been checkpointed."""


def past_deadline() -> bool:
    d = os.environ.get("FEDBERT_DEADLINE")
    return bool(d) and time.time() > float(d)


# ============================================================================ data
class TokenStore:
    """Token ids of many documents as one flat int32 array plus offsets (compact for millions of texts)."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat, self.offsets = flat, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.flat[self.offsets[i]:self.offsets[i + 1]]

    def lengths(self):
        return np.diff(self.offsets)

    def support_size(self, rows) -> int:
        """Number of distinct token ids over the given documents."""
        if len(rows) == 0:
            return 0
        seen = np.zeros(int(self.flat.max()) + 1 if len(self.flat) else 1, dtype=bool)
        for r in rows:
            seen[self[int(r)]] = True
        return int(seen.sum())

    @classmethod
    def build(cls, encode, texts, chunk=20000):
        parts, lens = [], []
        for i in range(0, len(texts), chunk):
            ids = encode(texts[i:i + chunk])
            parts.extend(np.asarray(x, dtype=np.int32) for x in ids)
            lens.extend(len(x) for x in ids)
            if (i // chunk) % 25 == 0:
                print(f"    tokenized {min(i + chunk, len(texts)):,}/{len(texts):,}", flush=True)
        flat = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        return cls(flat, np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))

    def save(self, path: Path):
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, flat=self.flat, offsets=self.offsets)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        return cls(z["flat"], z["offsets"])


def yelp_data(cfg: dict, bundle_path: Path | None = None) -> dict:
    """Full Yelp Polarity with the benchmark's role split (src/data/prepare.py::_build_standard)."""
    from src.data.datasets import load_text_dataset

    data_cfg = cfg["data"]
    ds = load_text_dataset(data_cfg)
    role_seed = derive_seed("roles", data_cfg["dataset"], data_cfg.get("role_seed", 0))
    roles = np.asarray(assign_roles(ds.train_labels, data_cfg["role_fractions"], role_seed))
    labels = np.asarray(ds.train_labels, np.int64)
    verified = False
    if bundle_path is not None and (bundle_path / "roles.npy").exists():
        if not (np.array_equal(roles, np.load(bundle_path / "roles.npy"))
                and np.array_equal(labels, np.load(bundle_path / "y_train.npy"))
                and np.array_equal(np.asarray(ds.test_labels), np.load(bundle_path / "y_test.npy"))):
            raise SystemExit("Yelp data/roles do NOT match the official bundle; refusing to run")
        verified = True
    return {"texts": list(ds.train_texts), "labels": labels, "roles": roles, "codes": None,
            "test_texts": list(ds.test_texts), "test_labels": np.asarray(ds.test_labels, np.int64), "test_rows": None,
            "num_classes": int(ds.num_classes), "meta": {**ds.meta, "verified_against_bundle": verified}}


def amazon_data(cfg: dict, bundle_path: Path) -> dict:
    """Amazon texts in the official bundle's row order, verified against its labels and client codes."""
    from src.data import amazon2023
    from src.data.prepare import cache_root

    data = cfg["data"]
    if data["client_filter"].get("max_reviews") or data["client_filter"].get("max_clients"):
        raise ValueError("row-order reconstruction assumes no per-client cap and no client sampling (as in fedbench_v1)")
    if not (bundle_path / "client_codes.npy").exists():
        raise SystemExit(f"official Amazon bundle not found at {bundle_path}; attach it before preparing")
    store_path, _ = amazon2023.open_review_store(data, cache_root(cfg))
    raw_keys, raw_counts = amazon2023.client_counts(store_path, data["client_definition"])
    kept = np.flatnonzero(raw_counts.sum(axis=1) >= int(data["client_filter"]["min_reviews"]))
    code_of = {raw_keys[i]: code for code, i in enumerate(kept)}
    col = data["client_definition"]
    if col not in amazon2023.CLIENT_DEFINITIONS:
        raise ValueError(col)
    texts, labels, codes = [], [], []
    with sqlite3.connect(store_path) as con:     # bundle rows are ordered by source row_id (_build_natural)
        for key, label, text in con.execute(f"SELECT {col}, label, text FROM reviews ORDER BY row_id"):
            code = code_of.get(key)
            if code is not None:
                texts.append(text)
                labels.append(label)
                codes.append(code)
    labels, codes = np.asarray(labels, np.int64), np.asarray(codes, np.int64)
    b_labels, b_codes = np.load(bundle_path / "y_train.npy"), np.load(bundle_path / "client_codes.npy")
    if not (np.array_equal(labels, b_labels) and np.array_equal(codes, b_codes)):
        raise SystemExit("Amazon texts do NOT match the official bundle (labels/client codes differ); refusing to run")
    roles = np.load(bundle_path / "roles.npy")
    meta = json.loads((bundle_path / "meta.json").read_text(encoding="utf-8"))
    return {"texts": texts, "labels": labels, "roles": roles, "codes": codes, "test_texts": None,
            "test_rows": np.flatnonzero(roles == UNSEEN), "test_labels": labels[roles == UNSEEN],
            "num_classes": int(meta.get("num_classes", amazon2023.NUM_CLASSES)),
            "meta": {"bundle": bundle_path.name, "verified_against_bundle": True,
                     **{k: meta.get(k) for k in ("revision", "source_sha256", "train_fingerprint")}}}


def make_clients(data: dict, part_cfg: dict, seed: int):
    """Clients exactly as src/utils/runtime.py::prepare_run builds them."""
    part = {k: v for k, v in part_cfg.items() if k != "dir"}
    groups = np.where(data["roles"] != UNSEEN, data["codes"], -1) if part["scheme"] == "natural" else None
    clients, _ = make_partition(data["labels"], part, derive_seed("partition", seed), groups)
    return clients


# ============================================================================ backends
class HFBackend:
    def __init__(self, name: str, num_labels: int, max_len: int, device: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.num_labels, self.max_len, self.device = name, num_labels, max_len, device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self._cls = AutoModelForSequenceClassification

    def encode(self, texts):
        return self.tokenizer(list(texts), truncation=True, max_length=self.max_len)["input_ids"]

    def new_model(self):
        model = self._cls.from_pretrained(self.name, num_labels=self.num_labels).to(self.device)
        self.model_commit = getattr(model.config, "_commit_hash", None)   # resolved hub revision, recorded per run
        return model

    def pad_id(self):
        return self.tokenizer.pad_token_id

    def forward(self, model, ids, mask):
        return model(input_ids=ids, attention_mask=mask).logits

    def word_embedding_shape(self, model):
        w = model.get_input_embeddings().weight
        return int(w.shape[0]), int(w.shape[1])


class TinyBackend:
    """Small stand-in used by the tests (no transformers needed). Same interface as HFBackend."""

    def __init__(self, num_labels: int, device: str = "cpu", buckets: int = 257, dim: int = 8, max_len: int = 64):
        self.num_labels, self.device, self.buckets, self.dim, self.max_len = num_labels, device, buckets, dim, max_len

    def encode(self, texts):
        import zlib

        return [[1 + zlib.crc32(w.encode()) % (self.buckets - 1) for w in t.lower().split()][: self.max_len] or [1] for t in texts]

    def new_model(self):
        b, d, k = self.buckets, self.dim, self.num_labels

        class Tiny(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = torch.nn.Embedding(b, d, padding_idx=0)
                s.norm = torch.nn.LayerNorm(d)
                s.out = torch.nn.Linear(d, k)

            def forward(s, ids, mask):
                e = s.emb(ids) * mask.unsqueeze(-1)
                return s.out(s.norm(e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)))

        return Tiny().to(self.device)

    def pad_id(self):
        return 0

    def forward(self, model, ids, mask):
        return model(ids, mask)

    def word_embedding_shape(self, model):
        return self.buckets, self.dim


# ============================================================================ training primitives
def collate(tokens: TokenStore, rows, pad_id, device):
    seqs = [tokens[int(r)] for r in rows]
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def float_keys(model):
    return [k for k, v in model.state_dict().items() if torch.is_floating_point(v)]


def get_state(model, keys):
    sd = model.state_dict()
    return {k: sd[k].detach().clone().float() for k in keys}


def set_state(model, state):
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            sd[k].copy_(v.to(sd[k].dtype))


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim == 1 or "LayerNorm" in n or "layer_norm" in n or n.endswith(".bias")) else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def train_step(backend, model, opt, scaler, tokens, labels, b, mu=0.0, anchor=None, named=None):
    amp = backend.device.startswith("cuda")
    ids, mask = collate(tokens, b, backend.pad_id(), backend.device)
    y = torch.as_tensor(labels[b], device=backend.device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss = torch.nn.functional.cross_entropy(backend.forward(model, ids, mask).float(), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    if mu > 0 and anchor is not None:          # FedProx: gradient of (mu/2)||w - w_global||^2
        with torch.no_grad():
            for n, p in named.items():
                if p.grad is not None and n in anchor:
                    p.grad.add_((p.detach().float() - anchor[n]).to(p.grad.dtype), alpha=mu)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach())


def local_train(backend, model, tokens, labels, rows, *, epochs, batch_size, lr, weight_decay, seed, mu=0.0, anchor=None):
    """One client's local training: fresh AdamW, mini-batches in a seeded order, optional FedProx term."""
    model.train()
    opt = make_optimizer(model, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=backend.device.startswith("cuda"))
    named = dict(model.named_parameters())
    steps, losses = 0, []
    rows = np.asarray(rows)
    for ep in range(int(math.ceil(epochs))):
        order = np.random.default_rng(seed + ep).permutation(len(rows))
        for i in range(0, len(rows), batch_size):
            losses.append(train_step(backend, model, opt, scaler, tokens, labels, rows[order[i:i + batch_size]], mu, anchor, named))
            steps += 1
    return steps, float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(backend, model, tokens, rows, batch_size=256):
    model.eval()
    amp = backend.device.startswith("cuda")
    rows = np.asarray(rows)
    order = np.argsort(tokens.lengths()[rows], kind="stable")       # length-sorted for speed; outputs restored
    out = np.empty(len(rows), dtype=np.int64)
    for i in range(0, len(rows), batch_size):
        idx = order[i:i + batch_size]
        ids, mask = collate(tokens, rows[idx], backend.pad_id(), backend.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out[idx] = backend.forward(model, ids, mask).float().argmax(1).cpu().numpy()
    return out


def scores(y_true, y_pred, k) -> dict:
    m = metrics_from_confusion(confusion_matrix(y_true, y_pred, k))
    return {key: float(m[key]) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1") if key in m} | {"n": int(len(y_true))}


def client_dispersion(labels, pred_of_row: np.ndarray, clients, roles, k, min_n=5):
    """Per-client macro-F1 on each client's CLIENT_TEST rows (clients with >= min_n such rows)."""
    per = []
    for cid, rows in enumerate(clients):
        r = rows[roles[rows] == CLIENT_TEST]
        if len(r) >= min_n:
            per.append({"client": cid, "n": int(len(r)), "macro_f1": scores(labels[r], pred_of_row[r], k)["macro_f1"]})
    v = np.array([p["macro_f1"] for p in per])
    summary = {"clients": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
               "median": float(np.median(v)), "worst": float(v.min())} if len(v) else {}
    return per, summary


def comm_accounting(model, backend) -> dict:
    total = sum(v.numel() for v in model.state_dict().values() if torch.is_floating_point(v))
    vocab, hidden = backend.word_embedding_shape(model)
    return {"float_parameters": int(total), "dense_message_bytes": int(4 * total), "word_embedding_rows": vocab, "hidden": hidden}


def support_message_bytes(acc: dict, k: int) -> int:
    """Derived payload of a message carrying all non-embedding parameters densely and only the k word-embedding
    rows of the client's token support, plus the cheaper of int32 indices or a vocabulary bitmask."""
    dense_rest = acc["dense_message_bytes"] - 4 * acc["word_embedding_rows"] * acc["hidden"]
    return int(dense_rest + 4 * acc["hidden"] * k + min(4 * k, math.ceil(acc["word_embedding_rows"] / 8)))


# ============================================================================ run bookkeeping
def environment() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    env = {"python": platform.python_version(), "torch": torch.__version__, "git_commit": git("rev-parse", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = None
    return env


def claim(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "COMPLETE").exists():
        return False
    try:
        fd = os.open(run_dir / "CLAIM", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def save_checkpoint(path: Path, obj: dict):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def finish(run_dir: Path, final: dict, history: list, meta: dict, per_client=None):
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    if per_client is not None:
        (run_dir / "clients.json").write_text(json.dumps(per_client))
    (run_dir / "final.json").write_text(json.dumps(final, indent=1, default=str))
    (run_dir / "checkpoint.pt").unlink(missing_ok=True)
    (run_dir / "COMPLETE").write_text("{}")
    (run_dir / "CLAIM").unlink(missing_ok=True)


def held_out_predictions(backend, model, ctx, dataset):
    data = ctx["data"][dataset]
    if data["test_rows"] is None:            # Yelp: official test split
        return predict(backend, model, ctx["test_tokens"][dataset], np.arange(len(data["test_labels"])))
    return predict(backend, model, ctx["tokens"][dataset], data["test_rows"])   # Amazon: unseen clients


# ============================================================================ runs
def federated_run(ctx, regime: str, seed: int, method: str, mu: float, run_dir: Path, *, eval_test=True, max_rounds=None):
    backend, study, R = ctx["backend"], ctx["study"], ctx["regimes"][regime]
    ds = R["dataset"]
    data, tokens = ctx["data"][ds], ctx["tokens"][ds]
    fl = R["fl"]
    rounds = int(max_rounds or fl["rounds"])
    clients = make_clients(data, R["partition"], seed)
    roles, y = data["roles"], data["labels"]
    train_rows = [rows[roles[rows] == TRAIN] for rows in clients]
    eligible = [c for c, r in enumerate(train_rows) if len(r)]
    val_rows = np.flatnonzero(roles == VAL)
    torch.manual_seed(derive_seed("init", seed))
    model = backend.new_model()
    keys = float_keys(model)
    glob = get_state(model, keys)
    acc_info = comm_accounting(model, backend)
    history, start, t0 = [], 1, time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    ck = run_dir / "checkpoint.pt"
    if ck.exists():   # resume (client sampling and batch orders are deterministic per round)
        saved = torch.load(ck, map_location=backend.device, weights_only=False)
        glob, history, start = {k: v.to(backend.device) for k, v in saved["glob"].items()}, saved["history"], saved["round"] + 1
        print(f"    resumed {run_dir} at round {start}", flush=True)
    support = {}
    cum_d = history[-1]["cum_dense_bytes"] if history else 0
    cum_s = history[-1]["cum_support_bytes"] if history else 0
    for rnd in range(start, rounds + 1):
        if past_deadline():
            raise Paused(str(run_dir))
        tr = time.time()
        sel = select_clients(eligible, float(fl["fraction_fit"]), int(fl.get("min_fit_clients", 1)), seed, rnd)
        total = float(sum(len(train_rows[c]) for c in sel))
        agg = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in glob.items()}
        steps, losses = 0, []
        for c in sel:
            set_state(model, glob)
            s, loss = local_train(backend, model, tokens, y, train_rows[c], epochs=float(fl["local_epochs"]),
                                  batch_size=int(fl["batch_size"]), lr=float(study["lr"]), weight_decay=float(study["weight_decay"]),
                                  seed=derive_seed("batch", seed, c, rnd), mu=mu if method == "fedprox" else 0.0, anchor=glob)
            w = len(train_rows[c]) / total
            sd = model.state_dict()
            for k in keys:                       # streaming sample-size-weighted FedAvg aggregation (float64)
                agg[k] += sd[k].detach().double() * w
            steps += s
            losses.append(loss)
            if c not in support:
                support[c] = tokens.support_size(train_rows[c])
        glob = {k: v.float() for k, v in agg.items()}
        del agg
        set_state(model, glob)
        cum_d += len(sel) * 2 * acc_info["dense_message_bytes"]
        cum_s += sum(2 * support_message_bytes(acc_info, support[c]) for c in sel)
        row = {"round": rnd, "selected": len(sel), "client_steps": steps, "train_loss": float(np.mean(losses)),
               "train_wall_s": time.time() - tr, "cum_dense_bytes": cum_d, "cum_support_bytes": cum_s}
        last = rnd == rounds
        if last or rnd % int(study.get("eval_val_every", 1)) == 0:
            te = time.time()
            row.update({f"val_{k}": v for k, v in scores(y[val_rows], predict(backend, model, tokens, val_rows), data["num_classes"]).items()})
            row["val_wall_s"] = time.time() - te
        if eval_test and (rnd in set(study.get("eval_test_rounds", [])) or last):
            te = time.time()
            row.update({f"held_{k}": v for k, v in scores(data["test_labels"], held_out_predictions(backend, model, ctx, ds), data["num_classes"]).items()})
            row["held_wall_s"] = time.time() - te
        row["round_wall_s"] = time.time() - tr
        history.append(row)
        print(f"    {regime} {method} s{seed} r{rnd}: val_f1={row.get('val_macro_f1', float('nan')):.4f} "
              f"held_f1={row.get('held_macro_f1', float('nan')):.4f} ({row['round_wall_s']:.0f}s)", flush=True)
        if not last and (rnd % int(study.get("checkpoint_every", 1)) == 0 or past_deadline()):
            save_checkpoint(ck, {"glob": {k: v.cpu() for k, v in glob.items()}, "history": history, "round": rnd})
    te = time.time()
    ct = np.flatnonzero(roles == CLIENT_TEST)
    pred = np.full(len(y), -1, dtype=np.int64)
    pred[ct] = predict(backend, model, tokens, ct)
    per, disp = client_dispersion(y, pred, clients, roles, data["num_classes"])
    pick = lambda pre: {k[len(pre):]: v for k, v in history[-1].items() if k.startswith(pre) and not k.endswith("wall_s")}
    final = {"kind": "federated", "regime": regime, "method": method, "mu": mu, "seed": seed, "rounds": rounds,
             "val": pick("val_"), "held_out": pick("held_"), "client_dispersion": disp,
             "client_test": scores(y[ct], pred[ct], data["num_classes"]),
             "communication": {**acc_info, "training_dense_bytes": cum_d, "training_support_bytes_derived": cum_s,
                               "note": "derived float32 payload estimates, not measured traffic; support-only applies "
                                       "to word-embedding rows only (all other parameters dense)"},
             "resources": {"session_wall_s": time.time() - t0, "client_test_wall_s": time.time() - te,
                           "mean_train_wall_s_per_round": float(np.mean([h["train_wall_s"] for h in history])),
                           "note": "simulation wall time on one GPU, not deployment latency"}}
    finish(run_dir, final, history, {"study": study, "regime": {k: v for k, v in R.items() if k != "cfg"}, "seed": seed,
                                     "method": method, "mu": mu, "environment": environment(),
                                     "partition_sha256": partition_hash(clients), "data_meta": data["meta"], "model_commit": getattr(backend, "model_commit", None)}, per)
    return final


def centralized_run(ctx, dataset: str, seed: int, run_dir: Path, part_for_dispersion):
    backend, study = ctx["backend"], ctx["study"]
    data, tokens = ctx["data"][dataset], ctx["tokens"][dataset]
    roles, y = data["roles"], data["labels"]
    train_rows, val_rows, ct = (np.flatnonzero(roles == r) for r in (TRAIN, VAL, CLIENT_TEST))
    torch.manual_seed(derive_seed("init", seed))
    model = backend.new_model()
    opt = make_optimizer(model, float(study["lr"]), float(study["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=backend.device.startswith("cuda"))
    bs, epochs = int(study.get("centralized_batch_size", 32)), int(study["centralized_epochs"])
    every = int(study.get("centralized_checkpoint_steps", 5000))
    history, start_ep, start_step, losses, t0 = [], 1, 0, [], time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    ck = run_dir / "checkpoint.pt"

    def snapshot(ep, step):
        save_checkpoint(ck, {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                             "history": history, "epoch": ep, "step": step, "losses": losses})

    if ck.exists():   # resume mid-epoch (the batch order of an epoch is deterministic)
        saved = torch.load(ck, map_location=backend.device, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        scaler.load_state_dict(saved["scaler"])
        history, start_ep, start_step, losses = saved["history"], saved["epoch"], saved["step"], saved["losses"]
        print(f"    resumed {run_dir} at epoch {start_ep} step {start_step}", flush=True)
    for ep in range(start_ep, epochs + 1):
        te = time.time()
        model.train()
        order = np.random.default_rng(derive_seed("batch", seed, "central", ep)).permutation(train_rows)
        for step in range(start_step if ep == start_ep else 0, math.ceil(len(order) / bs)):
            if step and step % every == 0:
                if past_deadline():
                    snapshot(ep, step)
                    raise Paused(str(run_dir))
                if step % (4 * every) == 0:
                    snapshot(ep, step)
                model.train()
            losses.append(train_step(backend, model, opt, scaler, tokens, y, order[step * bs:(step + 1) * bs]))
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "epoch_wall_s_this_session": time.time() - te}
        losses = []
        row.update({f"val_{k}": v for k, v in scores(y[val_rows], predict(backend, model, tokens, val_rows), data["num_classes"]).items()})
        history.append(row)
        print(f"    centralized {dataset} s{seed} epoch {ep}: val_f1={row['val_macro_f1']:.4f}", flush=True)
        if ep < epochs:
            snapshot(ep + 1, 0)
    held = held_out_predictions(backend, model, ctx, dataset)
    clients = make_clients(data, part_for_dispersion, seed)
    pred = np.full(len(y), -1, dtype=np.int64)
    pred[ct] = predict(backend, model, tokens, ct)
    per, disp = client_dispersion(y, pred, clients, roles, data["num_classes"])
    final = {"kind": "centralized", "dataset": dataset, "seed": seed, "epochs": epochs,
             "val": {k[4:]: v for k, v in history[-1].items() if k.startswith("val_")},
             "held_out": scores(data["test_labels"], held, data["num_classes"]),
             "client_test": scores(y[ct], pred[ct], data["num_classes"]), "client_dispersion": disp,
             "resources": {"session_wall_s": time.time() - t0, "note": "simulation wall time on one GPU"}}
    finish(run_dir, final, history, {"study": study, "dataset": dataset, "seed": seed, "environment": environment(),
                                     "dispersion_partition": part_for_dispersion, "data_meta": data["meta"], "model_commit": getattr(backend, "model_commit", None)}, per)
    return final


# ============================================================================ study
def load_study(path: str) -> dict:
    p = Path(path)
    return yaml.safe_load((p if p.is_absolute() else ROOT / "configs" / p).read_text(encoding="utf-8"))


def resolve_regimes(study: dict) -> dict:
    """Regime configs exactly as the benchmark resolves them (config + cell); FL settings copied from them."""
    out = {}
    for name, spec in study["regimes"].items():
        cfg = cell_config(load_config(spec["config"]), spec["cell"])
        fl = cfg["fl"]
        if fl.get("dropout_prob", 0) or fl.get("straggler_prob", 0) or cfg["data"].get("label_noise", 0):
            raise ValueError(f"{name}: dropout, stragglers and label noise are not part of this study")
        out[name] = {"dataset": spec["dataset"], "cfg": cfg, "partition": cfg["partition"],
                     "fl": {k: fl[k] for k in ("rounds", "fraction_fit", "local_epochs", "batch_size")}
                     | {"min_fit_clients": fl.get("min_fit_clients", 1)}}
        test = study.get("test_overrides")       # tests only (tiny synthetic data); never in a study config
        if test:
            out[name]["fl"] |= test.get("fl", {})
            if cfg["partition"]["scheme"] != "natural":
                out[name]["partition"] = {**cfg["partition"], **test.get("partition", {})}
    return out


def raw_dataset(ds: str, cfg: dict) -> dict:
    from src.data.prepare import bundle_dir
    return yelp_data(cfg, bundle_dir(cfg)) if ds == "yelp" else amazon_data(cfg, bundle_dir(cfg))


def cache_tag(study: dict, cfg: dict) -> str:
    from src.data.prepare import bundle_dir
    return f"{bundle_dir(cfg).name}__{study['model'].replace('/', '_')}__len{study['max_len']}"


ARRAY_KEYS = ("labels", "roles", "codes", "test_labels", "test_rows")


def load_context(study: dict, cache: Path, backend=None, raw: dict | None = None) -> dict:
    """Data arrays + token caches per dataset. Raw texts are needed only once; afterwards the cache suffices.
    ``backend``/``raw`` let tests inject a TinyBackend and synthetic data."""
    regimes = resolve_regimes(study)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache.mkdir(parents=True, exist_ok=True)
    first = {}
    for R in regimes.values():
        first.setdefault(R["dataset"], R)
    data, tokens, test_tokens, backends = {}, {}, {}, {}
    for ds, R in first.items():
        tag = cache_tag(study, R["cfg"])
        f_arr, f_meta, f_tok, f_test = (cache / f"{tag}_{s}" for s in ("arrays.npz", "meta.json", "train.npz", "test.npz"))
        cached = f_arr.exists() and f_meta.exists() and f_tok.exists()
        if cached:
            z, meta = np.load(f_arr), json.loads(f_meta.read_text(encoding="utf-8"))
            d = {k: (z[k] if meta["present"][k] else None) for k in ARRAY_KEYS}
            d |= {"num_classes": meta["num_classes"], "meta": meta["meta"]}
        else:
            d0 = (raw or {}).get(ds) or raw_dataset(ds, R["cfg"])
            d = {k: d0[k] for k in (*ARRAY_KEYS, "num_classes", "meta")}
        if backend is None and study.get("backend") == "tiny":    # tests only
            backend = TinyBackend(d["num_classes"])
        backends[ds] = backend or HFBackend(study["model"], d["num_classes"], int(study["max_len"]), device)
        if not cached:
            print(f"tokenizing {len(d0['texts']):,} {ds} documents", flush=True)
            TokenStore.build(backends[ds].encode, d0["texts"]).save(f_tok)
            if d0["test_texts"] is not None:
                TokenStore.build(backends[ds].encode, d0["test_texts"]).save(f_test)
            del d0
            np.savez(f_arr, **{k: (d[k] if d[k] is not None else np.zeros(0, np.int64)) for k in ARRAY_KEYS})
            f_meta.write_text(json.dumps({"present": {k: d[k] is not None for k in ARRAY_KEYS}, "num_classes": d["num_classes"],
                                          "meta": d["meta"], "n_train": int(len(d["labels"])), "n_test": int(len(d["test_labels"])),
                                          "model": study["model"], "max_len": study["max_len"]}, indent=1, default=str), encoding="utf-8")
        data[ds] = d
        tokens[ds] = TokenStore.load(f_tok)
        if d["test_rows"] is None:
            test_tokens[ds] = TokenStore.load(f_test)
            if len(test_tokens[ds]) != len(d["test_labels"]):
                raise SystemExit(f"{ds}: test token cache does not match the test labels")
        if len(tokens[ds]) != len(d["labels"]):
            raise SystemExit(f"{ds}: token cache does not match the data arrays")
    return {"study": study, "regimes": regimes, "data": data, "tokens": tokens, "test_tokens": test_tokens, "backends": backends}


def run_list(study, phase, selections=None):
    runs = []
    if phase == "tune":
        for reg in study["regimes"]:
            for mu in study["mu_grid"]:
                runs.append(("fed", reg, int(study["tuning_seed"]), "fedprox", float(mu), f"tune/{reg}/fedprox_mu{mu}/seed{study['tuning_seed']}"))
    elif phase == "evaluate":
        for seed in study["seeds"]:
            for reg in study["regimes"]:
                runs.append(("fed", reg, seed, "fedavg", 0.0, f"evaluate/{reg}/fedavg/seed{seed}"))
                runs.append(("fed", reg, seed, "fedprox", float(selections[reg]["mu"]), f"evaluate/{reg}/fedprox/seed{seed}"))
            for ds in study["centralized"]:
                runs.append(("central", ds, seed, "centralized", 0.0, f"evaluate/centralized_{ds}/seed{seed}"))
    return runs


def select_mu(study, out: Path) -> dict:
    sel = {}
    for reg in study["regimes"]:
        vals = {}
        for mu in study["mu_grid"]:
            f = out / f"tune/{reg}/fedprox_mu{mu}/seed{study['tuning_seed']}/final.json"
            if not f.exists():
                raise SystemExit(f"tuning incomplete: {f} missing")
            vals[float(mu)] = json.loads(f.read_text())["val"]["macro_f1"]
        best = max(sorted(vals), key=lambda m: (vals[m], -m))   # ties -> smaller mu
        sel[reg] = {"mu": best, "val_macro_f1": {str(m): v for m, v in vals.items()}, "on_grid_edge": best in (min(vals), max(vals))}
    (out / "selections.json").write_text(json.dumps(sel, indent=1))
    return sel


def worker(ctx, phase, out: Path, selections=None) -> bool:
    """Run every unclaimed run of the phase. Returns False if stopped by the deadline."""
    for kind, reg, seed, method, mu, rel in run_list(ctx["study"], phase, selections):
        d = out / rel
        if not claim(d):
            continue
        print(f"[worker {os.getpid()}] {rel}", flush=True)
        try:
            if kind == "fed":
                ctx["backend"] = ctx["backends"][ctx["regimes"][reg]["dataset"]]
                federated_run(ctx, reg, seed, method, mu, d)
            else:
                ctx["backend"] = ctx["backends"][reg]
                centralized_run(ctx, reg, seed, d, ctx["regimes"][ctx["study"]["centralized"][reg]]["partition"])
        except Paused:
            (d / "CLAIM").unlink(missing_ok=True)
            print(f"[worker {os.getpid()}] deadline reached; {rel} checkpointed", flush=True)
            return False
    return True


def pilot(ctx, out: Path) -> dict:
    """One full-scale FedAvg round per regime (seed 42) with one validation and one held-out pass;
    projects per-run hours on this GPU (train time x rounds + evaluations at the configured checkpoints)."""
    res = {}
    for reg, R in ctx["regimes"].items():
        ctx["backend"] = ctx["backends"][R["dataset"]]
        d = out / "pilot" / reg
        if not (d / "COMPLETE").exists():
            d.mkdir(parents=True, exist_ok=True)
            federated_run(ctx, reg, 42, "fedavg", 0.0, d, eval_test=True, max_rounds=1)
        h = json.loads((d / "history.json").read_text())[0]
        fin = json.loads((d / "final.json").read_text())
        rounds = int(R["fl"]["rounds"])
        n_val = math.ceil(rounds / int(ctx["study"].get("eval_val_every", 1)))
        n_held = len(set(ctx["study"].get("eval_test_rounds", [])) | {rounds})
        proj = (h["train_wall_s"] * rounds + h["val_wall_s"] * n_val + h["held_wall_s"] * n_held
                + fin["resources"]["client_test_wall_s"]) / 3600
        res[reg] = {"train_wall_s_per_round": h["train_wall_s"], "val_wall_s": h["val_wall_s"], "held_wall_s": h["held_wall_s"],
                    "selected": h["selected"], "client_steps": h["client_steps"], "projected_hours_per_federated_run": proj,
                    "round1_val_macro_f1": h.get("val_macro_f1"), "gpu": environment()["gpu"]}
    n_runs = len(ctx["study"]["mu_grid"]) + 2 * len(ctx["study"]["seeds"])
    summary = {"regimes": res,
               "projected_federated_gpu_hours_total": sum(v["projected_hours_per_federated_run"] * n_runs for v in res.values()),
               "note": "excludes centralized runs; assumes every round costs as much as round 1"}
    (out / "pilot" / "pilot.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exploratory/extend_v1/bert.yaml")
    ap.add_argument("--phase", choices=["prepare", "pilot", "tune", "select", "evaluate", "all"], default="all")
    ap.add_argument("--gpus", type=int, default=1, help="number of workers, one per GPU (phases tune/evaluate/all)")
    ap.add_argument("--deadline-hours", type=float, help="checkpoint and stop cleanly after this many hours")
    ap.add_argument("--worker-phase", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cache", type=Path)
    a = ap.parse_args()
    study = load_study(a.config)
    out = a.out or ROOT / study["results_dir"]
    cache = a.cache or ROOT / "data_cache" / "bert_tokens"
    out.mkdir(parents=True, exist_ok=True)
    if a.deadline_hours is not None and not os.environ.get("FEDBERT_DEADLINE"):
        os.environ["FEDBERT_DEADLINE"] = str(time.time() + 3600 * a.deadline_hours)
    if a.worker_phase:                       # child process: one GPU, one phase
        ctx = load_context(study, cache)
        sel = json.loads((out / "selections.json").read_text()) if a.worker_phase == "evaluate" else None
        sys.exit(0 if worker(ctx, a.worker_phase, out, sel) else 3)
    if a.phase == "prepare":
        load_context(study, cache)
        print("data verified and token caches ready", flush=True)
        return
    if a.phase == "pilot":
        pilot(load_context(study, cache), out)
        return
    if a.phase == "select":
        print(json.dumps(select_mu(study, out), indent=1))
        return
    for f in out.rglob("CLAIM"):             # claims left by an ended session; those runs resume from checkpoints
        f.unlink()
    load_context(study, cache)               # verify + tokenize once before the workers start
    for ph in (["tune", "evaluate"] if a.phase == "all" else [a.phase]):
        if ph == "evaluate":
            print(json.dumps(select_mu(study, out), indent=1), flush=True)
        procs = []
        for g in range(max(1, a.gpus)):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)} if a.gpus > 1 else dict(os.environ)
            cmd = [sys.executable, __file__, "--config", a.config, "--worker-phase", ph, "--out", str(out), "--cache", str(cache)]
            procs.append(subprocess.Popen(cmd, env=env))
        codes = [p.wait() for p in procs]
        if any(c not in (0, 3) for c in codes):
            raise SystemExit(f"phase {ph}: worker exit codes {codes}")
        if 3 in codes:
            print(f"deadline reached during phase {ph}; rerun the same command in a new session to resume", flush=True)
            break
    print(f"complete runs: {len(list(out.rglob('COMPLETE')))} under {out}", flush=True)


if __name__ == "__main__":
    main()
