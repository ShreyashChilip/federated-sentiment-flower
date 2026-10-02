"""Flower ClientApp: local training, local evaluation and a registration query.

The client identity is the ``partition-id`` assigned by the Flower simulation
runtime. Everything else (configuration, partition, labels) is read from the
run directory named in the incoming message.
"""
from __future__ import annotations

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.clients.trainer import evaluate, local_train
from src.models.registry import build_model, set_weights
from src.resources.monitor import peak_vram_mb, rss_mb
from src.utils.runtime import load_runtime, resolve_device
from src.utils.seeding import derive_seed

app = ClientApp()


def _setup(msg: Message, context: Context):
    rt = load_runtime(msg.content["config"]["run_dir"])
    client_id = int(context.node_config["partition-id"])
    torch.set_num_threads(int(rt.cfg.get("client_threads", 1)))
    return rt, client_id


@app.query()
def register(msg: Message, context: Context) -> Message:
    """Tell the server which partition this node holds and how large it is."""
    rt, cid = _setup(msg, context)
    metrics = MetricRecord({
        "client_id": cid,
        "n_train": int(len(rt.client_rows(cid, "train"))),
        "n_val": int(len(rt.client_rows(cid, "val"))),
        "n_client_test": int(len(rt.client_rows(cid, "client_test"))),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    rt, cid = _setup(msg, context)
    conf = msg.content["config"]
    fl = rt.cfg["fl"]
    algorithm = fl["algorithm"]
    device = resolve_device(rt.cfg)
    rnd = int(conf["round"])

    rows = rt.client_rows(cid, "train")
    model = build_model(rt.cfg["model"], rt.bundle.input_dim, rt.bundle.num_classes)
    global_weights = msg.content["arrays"].to_numpy_ndarrays()
    set_weights(model, global_weights)

    extra = {}
    if algorithm == "scaffold":
        extra["c_global"] = msg.content["c_global"].to_numpy_ndarrays()
        if "c_local" in context.state.array_records:
            extra["c_local"] = context.state["c_local"].to_numpy_ndarrays()
        else:
            extra["c_local"] = [np.zeros_like(w) for w in global_weights]

    result = local_train(
        model, rt.bundle.x_train[rows], rt.y_train[rows],
        epochs=float(conf["epochs"]), batch_size=int(fl["batch_size"]), lr=float(conf["lr"]),
        momentum=float(fl.get("momentum", 0.0)), weight_decay=float(fl.get("weight_decay", 0.0)),
        seed=derive_seed("batch", rt.seed, cid, rnd), device=device,
        algorithm=algorithm, mu=float(fl.get("mu", 0.0)), **extra,
    )

    content = {"arrays": ArrayRecord(numpy_ndarrays=result["weights"])}
    if algorithm == "scaffold":
        context.state["c_local"] = ArrayRecord(numpy_ndarrays=result["c_new"])
        content["c_delta"] = ArrayRecord(numpy_ndarrays=result["c_delta"])
    content["metrics"] = MetricRecord({
        "client_id": cid,
        "num_examples": int(result["num_examples"]),
        "steps": int(result["steps"]),
        "train_loss": float(result["train_loss"]),
        "train_wall_s": float(result["train_wall_s"]),
        "train_cpu_s": float(result["train_cpu_s"]),
        "on_gpu": int(device == "cuda"),
        "rss_mb": float(rss_mb()),
        "vram_mb": float(peak_vram_mb()),
        # Norm of the control variate this client started the round with; it is
        # non-zero from the client's second participation if state persists.
        "c_local_norm": float(np.sqrt(sum(float((c ** 2).sum()) for c in extra.get("c_local", [])))),
    })
    return Message(content=RecordDict(content), reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on this client's held-out rows."""
    rt, cid = _setup(msg, context)
    device = resolve_device(rt.cfg)
    model = build_model(rt.cfg["model"], rt.bundle.input_dim, rt.bundle.num_classes)
    set_weights(model, msg.content["arrays"].to_numpy_ndarrays())
    metrics: dict = {"client_id": cid}
    for role in ("val", "client_test"):
        rows = rt.client_rows(cid, role)
        # Evaluation always uses the clean labels.
        out = evaluate(model, rt.bundle.x_train[rows], rt.bundle.y_train[rows], rt.bundle.num_classes, device)
        metrics[f"{role}_n"] = int(out["n"])
        if out["n"] > 0:
            metrics[f"{role}_loss"] = float(out["loss"])
            metrics[f"{role}_confusion"] = [int(v) for row in out["confusion_matrix"] for v in row]
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


def config_record(run_dir, **values) -> ConfigRecord:
    return ConfigRecord({"run_dir": str(run_dir), **values})
