"""Disk-backed per-client state (SCAFFOLD client control variates).

One float32 row of the flattened model size per client, in a memory-mapped
file. A client that never participated reads as zeros (SCAFFOLD's c_i
initialization). On Linux the file is sparse: only rows that were written
take disk space. Nothing of it is kept in RAM beyond the OS page cache.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class ClientStateStore:
    def __init__(self, path: Path, num_clients: int, shapes: list[tuple]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shapes = [tuple(s) for s in shapes]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.width = sum(self.sizes)
        self.num_clients = num_clients
        self._mm = np.memmap(self.path, dtype=np.float32, mode="w+", shape=(num_clients, self.width))
        self._written = np.zeros(num_clients, dtype=bool)

    @property
    def num_written(self) -> int:
        return int(self._written.sum())

    @property
    def disk_bytes_upper_bound(self) -> int:
        return self.num_clients * self.width * 4

    def get(self, cid: int) -> list[np.ndarray]:
        if not self._written[cid]:
            return [np.zeros(s, dtype=np.float32) for s in self.shapes]
        row = np.array(self._mm[cid])
        out, start = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(row[start:start + size].reshape(shape))
            start += size
        return out

    def put(self, cid: int, arrays: list[np.ndarray]) -> None:
        self._mm[cid] = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
        self._written[cid] = True

    def close(self, delete: bool = True) -> None:
        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None
        if delete:
            try:
                os.remove(self.path)
            except OSError:
                pass
