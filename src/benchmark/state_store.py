"""Disk-backed per-client state (SCAFFOLD client control variates).

Layout: one file of ``num_clients`` rows; row ``cid`` holds the client's
arrays flattened in order, float32, starting at byte ``cid * width * 4``
(native little-endian, no header). A client that never participated reads as
zeros (SCAFFOLD's c_i initialization) and is never written.

Rows are read and written with ordinary positioned file I/O. The file is NOT
memory-mapped: mapped pages that were written stay in the process's resident
set, so a mapping grows the process by the whole written state (measured 1:1,
about 21 GB for the Amazon population). With file I/O the process holds one
row at a time; written data lives in the kernel page cache, which is
reclaimable. On Linux the file is sparse: unwritten rows take no disk space.
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
        self.row_bytes = self.width * 4
        self.num_clients = num_clients
        self._file = open(self.path, "w+b", buffering=0)
        self._file.truncate(num_clients * self.row_bytes)   # zero-filled, sparse where supported
        self._written = np.zeros(num_clients, dtype=bool)

    @property
    def num_written(self) -> int:
        return int(self._written.sum())

    @property
    def disk_bytes_upper_bound(self) -> int:
        return self.num_clients * self.row_bytes

    def get(self, cid: int) -> list[np.ndarray]:
        if not self._written[cid]:
            return [np.zeros(s, dtype=np.float32) for s in self.shapes]
        buf = bytearray(self.row_bytes)
        self._file.seek(cid * self.row_bytes)
        view, got = memoryview(buf), 0
        while got < self.row_bytes:                          # raw reads may return fewer bytes
            n = self._file.readinto(view[got:])
            if not n:
                raise IOError(f"short read of client {cid} state from {self.path}")
            got += n
        row = np.frombuffer(buf, dtype=np.float32)
        out, start = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(row[start:start + size].reshape(shape))
            start += size
        return out

    def put(self, cid: int, arrays: list[np.ndarray]) -> None:
        row = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
        if row.size != self.width:
            raise ValueError(f"client {cid}: expected {self.width} values, got {row.size}")
        data = memoryview(row.tobytes())
        self._file.seek(cid * self.row_bytes)
        written = 0
        while written < len(data):                           # raw writes may be partial
            written += self._file.write(data[written:])
        self._written[cid] = True

    def close(self, delete: bool = True) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if delete:
            try:
                os.remove(self.path)
            except OSError:
                pass
