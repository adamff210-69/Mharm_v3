"""Atomic JSON I/O, resumable-loop checkpoints and chunked parquet writing."""
from __future__ import annotations

import json
import os
import time

import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(rows: list[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# --- resumable loop checkpoint -----------------------------------------------

class Checkpoint:
    """Tracks completed sample ids across process restarts.

    Usage:
        ck = Checkpoint(out/progress/extract.json)
        for s in samples:
            if ck.done(s["id"]): continue
            ... work ...
            ck.mark_done(s["id"])
        ck.finish()
    """

    def __init__(self, path: str):
        self.path = path
        self.done_ids: set[str] = set()
        self.finished = False
        self.started = time.time()
        d = load_json(path)
        if d:
            self.done_ids = set(d.get("done", []))
            self.finished = d.get("finished", False)
            self.started = d.get("started", self.started)

    def done(self, sid: str) -> bool:
        return sid in self.done_ids

    def mark_done(self, sid: str) -> None:
        self.done_ids.add(sid)

    def _flush(self) -> None:
        save_json({"done": sorted(self.done_ids), "finished": self.finished,
                   "started": self.started}, self.path)

    def mark_done_batch(self, sids: list[str], flush_every: int = 25) -> None:
        n = 0
        for s in sids:
            self.done_ids.add(s)
            n += 1
            if n % flush_every == 0:
                self._flush()

    def save(self) -> None:
        self._flush()

    def finish(self) -> None:
        self.finished = True
        self._flush()


# --- chunked parquet (v3 2.2: never accumulate the full matrix in memory) ----

class ParquetSinker:
    """Accumulates rows and flushes to parquet every ``chunk`` rows so peak
    in-memory cost is O(chunk), not O(dataset)."""

    def __init__(self, path: str, chunk: int = 200):
        self.path = path
        self.chunk = chunk
        self.buf: list[dict] = []
        self.n_flushed = 0
        ensure_dir(os.path.dirname(path) or ".")

    def add(self, row: dict) -> None:
        self.buf.append(row)
        if len(self.buf) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        df = pd.DataFrame(self.buf)
        mode = "w" if self.n_flushed == 0 else "a"
        df.to_parquet(self.path, engine="pyarrow", write_index=False) \
            if mode == "w" else \
            _append_parquet(self.path, df)
        self.n_flushed += len(self.buf)
        self.buf = []

    def close(self) -> None:
        self.flush()


def _append_parquet(path: str, df: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    old = pq.read_table(path)
    new = pa.concat_tables([old, pa.Table.from_pandas(df, preserve_index=False)])
    pq.write_table(new, path)
