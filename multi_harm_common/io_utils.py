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

    def save(self) -> None:
        self._flush()

    def finish(self) -> None:
        self.finished = True
        self._flush()


# --- chunked parquet (v3 2.2: never accumulate the full matrix in memory) ----

class ParquetSinker:
    """Buffers rows and flushes to parquet every ``chunk`` rows.

    Two costs are bounded here, both of which mattered in v3.0 (where
    ``save_row`` appended one row at a time and every append re-read and
    re-wrote the whole file — O(n^2) row-writes across ~18 files):

    * peak in-memory rows = ``chunk``, never the whole dataset (v3 2.2);
    * full-file rewrites = ``n/chunk`` instead of ``n``.

    Resume-safe: the starting row count is read from an existing file, so a
    restarted extraction run appends to the same file rather than truncating
    the already-extracted rows.
    """

    def __init__(self, path: str, chunk: int = 200):
        self.path = path
        self.chunk = max(1, int(chunk))
        self.buf: list[dict] = []
        self.n_flushed = parquet_rows(path)
        ensure_dir(os.path.dirname(path) or ".")

    @property
    def pending(self) -> int:
        return len(self.buf)

    def add(self, row: dict) -> None:
        self.buf.append(row)
        if len(self.buf) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        df = pd.DataFrame(self.buf)
        if self.n_flushed == 0:
            # index=False, not write_index=False: the latter is a fastparquet
            # kwarg, and passing it to the pyarrow engine raises TypeError —
            # which is why this class was dead code in v3.0 (README claimed
            # chunked writes; save_row appended one row at a time instead).
            df.to_parquet(self.path, engine="pyarrow", index=False)
        else:
            _append_parquet(self.path, df)
        self.n_flushed += len(self.buf)
        self.buf = []

    def close(self) -> None:
        self.flush()


def parquet_rows(path: str) -> int:
    """Row count of an existing parquet file (0 if absent); metadata-only."""
    if not os.path.exists(path):
        return 0
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import pandas as pd
        return int(len(pd.read_parquet(path)))


def _append_parquet(path: str, df: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    old = pq.read_table(path)
    new = pa.concat_tables([old, pa.Table.from_pandas(df, preserve_index=False)])
    pq.write_table(new, path)
