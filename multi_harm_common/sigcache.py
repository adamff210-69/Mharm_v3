"""On-disk cache of extracted signals (written by 03_extract_signals.py).

Layout (v3 §2.2 — chunked writes, nothing accumulated in one DataFrame):

    data/signals/signals.parquet      sample_id, split, attack_type, goal,
                                      label, passage_tokens, inj_tokens,
                                      query_tokens, masses_json
    data/signals/hid_l{L}.parquet     sample_id, vec (list<halfprecision>, D)
    data/signals/schema.json            schema version marker

Row order in every file follows the extraction order of data/dataset.parquet,
so the hidden-state arrays are index-aligned with ``meta``.

Writes go through one :class:`~multi_harm_common.io_utils.ParquetSinker` per
output file: rows are buffered and the file is rewritten once per ``chunk_rows``
(not once per row), which bounds both peak memory and total I/O. Call
``flush_sinks()`` whenever you also flush the extraction checkpoint, and
``close_sinks()`` at the end — otherwise the last partial chunk is lost while
the checkpoint would claim those samples are done.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_utils import ParquetSinker

# Bump when the *meaning* of cached columns changes, so a cache written by an
# older definition cannot be silently reused (see load_cache's error text).
#   1 = v3.0  raw per-head column sums, widths taken from un-clipped spans
#   2 = v3.1  widths measured on spans clipped to max_seq_len
SCHEMA_VERSION = 2
_META_NAME = "signals.parquet"
_SCHEMA_NAME = "schema.json"
_PROV_NAME = "sigcache_meta.json"

_SINKS: dict[str, ParquetSinker] = {}
_CHUNK = {"rows": 200}
_DIR = {"data": None}


def set_data_dir(data_dir: str) -> None:
    """Legacy hook: pin the data dir used by save_row when no dir is passed."""
    _DIR["data"] = data_dir


def configure(data_dir: str, chunk_rows: int = 200) -> None:
    """Call once before the first ``save_row``.

    Writes the schema marker and refuses to proceed if the existing cache was
    produced by an incompatible schema.
    """
    set_data_dir(data_dir)
    _CHUNK["rows"] = max(1, int(chunk_rows))
    sigdir = os.path.join(data_dir, "signals")
    os.makedirs(sigdir, exist_ok=True)
    prior = _read_schema(sigdir)
    if prior is not None and prior != SCHEMA_VERSION:
        raise RuntimeError(
            f"data/signals was extracted with schema v{prior}, this code writes "
            f"v{SCHEMA_VERSION}. Delete data/signals and out/progress/extract.json "
            f"and re-run 03_extract_signals.py.")
    with open(os.path.join(sigdir, _SCHEMA_NAME), "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION}, f, indent=2)


def provenance_path(data_dir: str) -> str:
    return os.path.join(data_dir, "signals", _PROV_NAME)


def save_provenance(meta: dict, data_dir: str) -> str:
    """Record *how* the cache was produced, next to it. Cheap, and it is what turns
    a stale cache from an undiagnosable wrong answer into a named error."""
    path = provenance_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(meta)
    meta["schema_version"] = SCHEMA_VERSION
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def load_provenance(data_dir: str) -> dict:
    try:
        with open(provenance_path(data_dir)) as f:
            return json.load(f)
    except Exception:
        return {}


def _sinker(path: str) -> ParquetSinker:
    s = _SINKS.get(path)
    if s is None:
        s = _SINKS[path] = ParquetSinker(path, chunk=_CHUNK["rows"])
    return s


def flush_sinks() -> None:
    """Push buffered rows to disk (call together with Checkpoint.save())."""
    for s in _SINKS.values():
        s.flush()


def close_sinks() -> None:
    flush_sinks()
    _SINKS.clear()


def pending_rows() -> int:
    return sum(s.pending for s in _SINKS.values())


# ---------------------------------------------------------------------------
# Cache object
# ---------------------------------------------------------------------------

@dataclass
class SigCache:
    meta: pd.DataFrame                 # sample_id,split,attack_type,goal,label,+widths
    masses: dict                       # sample_id -> {(l, h): [m_qp, m_qi, m_qq]}
                                       # (RAW column sums; normalized in
                                       #  signals.head_ratio at use time)
    widths: dict                       # sample_id -> (W_p, W_i, W_q) in tokens,
                                       # measured on spans clipped to max_seq_len
    hidden: dict                       # layer -> np.ndarray (n, D) float32
    layers: list                       # sorted candidate layers

    # ---- accessors ---------------------------------------------------------
    def idx(self, sample_id: str) -> int:
        return int(self._index[sample_id])

    def has(self, sample_id: str) -> bool:
        return sample_id in self._index

    def _build_index(self):
        self._index = {s: i for i, s in enumerate(self.meta["sample_id"])}

    def subset(self, sample_ids: list[str]) -> dict:
        """Per-sample views for calibration routines (signals.py API):
        masses: list of dicts; hid: {l: (k, D)} float32; order ids.
        """
        ids = list(sample_ids)
        missing = [s for s in ids if s not in self._index]
        if missing:
            raise KeyError(
                f"{len(missing)} sample id(s) are in the dataset but not in the "
                f"signal cache (e.g. {missing[:5]}) — extraction skipped them, "
                f"or the cache is stale. Load the frame through "
                f"sigcache.usable_df(df, cache) (what 04-10 now do) to restrict "
                f"every stage to extracted samples, or re-run "
                f"03_extract_signals.py.")
        ixs = [self._index[s] for s in ids]
        masses = [self.masses[s] for s in ids]
        widths = [self.widths[s] for s in ids]
        hid = {l: self.hidden[l][ixs] for l in self.layers}
        m = self.meta.set_index("sample_id")
        labels = np.array([int(m.loc[s, "label"]) for s in ids])
        types = np.array([m.loc[s, "attack_type"] for s in ids])
        return {"ids": ids, "masses": masses, "widths": widths, "hid": hid,
                "labels": labels, "types": types}

    def hidden_dict_for(self, sample_ids: list[str], layers: list[int]) -> list[dict]:
        """Per-sample {layer: vector} dicts for the given layers only (cheap)."""
        ixs = [self._index[s] for s in sample_ids]
        out = []
        for k, i in enumerate(ixs):
            out.append({l: self.hidden[l][i] for l in layers})
        return out


_warned: set[tuple] = set()


def usable_df(df: pd.DataFrame, cache: SigCache, verbose: bool = True) -> pd.DataFrame:
    """Restrict a dataset frame to the samples that actually have signals.

    03 skips (and checkpoint-skips) samples whose token ranges cannot be
    validated, which desynchronizes dataset.parquet from the cache; every
    calibration/analysis stage must then ignore those ids instead of raising a
    KeyError from deep inside the cache. The dropped ids are reported so the
    skip is never silent.
    """
    have = set(cache.meta["sample_id"])
    keep = df["id"].isin(have)
    dropped = df.loc[~keep, "id"].tolist()
    key = (len(dropped), dropped[0] if dropped else None)
    if dropped and verbose and key not in _warned:
        _warned.add(key)
        print(f"  [sigcache] {len(dropped)} dataset row(s) have no cached "
              f"signals and are excluded from this stage "
              f"(first 8: {dropped[:8]}). See "
              f"out/validation/extraction_report.json for why.")
    return df.loc[keep].reset_index(drop=True)


def load_cache(data_dir: str) -> SigCache:
    sigdir = os.path.join(data_dir, "signals")
    meta_path = os.path.join(sigdir, _META_NAME)
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"no signal cache at {meta_path} — run 03_extract_signals.py first "
            f"(everything from 04 onward reads only this cache).")
    prior = _read_schema(sigdir)
    if prior is not None and prior != SCHEMA_VERSION:
        raise RuntimeError(
            f"data/signals was extracted with schema v{prior}, this code expects "
            f"v{SCHEMA_VERSION}. Delete data/signals and "
            f"out/progress/extract.json and re-run 03_extract_signals.py.")
    meta = pd.read_parquet(meta_path)
    width_cols = ("passage_tokens", "inj_tokens", "query_tokens")
    if any(c not in meta.columns for c in width_cols):
        raise RuntimeError(
            "Signal cache predates the span-width schema (masses are stored as "
            "RAW SUMS and must be normalized to per-column means at use time). "
            "Delete data/signals and out/progress/extract.json and re-run "
            "03_extract_signals.py.")
    wp = np.maximum(1, meta["passage_tokens"].to_numpy().astype(float))
    wi = np.maximum(1, meta["inj_tokens"].to_numpy().astype(float))
    wq = np.maximum(1, meta["query_tokens"].to_numpy().astype(float))
    # RAW column-sums are stored as extracted; the span-width-invariant
    # normalization happens in exactly one place — signals.head_ratio
    # (see its docstring for why).
    masses = {}
    for sid, js in zip(meta["sample_id"], meta["masses_json"]):
        d = json.loads(js)
        masses[sid] = {tuple(map(int, k.split("|"))): list(map(float, m))
                       for k, m in d.items()}
    widths = {(sid): (int(w_p), int(w_i), int(w_q))
              for sid, w_p, w_i, w_q in zip(meta["sample_id"], wp, wi, wq)}
    layers = []
    hid = {}
    for f in sorted(os.listdir(sigdir)):
        if f.startswith("hid_l") and f.endswith(".parquet"):
            l = int(f[5:-8])
            df = pd.read_parquet(os.path.join(sigdir, f))
            arr = np.array(df["vec"].tolist(), dtype=np.float32)
            hid[l] = arr
            layers.append(l)
    layers.sort()
    cache = SigCache(meta=meta, masses=masses, widths=widths, hidden=hid,
                     layers=layers)
    cache._build_index()
    return cache


def _read_schema(sigdir: str) -> int | None:
    p = os.path.join(sigdir, _SCHEMA_NAME)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return int(json.load(f).get("schema_version", 1))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_row(data_dir: str, row: dict) -> None:
    """Buffer one extracted sample; flushed by ParquetSinker every chunk_rows."""
    sigdir = os.path.join(data_dir, "signals")
    os.makedirs(sigdir, exist_ok=True)
    meta_row = {k: row[k] for k in ("sample_id", "split", "attack_type",
                                    "goal", "label")}
    w_p, w_i, w_q = row["widths"]
    meta_row["passage_tokens"] = int(w_p)
    meta_row["inj_tokens"] = int(w_i)
    meta_row["query_tokens"] = int(w_q)
    meta_row["masses_json"] = json.dumps({f"{l}|{h}": m
                                          for (l, h), m in row["masses"].items()})
    _sinker(os.path.join(sigdir, _META_NAME)).add(meta_row)
    for l, vec in row["hidden"].items():
        _sinker(os.path.join(sigdir, f"hid_l{l}.parquet")).add(
            {"sample_id": row["sample_id"], "vec": np.asarray(vec).astype(np.float16)})
