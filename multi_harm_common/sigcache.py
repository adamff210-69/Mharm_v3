"""On-disk cache of extracted signals (written by 03_extract_signals.py).

Layout (v3 §2.2 — chunked writes, nothing accumulated in one DataFrame):

    data/signals/signals.parquet      sample_id, split, attack_type, goal,
                                      label, masses_json
    data/signals/hid_l{L}.parquet     sample_id, vec (list<halffloat>, 4096)

Row order in every file follows data/dataset.parquet, so arrays are index-
aligned with ``meta``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SigCache:
    meta: pd.DataFrame                 # columns: sample_id,split,attack_type,goal,label
    masses: dict                       # sample_id -> {(l, h): [m_qp/W_p, m_qi/W_i, m_qq/W_q]}
                                       # (per-column MEANS — normalized at load)
    widths: dict                       # sample_id -> (W_p, W_i, W_q) in tokens
    hidden: dict                       # layer -> np.ndarray (n, D) float32
    layers: list                       # sorted candidate layers

    # ---- accessors ---------------------------------------------------------
    def idx(self, sample_id: str) -> int:
        return int(self._index[sample_id])

    def _build_index(self):
        self._index = {s: i for i, s in enumerate(self.meta["sample_id"])}

    def subset(self, sample_ids: list[str]) -> dict:
        """Per-sample views for calibration routines (signals.py API):
        masses: list of dicts; hidden_by_layer: {l: (k, D)} float32; order ids.
        """
        ids = list(sample_ids)
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


def load_cache(data_dir: str) -> SigCache:
    meta = pd.read_parquet(os.path.join(data_dir, "signals", "signals.parquet"))
    width_cols = ("passage_tokens", "inj_tokens", "query_tokens")
    if any(c not in meta.columns for c in width_cols):
        raise RuntimeError(
            "Signal cache predates the span-width schema (masses are stored as "
            "RAW SUMS and must be normalized to per-column means at load time). "
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
    sigdir = os.path.join(data_dir, "signals")
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


def save_row(data_dir: str, row: dict) -> None:
    """Append one extracted sample (chunked; ParquetSinker batches)."""
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
    _meta_add(meta_row)
    for l, vec in row["hidden"].items():
        _hid_add(os.path.join(sigdir, f"hid_l{l}.parquet"),
                 {"sample_id": row["sample_id"], "vec": vec.astype(np.float16)},
                 l)


# ---- small append helpers (each file appended independently) ---------------

def _meta_add(row: dict) -> None:
    from .io_utils import _append_parquet, ensure_dir
    path = os.path.join(_data_dir_hint(), "signals", "signals.parquet")
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        pd.DataFrame([row]).to_parquet(path, index=False)
    else:
        _append_parquet(path, pd.DataFrame([row]))


def _hid_add(path: str, row: dict, layer: int) -> None:
    from .io_utils import _append_parquet, ensure_dir
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        pd.DataFrame([row]).to_parquet(path, index=False)
    else:
        _append_parquet(path, pd.DataFrame([row]))


_DIR = {"data": None}


def set_data_dir(data_dir: str) -> None:
    _DIR["data"] = data_dir


def _data_dir_hint() -> str:
    if _DIR["data"] is None:
        raise RuntimeError("set_data_dir() not called")
    return _DIR["data"]
