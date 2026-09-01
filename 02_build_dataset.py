#!/usr/bin/env python3
"""02 — Dataset construction (v3 Phase 1: 1,000 clean MS-MARCO pairs + 1,000
injected = 4 attack types x 5 goals x 50; 60/20/20 split stratified by attack
type AND goal).

Writes data/dataset.parquet with columns:
    id, split, attack_type, goal, passage, query, injection,
    injection_offset (passage-relative char span), label

Run:  python 02_build_dataset.py [--force] [--synthetic] [--jsonl PATH]
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, ".")
from config import load_config, GOALS
from multi_harm_common.dataset import build_dataset, find_qrag_jsonl
from multi_harm_common.io_utils import ensure_dir, save_json, load_json


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fingerprint(cfg, dataset_path: str) -> None:
    save_json({"sha256": _sha256_file(dataset_path)},
              os.path.join(cfg.data_dir, "dataset_fingerprint.json"))


def read_fingerprint(cfg) -> str | None:
    fp = load_json(os.path.join(cfg.data_dir, "dataset_fingerprint.json"))
    return fp.get("sha256") if fp else None


def _invalidate_signals(cfg, dataset_path: str) -> None:
    """Remove signal cache + extraction checkpoint if they were produced from
    a different dataset (same sample-id scheme, different content would
    silently corrupt every downstream number)."""
    sigdir = os.path.join(cfg.data_dir, "signals")
    ckpt = os.path.join(cfg.out_dir, "progress", "extract.json")
    if os.path.exists(sigdir) or os.path.exists(ckpt):
        print("  -> clearing stale signal cache / checkpoint")
        import shutil
        if os.path.exists(sigdir):
            shutil.rmtree(sigdir)
        if os.path.exists(ckpt):
            os.remove(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--synthetic", action="store_true",
                    help="offline synthetic clean pairs (no HF download)")
    ap.add_argument("--jsonl", default="",
                    help="QuietRAG qrag_v1.jsonl (pre-spanned attacks)")
    args = ap.parse_args()

    cfg = load_config()
    if args.synthetic:
        cfg.synthetic_clean = True
    if args.jsonl:
        cfg.qrag_jsonl = args.jsonl

    import pandas as pd
    out = os.path.join(cfg.data_dir, "dataset.parquet")
    qpath = find_qrag_jsonl(cfg.qrag_jsonl)
    if qpath:
        expected_n = None
        cfg.qrag_jsonl = qpath
        print(f"  using QuietRAG jsonl: {qpath}")
    else:
        expected_n = cfg.n_clean + 4 * len(GOALS) * cfg.n_inj_per_cell
    reusable = False
    if os.path.exists(out) and not args.force:
        cur_hash = _sha256_file(out)
        fp_hash = read_fingerprint(cfg)
        df = pd.read_parquet(out)
        if ((expected_n is None) or len(df) == expected_n) and (
                fp_hash is None or fp_hash == cur_hash) and not qpath:
            reusable = True
        else:
            print(f"WARNING: existing dataset is stale (rows={len(df)} vs "
                  f"expected {expected_n}, fingerprint match="
                  f"{fp_hash == cur_hash}). Rebuilding and invalidating the "
                  f"signal cache.")
    if reusable:
        print(f"Dataset already built: {len(df)} rows -> {out}")
        print_split_stats(df)
        return
    _invalidate_signals(cfg, out)

    print("Loading clean pairs ...")
    df = build_dataset(cfg)
    ensure_dir(cfg.data_dir)
    df.to_parquet(out, index=False)
    write_fingerprint(cfg, out)
    print(f"\nWrote {len(df)} rows -> {out}")
    print_split_stats(df)
    # preview
    r = df[df["label"] == 1].iloc[0]
    print("\nPreview (first injected):")
    print(f"  type={r.attack_type} goal={r.goal}")
    print(f"  passage tail: ...{r.passage[-160:]}")
    print(f"  query: {r.query}")


def print_split_stats(df):
    import pandas as pd
    print("\n  split x attack_type:")
    print(pd.crosstab(df["attack_type"], df["split"]).to_string())
    print("\n  injected cells (type x goal), test counts:")
    inj = df[df["label"] == 1]
    print(pd.crosstab(inj["attack_type"], inj["goal"]).to_string())
    print(f"\n  injection length chars: min={inj['injection'].str.len().min()} "
          f"mean={inj['injection'].str.len().mean():.0f} "
          f"max={inj['injection'].str.len().max()}")


if __name__ == "__main__":
    main()
