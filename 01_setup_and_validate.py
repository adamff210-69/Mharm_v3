#!/usr/bin/env python3
"""01 — Environment & model setup (v3 Phase 1).

* prints a full environment report (device, quant, RAM, GPU)
* downloads/loads the model with attention + hidden-state support
* runs a 3-sentence shape smoke test of the signal-aware forward pass
* writes out/env_report.json

Run:  python 01_setup_and_validate.py
Colab free-tier note: 8B in fp16 (~16 GB) will NOT fit a T4; the default
quant='auto' resolves to nf4 on CUDA, which is the intended deployment
condition for this project.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common import env as ENV
from multi_harm_common.io_utils import save_json, ensure_dir
from multi_harm_common.model import (forward_signals, get_n_heads, get_n_layers,
                                     load_model)
from multi_harm_common.chat import encode_sample


def main():
    cfg = load_config()
    device = ENV.pick_device()
    quant = ENV.resolve_quant(cfg.quant, device)
    report = ENV.print_env(cfg, device, quant)

    print("\nLoading model ...")
    t0 = time.time()
    model, tokenizer, device, quant = load_model(cfg)
    n_layers = get_n_layers(model)
    cand = cfg.candidate_layers(n_layers)
    report.update({"load_seconds": round(time.time() - t0, 1),
                   "n_layers": n_layers,
                   "candidate_hidden_layers": cand,
                   "attn_layers_used": list(range(n_layers - cfg.attn_last_k, n_layers))})
    print(f"  loaded in {report['load_seconds']}s | layers={n_layers} | "
          f"L* candidates={cand} | attention layers="
          f"{list(range(n_layers - cfg.attn_last_k, n_layers))}")

    # ---- shape smoke test -------------------------------------------------
    print("\nShape smoke test (3 texts, attention + hidden extraction) ...")
    from config import GOALS
    from multi_harm_common.dataset import build_injection
    p2 = ("Urban planning reviews often balance growth against "
          "housing affordability across districts. " * 4)
    p2_inj, off2 = build_injection(p2, "naive", GOALS[0])
    texts = [
        {"id": "smoke1", "passage": ("The quick brown fox jumps over the lazy dog. " * 6),
         "query": "What does the fox do?", "injection": "", "injection_offset": [None, None]},
        {"id": "smoke2", "passage": p2_inj,
         "query": "What do planners balance?", "injection": p2_inj[off2[0]:off2[1]],
         "injection_offset": list(off2)},
        {"id": "smoke3", "passage": ("Marine biology research tracks coral bleaching events "
                                      "using satellite imagery and field surveys. " * 4),
         "query": "How is bleaching tracked?", "injection": "", "injection_offset": [None, None]},
    ]
    for t in texts:
        enc = encode_sample(tokenizer, t, cfg.max_seq_len, cfg.tail_len)
        assert enc.valid, f"smoke encode failed: {enc.note}"
        sig = forward_signals(model, enc, cfg.attn_last_k, cand)
        n_mass = len(sig["masses"])
        n_hid = len(sig["hidden"])
        n_layers_att = len(sig["attn_layers"])
        sample_m = list(sig["masses"].values())[0]
        n_heads = get_n_heads(model)
        print(f"  {t['id']}: tokens={enc.n_tokens} "
              f"masses={n_layers_att}x{n_mass // n_layers_att} heads "
              f"hidden_layers={n_hid} m_qp={sample_m[0]:.4f} m_qi={sample_m[1]:.4f}")
        assert n_mass == n_layers_att * n_heads
        assert n_hid == len(cand)

    ensure_dir(cfg.out_dir)
    save_json(report, f"{cfg.out_dir}/env_report.json")
    print(f"\nOK — env report written to {cfg.out_dir}/env_report.json")


if __name__ == "__main__":
    main()
