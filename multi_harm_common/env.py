"""Device / dtype detection and environment reporting (portable across
Colab, Kaggle and local machines)."""
from __future__ import annotations

import platform
import sys

import numpy as np
import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_quant(quant: str, device: str) -> str:
    """'auto' -> nf4 on CUDA, fp16 on MPS, fp32 on CPU."""
    if quant and quant != "auto":
        return quant
    if device == "cuda":
        return "nf4"
    if device == "mps":
        return "fp16"
    return "fp32"


def dtype_for(quant: str, device: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "nf4": torch.float16,   # compute dtype for nf4 (bnb handles storage)
        "int8": torch.bfloat16 if device == "cuda" else torch.float32,
    }[quant]


def bnb_config_for(quant: str):
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if quant == "int8":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def gpu_summary() -> dict:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu_name"] = p.name
        out["gpu_mem_gb"] = round(p.total_memory / 1e9, 2)
        free, total = torch.cuda.mem_get_info()
        out["gpu_free_gb"] = round(free / 1e9, 2)
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import bitsandbytes as bnb
        out["bitsandbytes"] = bnb.__version__
    except Exception:
        out["bitsandbytes"] = None
    try:
        import sklearn
        out["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(vm.total / 1e9, 2)
        out["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:
        pass
    return out


def print_env(cfg, device: str, quant: str) -> dict:
    s = gpu_summary()
    s.update({"device": device, "quant": quant, "model": cfg.model_name,
              "max_seq_len": cfg.max_seq_len, "test_mode": cfg.test_mode})
    print("=" * 78)
    print("MULTI-HARM ENVIRONMENT")
    print("=" * 78)
    for k, v in s.items():
        print(f"  {k:22s}: {v}")
    if device == "cpu" and not cfg.test_mode:
        print("  WARNING: running on CPU outside test mode. Use MULTI_HARM_TEST_MODE=true")
        print("           for CPU runs or move to a GPU machine.")
    return s
