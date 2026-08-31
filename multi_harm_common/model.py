"""Model loading (fp16/bf16/fp32, bitsandbytes nf4/int8) and signal-aware
forward passes."""
from __future__ import annotations

import torch

from . import env as _env


def load_model(cfg):
    """Load the base model with attention + hidden-state outputs.

    Returns (model, tokenizer, device, quant_used). The model is loaded with
    ``output_attentions`` / ``output_hidden_states`` requested at call time in
    :func:`forward_signals` (not at construction) so the same object can be
    used for both.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _env.pick_device()
    quant = _env.resolve_quant(cfg.quant, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = _env.bnb_config_for(quant)
    # attn_implementation="eager" is REQUIRED: this project consumes raw
    # attention tensors. transformers>=5 defaults to SDPA, which does not
    # materialize them (output_attentions would silently return empty).
    common = dict(trust_remote_code=True, attn_implementation="eager")
    if bnb_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=bnb_cfg, **common)
        model = model.to("cuda")
    else:
        dtype = _env.dtype_for(quant, device)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype, device_map=None, **common)
        model = model.to(device)
    model.eval()
    return model, tokenizer, device, quant


def _config_int(cfg, *names: str) -> int:
    for n in names:
        if hasattr(cfg, n):
            return int(getattr(cfg, n))
    raise AttributeError(
        f"{type(cfg).__name__} has none of {names}"
    )


def get_n_layers(model) -> int:
    """Layer count across GPT-2 (``n_layer``) and Llama/Mistral
    (``num_hidden_layers``)."""
    return _config_int(model.config, "num_hidden_layers", "n_layer", "n_layers")


def get_n_heads(model) -> int:
    """Head count across GPT-2 (``n_head``) and Llama/Mistral
    (``num_attention_heads``)."""
    return _config_int(model.config, "num_attention_heads", "n_head", "n_heads")


@torch.inference_mode()
def forward_signals(model, enc, attn_last_k: int, candidate_layers: list[int]):
    """One forward pass -> attention span masses (last K layers, per head) and
    last-token hidden states at every candidate layer.

    Returns a dict:
      masses:  {(layer_idx, head_idx): [m_qp, m_qi, m_qq]}   float32 lists
      hidden:  {layer_idx: np.ndarray (D,) float32}          last-token vector
      attn_layers: list of absolute layer indices used

    Memory note: attention tensors are processed layer by layer and released
    immediately; only the per-head scalar masses are retained. Per the v3
    §2.1 note, caching raw ``outputs.attentions[-4:]`` (4 x 32 x T x T per
    sample) is unnecessary for re-running head selection with a different
    top-K because the per-head masses determine any ratio we can form — they
    are ~2KB/sample vs ~268MB/sample for raw attention, and we cache those.
    """
    import numpy as np

    inputs = {"input_ids": enc.input_ids.to(model.device),
              "attention_mask": enc.attention_mask.to(model.device)}
    out = model(**inputs, output_attentions=True, output_hidden_states=True,
                use_cache=False)

    if not out.attentions:
        raise RuntimeError(
            "model returned no attention tensors. Multi-HARM requires "
            "attn_implementation='eager' (SDPA/flash do not materialize "
            "attentions). Check the load path in multi_harm_common/model.py.")
    n_layers = get_n_layers(model)
    attn_layers = list(range(n_layers - attn_last_k, n_layers))

    p_s, p_e = enc.passage_range
    q_s, q_e = enc.query_range
    i_s, i_e = (enc.inj_range if enc.inj_range is not None else (0, 0))

    masses: dict[tuple, list] = {}
    for abs_l, attn in zip(attn_layers, out.attentions[-attn_last_k:]):
        a = attn[0].float().cpu()          # (H, T, T) row-stochastic
        H = a.shape[0]
        m_qp = a[:, q_s:q_e, p_s:p_e].sum(dim=(1, 2))
        m_qi = a[:, q_s:q_e, i_s:i_e].sum(dim=(1, 2))
        m_qq = a[:, q_s:q_e, q_s:q_e].sum(dim=(1, 2))
        for h in range(H):
            masses[(abs_l, h)] = [float(m_qp[h]), float(m_qi[h]), float(m_qq[h])]

    last = enc.n_tokens - 1
    hidden: dict[int, np.ndarray] = {}
    # out.hidden_states[0] = embeddings; hidden_states[l] = layer l output
    for l in candidate_layers:
        vec = out.hidden_states[l][0, last].float().cpu().numpy()
        hidden[l] = vec
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Span widths (in tokens) — needed downstream to normalize the summed
    # masses to PER-COLUMN means (span-width-invariant ratio, see
    # multi_harm_common/signals.head_ratio).
    p_s, p_e = enc.passage_range
    q_s, q_e = enc.query_range
    i_s, i_e = (enc.inj_range if enc.inj_range is not None else (0, 0))
    widths = (max(1, p_e - p_s), max(1, i_e - i_s), max(1, q_e - q_s))
    return {"masses": masses, "hidden": hidden, "attn_layers": attn_layers,
            "widths": widths}
