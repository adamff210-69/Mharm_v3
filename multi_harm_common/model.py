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
        # bitsandbytes requires a device_map at load time; placing it there
        # directly avoids a post-hoc model.to("cuda") on an already-quantized
        # module (a no-op at best, and it raises under some accelerate hooks).
        dmap = {"": device} if device != "cpu" else None
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=bnb_cfg, device_map=dmap,
            **common)
    else:
        dtype = _env.dtype_for(quant, device)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype, device_map=None, **common)
        model = model.to(device)
    model.eval()
    return model, tokenizer, device, quant


# Attribute names differ by architecture family: GPT-2 / NeoX expose
# ``n_layer`` / ``n_head``, Llama / Mistral / Qwen expose
# ``num_hidden_layers`` / ``num_attention_heads``. The default model is
# Llama-3.1-8B, so reading only ``config.n_layer`` (as v3.0 did) crashes on
# the real target while passing the gpt2 smoke test — hence the lookup chain.
_N_LAYER_ATTRS = ("num_hidden_layers", "n_layer", "num_layers")
_N_HEAD_ATTRS = ("num_attention_heads", "n_head", "num_heads")


def _cfg_int(model, names: tuple[str, ...], what: str) -> int:
    cfg = getattr(model, "config", None)
    tried = []
    for name in names:
        val = getattr(cfg, name, None)
        if isinstance(val, (int,)) and int(val) > 0:
            return int(val)
        if val is not None:
            tried.append(f"{name}={val!r}")
    raise RuntimeError(
        f"could not determine {what} from {type(cfg).__name__}. "
        f"Looked for {names} (non-integer/missing: {tried or 'none'}); add the "
        f"right attribute name to _N_LAYER_ATTRS/_N_HEAD_ATTRS in "
        f"multi_harm_common/model.py for this architecture.")


def get_n_layers(model) -> int:
    return _cfg_int(model, _N_LAYER_ATTRS, "transformer layer count")


def get_n_heads(model) -> int:
    return _cfg_int(model, _N_HEAD_ATTRS, "attention head count")


@torch.inference_mode()
def forward_signals(model, enc, attn_last_k: int, candidate_layers: list[int]):
    """One forward pass -> attention span masses (last K layers, per head) and
    last-token hidden states at every candidate layer.

    Returns a dict:
      masses:  {(layer_idx, head_idx): [m_qp, m_qi, m_qq]}   float32 lists
      hidden:  {layer_idx: np.ndarray (D,) float32}          last-token vector
      attn_layers: list of absolute layer indices used
      widths:  (W_p, W_i, W_q) tokens, measured on the CLIPPED spans
      clipped: True if any span was shortened by max_seq_len truncation
      n_tokens_used: tokens actually fed to the model

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

    # Clip every span to the tokens actually kept (encode_sample truncates to
    # max_seq_len). Masses are sliced from the truncated attention matrix, so
    # the span WIDTHS used downstream to form per-column means must be clipped
    # identically — otherwise long samples (where W_i/W_p still count dropped
    # tokens) get their per-token intensity divided by a too-large width, which
    # is exactly the length confound the invariant ratio exists to remove.
    # Spans must be measured on the tokens ACTUALLY fed. If a backend pads, or a
    # checkpoint truncates differently than encode_sample assumed, slicing at the
    # logical length reads the wrong rows and still returns finite numbers — the
    # same class of silent mismatch that cost v3.0 its entire head selection.
    T = int(enc.input_ids.shape[1])
    T_fwd = int(out.attentions[-1].shape[-1])
    if T_fwd < T:
        raise RuntimeError(
            f"attention matrix is {T_fwd}x{T_fwd} but the encoding declares {T} "
            f"tokens; span masses would be sliced off the end of the matrix")
    if T_fwd > T:
        print(f"  NOTE: attention is {T_fwd} wide for {T} fed tokens (padding); "
              f"measuring spans on the first {T} positions only")
    clipped = False

    def _clip(rng, fallback):
        nonlocal clipped
        s, e = int(rng[0]), int(rng[1])
        cs, ce = max(0, min(s, T)), max(0, min(e, T))
        if (cs, ce) != (s, e):
            clipped = True
        return (cs, ce) if ce > cs else fallback

    p_s, p_e = _clip(enc.passage_range, (0, T))
    q_s, q_e = _clip(enc.query_range, (0, T))
    i_s, i_e = _clip(enc.inj_range if enc.inj_range is not None else (0, 0),
                     (0, 0))

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
    # Span widths (in tokens) from the CLIPPED ranges above, so widths and
    # masses always describe the same tokens. Needed downstream to normalize
    # the summed masses to per-column means (span-width-invariant ratio, see
    # multi_harm_common/signals.head_ratio).
    widths = (max(1, p_e - p_s), max(1, i_e - i_s), max(1, q_e - q_s))
    return {"masses": masses, "hidden": hidden, "attn_layers": attn_layers,
            "widths": widths, "clipped": clipped, "n_tokens_used": T}
