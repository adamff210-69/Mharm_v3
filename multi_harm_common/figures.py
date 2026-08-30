"""All paper figures (matplotlib, Agg). Each function reads plain data and
writes a PNG to out/figures/. Keep functions side-effect free until the final
write so tests can call them cheaply."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

TYPE_ORDER = ["topic", "naive", "fake", "combined"]
TYPE_COLORS = {"topic": "#4C72B0", "naive": "#DD8452", "fake": "#55A868",
               "combined": "#8172B3", "general": "#999999", "clean": "#CCCCCC"}


def _save(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


# v3 Phase 5 addition — THE key figure: per-specialist half-split AUROCs
def fig_halves_per_specialist(specs: list[dict], out_dir: str) -> str:
    names = [s["name"] for s in specs]
    att = [s["auroc"]["att_head"] for s in specs]
    hid = [s["auroc"]["hid"] for s in specs]
    fused = [s["auroc"]["fused"] for s in specs]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, att, w, label="Attention half (R)", color="#4C72B0")
    ax.bar(x, hid, w, label="Hidden half (P(inj))", color="#DD8452")
    ax.bar(x + w, fused, w, label="Fused (S)", color="#55A868")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x, names)
    ax.set_ylabel("AUROC (calibration set)")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.set_title("Specialization is carried by the residual-stream half")
    return _save(fig, out_dir, "fig_halves_per_specialist.png")


# v3 §4.8 — the spine table, as a figure
def fig_48_configs(configs: dict, out_dir: str) -> str:
    """configs: {config_name: {attack_type: auroc, 'spread': f}}"""
    names = list(configs.keys())
    types = TYPE_ORDER
    x = np.arange(len(types))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#8172B3", "#4C72B0", "#DD8452"]
    for i, nm in enumerate(names):
        vals = [configs[nm].get(t, float("nan")) for t in types]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w,
               label=f"{nm} (spread {configs[nm].get('spread', float('nan')):.3f})",
               color=palette[i % len(palette)])
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x, types)
    ax.set_ylabel("AUROC (test set)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Shared vs specialized calibration (v3 §4.8)")
    return _save(fig, out_dir, "fig_48_configs.png")


def fig_rocs(curves: dict, out_dir: str) -> str:
    """curves: {name: (fpr, tpr)}"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nm, (f, t) in curves.items():
        ax.plot(f, t, label=nm)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig_rocs.png")


def fig_confusion(records: list[dict], out_dir: str) -> str:
    import pandas as pd
    R = pd.DataFrame(records)
    det = R[(R["label"] == 1) & (R["decision"])]
    if det.empty:
        return ""
    classes = [t for t in TYPE_ORDER if t in set(det["attack_type"]) | set(det["attribution"])]
    classes += ["general"] if "general" in set(det["attribution"]) else []
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for _, r in det.iterrows():
        if r["attack_type"] in classes and r["attribution"] in classes:
            cm[classes.index(r["attack_type"]), classes.index(r["attribution"])] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Attributed type"); ax.set_ylabel("True type")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Attack-type attribution (detected injected, test)")
    return _save(fig, out_dir, "fig_confusion.png")


def fig_quant_compare(qc: dict, out_dir: str) -> str:
    """qc: {reference, per_layer_cos: {layer: cos}, head_corr: {...},
           pooled_r_corr, chosen_L: int|None}"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    pl = qc.get("per_layer_cos", {})
    if pl:
        layers = sorted(pl, key=int)
        vals = [pl[l] for l in layers]
        axes[0].bar([str(l) for l in layers], vals, color="#4C72B0")
        axes[0].axhline(0.9, color="red", lw=0.8, ls="--", label="0.9 criterion")
        if qc.get("chosen_L") is not None:
            axes[0].axvline(str(qc["chosen_L"]), color="green", lw=1.2,
                            ls=":")
        axes[0].set_ylabel(f"mean cos sim vs {qc.get('reference', 'ref')}")
        axes[0].set_title("Hidden-state fidelity per layer")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(fontsize=8)
    if qc.get("pooled_r_corr") is not None:
        axes[1].bar(["pooled R (top head)"], [qc["pooled_r_corr"]],
                    color="#DD8452")
        axes[1].axhline(0.9, color="red", lw=0.8, ls="--")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Pearson corr")
        axes[1].set_title(f"R(fp16-ref) vs R(4-bit)")
    return _save(fig, out_dir, "fig_quant_compare.png")


def fig_latency(lat: dict, out_dir: str) -> str:
    keys = [k for k in ("single_specialist_ms", "five_specialists_ms",
                        "forward_plus_1_ms", "forward_plus_5_ms")
            if isinstance(lat.get(k), (int, float))]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar([k.replace("_", " ") for k in keys], [lat[k] for k in keys],
           color="#4C72B0")
    for i, k in enumerate(keys):
        ax.text(i, lat[k], f"{lat[k]:.2f} ms", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("ms per sample")
    if lat.get("forward_overhead_pct") is not None:
        title = (f"Latency (forward+5 vs forward+1: "
                 f"{lat['forward_overhead_pct']:.3f}% overhead)")
    else:
        title = f"Latency (scoring extra {lat.get('scoring_extra_ms', 0):.3f} ms)"
    ax.set_title(title)
    return _save(fig, out_dir, "fig_latency.png")


def fig_scores_hist(records, out_dir: str) -> str:
    import pandas as pd
    R = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(R["s_max"].min() - 0.1, R["s_max"].max() + 0.1, 40)
    parts = []
    if (R["label"] == 0).any():
        parts.append(ax.hist(R.loc[R["label"] == 0, "s_max"], bins=bins,
                             alpha=0.7, label="clean", color="#CCCCCC"))
    for t in TYPE_ORDER:
        if ((R["label"] == 1) & (R["attack_type"] == t)).any():
            parts.append(ax.hist(R.loc[(R["label"] == 1) & (R["attack_type"] == t),
                                       "s_max"], bins=bins, alpha=0.7,
                                 label=t, color=TYPE_COLORS[t]))
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlabel("meta score (max specialist score)")
    ax.set_ylabel("count")
    ax.set_title("Score distributions (test)")
    return _save(fig, out_dir, "fig_scores_hist.png")
