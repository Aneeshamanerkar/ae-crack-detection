#!/usr/bin/env python
"""Figures for Chapter 5 (Discussion).

Derives every value from the recorded results rather than restating them, so
the figures cannot drift from Chapter 4:

  ch5_detection_vs_discrimination.png  the task split of Section 1.2, measured
  ch5_crack_reuse.png                  crack-waveform reuse across fold splits
  ch5_synthetic_real_gap.png           crack under cross-validation vs on the real stream

    python -m scripts.make_ch5_figures
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.constants import FS, HOP_LENGTH, PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "figures" / "finalPNG"
RES = PROJECT_ROOT / "outputs" / "rerun_full_library"

# Validated categorical slots 1 and 2 (see dataviz palette).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
LABEL = {"lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "transformer": "Transformer"}
ORDER = ["lstm", "cnn_lstm", "transformer"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=13)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    ax.set_axisbelow(True)


def per_class_f1(cm: np.ndarray, c: int) -> float:
    """Standard one-vs-rest F1 for class ``c`` from the full 3x3 matrix."""
    tp = float(cm[c, c])
    fp = float(cm[:, c].sum() - tp)
    fn = float(cm[c, :].sum() - tp)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def decompose(cm: np.ndarray) -> tuple[float, float]:
    """Return (detection F1, discrimination macro-F1) for one confusion matrix.

    Rows are true labels, columns predicted labels; 0 = background,
    1 = crack, 2 = noise (see ``outputs/rerun_full_library/results.json``
    ``aggregated_cm``).

    Detection collapses crack and noise into a single event class and scores
    event-vs-background as a binary F1: TP = cm[1:, 1:].sum(),
    FP = cm[0, 1:].sum(), FN = cm[1:, 0].sum(), P = TP/(TP+FP),
    R = TP/(TP+FN), F1_det = 2PR/(P+R).

    Discrimination is the macro F1 over the two event classes computed from
    the SAME full matrix (one-vs-rest per class, then averaged):
    F1_disc = (F1_crack + F1_noise)/2, where for each c in {1, 2},
    TP_c = cm[c, c], FP_c = cm[:, c].sum() - TP_c,
    FN_c = cm[c, :].sum() - TP_c. Both halves are therefore F1 scores, so
    detection error = 1 - F1_det and discrimination error = 1 - F1_disc
    are directly comparable (previously discrimination used accuracy).
    """
    cm = np.asarray(cm, dtype=float)
    tp = cm[1:, 1:].sum()
    fn = cm[1:, 0].sum()
    fp = cm[0, 1:].sum()
    p, r = tp / (tp + fp), tp / (tp + fn)
    det_f1 = 2 * p * r / (p + r)
    disc_f1 = (per_class_f1(cm, 1) + per_class_f1(cm, 2)) / 2.0
    return det_f1, disc_f1


def fig_detection_vs_discrimination(models: dict) -> None:
    """Error rates (both 1 - F1) on a log axis: the two halves differ by
    orders of magnitude, which a 0-1 score axis would flatten into three
    identical-looking bars. Both halves now use the SAME metric (F1), so
    detection error = 1 - detection-F1 and discrimination error =
    1 - macro-F1(crack, noise); the marker is labelled with the F1 score."""
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ys = np.arange(len(ORDER))[::-1]

    for y, key in zip(ys, ORDER):
        cm = np.array(models[key]["aggregated_cm"])
        det, disc = decompose(cm)  # disc is macro-F1 over crack/noise
        for off, err, color, name in (
            (0.16, 1 - det, BLUE, "Detection error (1 \u2212 F1)"),
            (-0.16, 1 - disc, ORANGE, "Discrimination error (1 \u2212 F1)"),
        ):
            err = max(err, 4e-5)          # keep a visible stem at exactly zero error
            ax.plot([3e-5, err], [y + off, y + off], color=color, lw=2, zorder=2)
            ax.plot(err, y + off, "o", ms=11, color=color, zorder=3,
                    label=name if y == ys[0] else None)
            ax.text(err * 1.35, y + off, f"{1 - (err if err > 4e-5 else 0):.4f}",
                    va="center", ha="left", fontsize=12.3, color=INK)

    ax.set_xscale("log")
    ax.set_xlim(3e-5, 0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=14.5)
    ax.set_xlabel("Error rate = 1 \u2212 F1 (log scale) \u2014 marker labelled with the F1 score",
                  fontsize=13, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=13, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ch5_detection_vs_discrimination.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_crack_reuse(reuse: dict) -> None:
    """Validation crack waveforms also present in training, split by cause:
    the same library waveform injected twice (drawing with replacement), or
    one injected event cut by a chunk boundary between training and validation."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    folds = [f["crack"] for f in reuse["folds"]]
    x = np.arange(1, len(folds) + 1)
    total = np.array([f["val_sources"] for f in folds])
    rep = np.array([f["via_repeat"] for f in folds])
    strad = np.array([f["either"] - f["via_repeat"] for f in folds])
    w = 0.38

    ax.bar(x - w / 2, total, w, color=BLUE, label="Crack waveforms in validation chunks", zorder=2)
    ax.bar(x + w / 2, rep, w, color=ORANGE, zorder=2,
           label="…also in training: same waveform injected twice")
    ax.bar(x + w / 2, strad, w, bottom=rep, color=ORANGE, alpha=0.45, zorder=2,
           label="…also in training: event split at a chunk boundary")
    for xi, t, r, st in zip(x, total, rep, strad):
        ax.text(xi - w / 2, t + 0.4, str(t), ha="center", fontsize=12.3, color=INK)
        ax.text(xi + w / 2, r + st + 0.4, f"{r}+{st}", ha="center", fontsize=12.3, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x], fontsize=14.5)
    ax.set_ylabel("Distinct crack waveforms", fontsize=13, color=MUTED)
    ax.set_ylim(0, max(total) * 1.55)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ch5_crack_reuse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_real_gap(models: dict, stream: dict) -> None:
    """Two panels, not two y-axes: an F1 and a frame percentage share no scale."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    x = np.arange(len(ORDER))
    names = [LABEL[k] for k in ORDER]

    ax = axes[0]
    f1 = [models[k]["f1_mean"][1] for k in ORDER]
    ax.bar(x, f1, 0.52, color=BLUE, zorder=2)
    for xi, v in zip(x, f1):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=13, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Crack F1", fontsize=13, color=MUTED)
    ax.set_title("On the synthetic stream\n(five-fold cross-validation)",
                 fontsize=13.8, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)

    ax = axes[1]
    n_frames = stream[ORDER[0]]["n_frames"]
    # Stream duration derived from the data (frames * hop / sampling rate),
    # not hard-coded: the reader now yields ~60 s per channel, not 120 s.
    dur_s = float(n_frames) * float(HOP_LENGTH) / float(FS)
    pct = [100 * stream[k]["predicted_fraction"]["Crack"] for k in ORDER]
    ax.bar(x, pct, 0.52, color=ORANGE, zorder=2)
    top = max(max(pct), 1.0) * 1.2
    for xi, v in zip(x, pct):
        ax.text(xi, v + top * 0.02, f"{v:.3f}%" if v < 1 else f"{v:.1f}%",
                ha="center", fontsize=13, color=INK)
    ax.set_ylim(0, top)
    ax.set_ylabel("Frames predicted crack (%)", fontsize=13, color=MUTED)
    ax.set_title(f"On the real {dur_s:.0f} s recording\n(out of {n_frames:,} frames)",
                 fontsize=13.8, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12.5)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ch5_synthetic_real_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((RES / "results.json").read_text())
    stream = json.loads((RES / "stream_summary.json").read_text())
    models, comp = res["models"], res["composition"]

    for key in ORDER:
        det, disc = decompose(np.array(models[key]["aggregated_cm"]))
        det_err, disc_err = 1 - det, 1 - disc
        ratio = disc_err / det_err if det_err > 0 else float("nan")
        print(f"{LABEL[key]:12s} detection F1 {det:.4f} (err {det_err:.4f})   "
              f"discrimination macro-F1 {disc:.4f} (err {disc_err:.4f})   "
              f"ratio disc/det {ratio:.2f}")

    fig_detection_vs_discrimination(models)
    fig_crack_reuse(json.loads((RES / "crack_reuse.json").read_text()))
    fig_synthetic_real_gap(models, stream)
    print(f"\nWrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
