#!/usr/bin/env python
"""Redraw the Chapter 4 stream figures from the saved checkpoints in
``outputs/checkpoints_rebuilt/`` (written by ``rerun_stream_eval``), without
retraining. Output goes straight to ``outputs/figures/finalPNG/``.

Figure 4.4 fix (supervisor point 27): crack and noise predictions are drawn
in SEPARATE, non-overlapping y-bands (one thin lane per class) at the foot
of the waveform panel, so the noise shading can never cover the crack
shading. Crack is also drawn last with a higher zorder. Hues, figsize and
file names are unchanged. The recording duration is derived from the data
(``inf["waveform"]`` / ``inf["time_s"]``); no 120 s / 240,003,072 constant
or hard-coded xlim remains (the stream reader now returns ~60 s per
channel: 120,001,536 samples at 2 MHz).

    python -m scripts.redraw_stream_figures
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from scripts.constants import CLASS_NAMES, NUM_CLASSES, PROJECT_ROOT
from scripts.evaluate_stream import predict_full_stream
from scripts.rerun_stream_eval import CKPT_DIR, CONFIG_PATH, OUT_DIR, model_cfgs
from models import build_model

FINAL = PROJECT_ROOT / "outputs" / "figures" / "finalPNG"
LABEL = {"lstm": "Bidirectional LSTM", "cnn_lstm": "CNN-LSTM", "transformer": "Transformer encoder"}
N_FREQ = 133
# Same hues as scripts.rerun_stream_eval.CLASS_COLORS (unchanged).
CLASS_COLORS = {0: "lightgrey", 1: "crimson", 2: "steelblue"}


def plot_stream(inf: dict, title: str, path: pathlib.Path) -> None:
    """Waveform + per-frame probabilities, sized for textwidth (16 cm).

    Top panel: waveform with crack/noise predictions in SEPARATE thin lanes
    stacked at the foot of the axes (non-overlapping y-ranges), instead of
    two full-height ``fill_between`` shadings where noise (drawn second)
    covered crack. Crack is drawn last with the higher zorder. Bottom panel:
    per-frame class probabilities (unchanged). The x-limits are taken from
    the data (waveform sample times); nothing is hard-coded.
    """
    with plt.rc_context({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13}):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
        ax = axes[0]
        wav = np.asarray(inf["waveform"])
        t = np.asarray(inf["time_s"])
        preds = np.asarray(inf["predictions"])
        ax.plot(wav[:, 0], wav[:, 1], color="#333333", lw=0.3, alpha=0.8)
        lo, hi = float(wav[:, 1].min()), float(wav[:, 1].max())
        span = hi - lo if hi > lo else 1.0
        # Two non-overlapping lanes at the foot of the waveform axes.
        lane_h = 0.13 * span
        crack_y0, crack_y1 = lo, lo + lane_h
        noise_y0, noise_y1 = lo + 1.25 * lane_h, lo + 2.25 * lane_h
        mask_crack = preds == 1
        mask_noise = preds == 2
        # Noise lane first (lower zorder), crack lane last (higher zorder);
        # lanes never overlap in y, so neither class can cover the other.
        if bool(mask_noise.any()):
            ax.fill_between(t, noise_y0, noise_y1, where=mask_noise,
                            color=CLASS_COLORS[2], alpha=0.55,
                            label=CLASS_NAMES[2], zorder=2)
        if bool(mask_crack.any()):
            ax.fill_between(t, crack_y0, crack_y1, where=mask_crack,
                            color=CLASS_COLORS[1], alpha=0.55,
                            label=CLASS_NAMES[1], zorder=3)
        ax.axhline(crack_y1, color=CLASS_COLORS[1], lw=0.8, alpha=0.6, zorder=4)
        ax.axhline(noise_y1, color=CLASS_COLORS[2], lw=0.8, alpha=0.6, zorder=4)
        ax.set_ylabel("Amplitude (V)")
        ax.set_title("Waveform with predicted regions (crack / noise lanes)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        # Duration / x-limits derived from the data (no hard-coded xlim).
        ax.set_xlim(float(wav[0, 0]), float(wav[-1, 0]))

        ax = axes[1]
        for cls in range(NUM_CLASSES):
            ax.plot(t, np.asarray(inf["probabilities"])[:, cls], label=CLASS_NAMES[cls],
                    color=CLASS_COLORS[cls], lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability")
        ax.set_title("Per-frame class probabilities")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(wav[0, 0]), float(wav[-1, 0]))

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    stream_path = str((CONFIG_PATH.parent / config["paths"]["stream_folder"]
                       / config["paths"]["stream_file"]).resolve())
    recorded = json.loads((OUT_DIR / "stream_summary.json").read_text())
    cfgs = model_cfgs(config["model"])
    for name in ("lstm", "cnn_lstm", "transformer"):
        model = build_model(name, n_freq=N_FREQ, n_classes=NUM_CLASSES, **cfgs[name])
        model.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location="cpu",
                                         weights_only=True))
        inf = predict_full_stream(model, stream_path, chunk_size=config["dataset"]["chunk_size"],
                                  device="cpu")
        crack = int((inf["predictions"] == 1).sum())
        # must reproduce the numbers already reported from the same checkpoint
        assert crack == recorded[name]["crack_frames"], (name, crack, recorded[name]["crack_frames"])
        # Duration derived from the data: sample-time extent of the returned
        # stream (reader now yields ~60 s per channel, not 120 s).
        dur_s = float(np.asarray(inf["waveform"])[-1, 0] - np.asarray(inf["waveform"])[0, 0])
        plot_stream(inf, f"{LABEL[name]}, unmodified {dur_s:.0f} s recording",
                    FINAL / f"ch4_stream_{name}.png")
        print(f"{name}: {crack} crack frames (matches stream_summary.json); "
              f"stream duration {dur_s:.2f} s; figure written")


if __name__ == "__main__":
    main()
