#!/usr/bin/env python
"""Screen every extracted event of both experiments with a trained model.

Only the 200 expert-labelled events carry labels; the other events of
2022_n/2V and 2024_n/1.3V are unlabelled. This script applies a model trained
on the labelled events (checkpoint from ``rerun_stream_eval``) to all of them:
events are added to the same quiet background segment that the training
streams use, 150 per stream at evenly spaced positions, and each event gets
one decision from the frames that cover it (as in ``event_level_analysis``):

* detected: at least one of its frames is predicted crack or noise;
* crack score: mean crack probability / (mean crack + mean noise probability);
* class: crack if detected and crack score > 0.5, noise if detected otherwise.

Waveforms are read from the event cache written by ``build_screening_cache``
from the same directories.

    python -m scripts.screen_full_dataset --model lstm
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.constants import CLASS_BACKGROUND, CLASS_CRACK, CLASS_NOISE, NUM_CLASSES, PROJECT_ROOT
from scripts.transforms import compute_spectrogram, log_magnitude
from scripts.rerun_stream_eval import CKPT_DIR, model_cfgs
from models import build_model

CONFIG_PATH = PROJECT_ROOT / "config" / "default_new.yaml"
CACHE = PROJECT_ROOT / "outputs" / "cache" / "screening"   # built by scripts.build_screening_cache
BACKGROUND = PROJECT_ROOT / "outputs" / "augmentation_rerun" / "background.npy"
OUT = PROJECT_ROOT / "outputs" / "final_labels"
EVENT_LEN, PER_STREAM, USABLE = 4096, 150, 1_585_000
N_EVENTS = 141_099            # 2022_n (140,864) + 2024_n (235); the aug rows follow and are skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "cnn_lstm", "transformer"])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    stft, cs = config["stft"], config["dataset"]["chunk_size"]
    meta = pd.read_csv(CACHE / "event_metadata.csv", usecols=["index", "filename", "specimen", "source_dir"])
    wave = np.load(CACHE / "event_waveforms.npy", mmap_mode="r")
    assert list(meta.source_dir.iloc[:N_EVENTS].unique()) == ["2022_n", "2024_n"]
    background = np.load(BACKGROUND).astype(np.float64)

    model = build_model(args.model, n_freq=stft["n_fft"] // 2 + 1, n_classes=NUM_CLASSES,
                        **model_cfgs(config["model"])[args.model])
    model.load_state_dict(torch.load(CKPT_DIR / f"{args.model}_best.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    spacing = USABLE // PER_STREAM
    starts = np.arange(PER_STREAM) * spacing + (spacing - EVENT_LEN) // 2
    hop = stft["hop_length"]
    score = np.zeros(N_EVENTS, dtype=np.float32)
    detected = np.zeros(N_EVENTS, dtype=bool)
    for first in range(0, N_EVENTS, PER_STREAM):
        ids = np.arange(first, min(first + PER_STREAM, N_EVENTS))
        stream = background.copy()
        for k, i in enumerate(ids):
            stream[starts[k]:starts[k] + EVENT_LEN] += wave[i]
        _, _, sxx = compute_spectrogram(stream, fs=config["signal"]["fs"], n_fft=stft["n_fft"],
                                        hop_length=hop, win_length=stft["win_length"],
                                        power=stft["power"])
        spec = log_magnitude(sxx)                                   # (F, T)
        n_chunks = spec.shape[1] // cs
        x = torch.tensor(spec[:, :n_chunks * cs].reshape(spec.shape[0], n_chunks, cs)
                         .transpose(1, 2, 0), dtype=torch.float32)  # (N, T, F)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=-1).reshape(-1, NUM_CLASSES).numpy()
        pred = prob.argmax(axis=1)
        for k, i in enumerate(ids):
            f0, f1 = starts[k] // hop, (starts[k] + EVENT_LEN) // hop + 1
            pc, pn = prob[f0:f1, CLASS_CRACK].mean(), prob[f0:f1, CLASS_NOISE].mean()
            score[i] = pc / (pc + pn)
            detected[i] = bool((pred[f0:f1] != CLASS_BACKGROUND).any())
        if first % (PER_STREAM * 100) == 0:
            print(f"  {first + len(ids):,} / {N_EVENTS:,} events", flush=True)

    cls = np.where(~detected, "missed", np.where(score > 0.5, "crack", "noise"))
    df = pd.DataFrame({"index": np.arange(N_EVENTS), "file": meta.filename.iloc[:N_EVENTS].values,
                       "experiment": np.where(meta.source_dir.iloc[:N_EVENTS] == "2022_n", "2022", "2024"),
                       "specimen": meta.specimen.iloc[:N_EVENTS].values,
                       "crack_score": score.round(4), "predicted": cls})
    df.to_csv(OUT / f"screening_{args.model}.csv", index=False)

    labelled = {}
    for lab in ("crack", "noise"):
        for f in config["events"][f"{lab}_files"]:
            labelled[pathlib.Path(f).name] = lab
    lab_df = df[df.file.isin(labelled)]
    agree = float((lab_df.predicted == lab_df.file.map(labelled)).mean())
    summary = {
        "model": args.model, "events": int(N_EVENTS),
        "predicted": df.predicted.value_counts().to_dict(),
        "by_experiment": {e: g.predicted.value_counts().to_dict() for e, g in df.groupby("experiment")},
        "by_specimen": {s: g.predicted.value_counts().to_dict() for s, g in df.groupby("specimen")},
        "high_confidence_crack_score_above_0.9": int(((df.predicted == "crack") & (df.crack_score > 0.9)).sum()),
        "labelled_events_screened": int(len(lab_df)), "labelled_events_agreement": agree,
        "clustering_reference": {"crack": 248, "noise": 140740, "unassigned": 125},
    }
    (OUT / f"screening_{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("predicted", "by_experiment", "high_confidence_crack_score_above_0.9",
                                             "labelled_events_agreement")}, indent=1))


if __name__ == "__main__":
    main()
