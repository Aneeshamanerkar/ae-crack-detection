#!/usr/bin/env python
"""Controlled experiments on crack-versus-noise discrimination.

Each condition changes one aspect of training relative to the baseline and is
evaluated on the same dataset (config/default_new.yaml), the same five folds,
and training seed 0:

  baseline     15 epochs, final weights (as in rerun_full_library)
  best_epoch   weights of the epoch with the lowest loss on an inner
               validation set carved out of the training chunks only
  augment      training-only gain / noise / frequency-mask augmentation
  both         best_epoch + augment

Besides the frame-level metrics, every injected event with frames in a
validation fold is given one event-level label, either by majority vote of its
frames' non-background predictions or by its mean class probabilities
("missed" if all of its frames were predicted background). Per-frame
validation probabilities are saved next to each result (.npz) so that
ensembles and ROC analyses can be computed afterwards.

    python -m scripts.discrimination_experiments --condition baseline --threads 2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch
import yaml

from scripts.constants import CLASS_BACKGROUND, CLASS_CRACK, CLASS_NOISE, PROJECT_ROOT
from scripts.prepare_data import event_library_paths, prepare_dataset
from scripts.cross_validation import stratified_kfold_cv
from models import build_model
from scripts.augmentation_experiment import EVENT_LEN, MODEL_CFGS, simulate_placements

CONFIG_PATH = PROJECT_ROOT / "config" / "default_new.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "discrimination"
CONDITIONS = {
    "baseline": {"inner_val_frac": 0.0, "augment": False},
    "best_epoch": {"inner_val_frac": 0.15, "augment": False},
    "augment": {"inner_val_frac": 0.0, "augment": True},
    "both": {"inner_val_frac": 0.15, "augment": True},
}


def frame_event_ids(config: dict, labels: np.ndarray) -> tuple[np.ndarray, list]:
    """Per frame, the injected event the frame's label comes from (-1 for
    background frames), and the placement list ``(class, lib_idx, start)``.

    A frame is labelled with the highest-priority class present in its analysis
    window (see ``align_labels_to_frames``); it is attributed to the event of
    that class with the most samples in the window, so the attribution follows
    the same rule as the label.
    """
    ds, stft = config["dataset"], config["stft"]
    paths = event_library_paths(config)
    sizes = {c: len(p) for c, p in paths.items()}
    placements = simulate_placements(ds["n_samples"], sizes, ds["n_per_class"], ds["seed"],
                                     ds.get("replace", True))
    event_cls = np.array([c for c, _i, _s in placements])
    instance = np.full(ds["n_samples"], -1, dtype=np.int64)
    for k, (_c, _i, start) in enumerate(placements):
        instance[start:start + EVENT_LEN] = k
    hop, win = stft["hop_length"], stft["win_length"]
    ids = np.full(len(labels), -1, dtype=np.int64)
    for f in np.nonzero(labels != CLASS_BACKGROUND)[0]:
        s = max(0, f * hop - win // 2)
        w = instance[s:s + win]
        w = w[w >= 0]
        w = w[event_cls[w] == labels[f]]
        ids[f] = np.bincount(w).argmax()
    return ids, placements


def event_decisions(folds: list[dict], ids: np.ndarray, placements: list,
                    chunk_size: int) -> list[dict]:
    """One record per injected event with frames in a validation fold.

    ``folds`` holds, per fold, ``val_chunk_idx`` and ``y_prob`` (frames x 3).
    An event cut by a fold boundary is scored once, in the fold holding most of
    its frames. For each event: its true class, whether it was detected (any
    frame predicted non-background), the majority-vote class among its
    non-background frame predictions, and ``crack_score``, the mean crack
    probability over its frames divided by the mean crack-plus-noise probability.
    """
    per_fold = []
    for fd in folds:
        frames = (np.asarray(fd["val_chunk_idx"])[:, None] * chunk_size
                  + np.arange(chunk_size)).reshape(-1)
        per_fold.append(ids[frames])
    home = {}
    for f, ev in enumerate(per_fold):
        for k, n in zip(*np.unique(ev[ev >= 0], return_counts=True)):
            if n > home.get(k, (-1, 0))[1]:
                home[k] = (f, n)
    records = []
    for f, (fd, ev) in enumerate(zip(folds, per_fold)):
        prob = np.asarray(fd["y_prob"])
        pred = prob.argmax(axis=1)
        for k in np.unique(ev[ev >= 0]):
            if home[k][0] != f:
                continue
            sel = ev == k
            p = pred[sel]
            nb = p[p != CLASS_BACKGROUND]
            n_c, n_n = int((nb == CLASS_CRACK).sum()), int((nb == CLASS_NOISE).sum())
            mc, mn = prob[sel, CLASS_CRACK].mean(), prob[sel, CLASS_NOISE].mean()
            records.append({
                "event": int(k), "true": int(placements[k][0]),
                "detected": bool(len(nb)),
                "vote": CLASS_CRACK if n_c > n_n else CLASS_NOISE if n_n > n_c else None,
                "crack_score": float(mc / (mc + mn)),
            })
    return records


def event_scores(records: list[dict], rule: str) -> dict:
    """Event-level confusion (rows true crack/noise; columns predicted
    crack / noise / missed) and scores, for ``rule`` = "vote" (majority of
    frame predictions; a tie counts as wrong) or "prob" (crack if crack_score
    > 0.5). Undetected events are "missed" under both rules."""
    cm = np.zeros((2, 3), dtype=int)
    for r in records:
        row = 0 if r["true"] == CLASS_CRACK else 1
        if not r["detected"]:
            col = 2
        elif rule == "vote":
            col = {CLASS_CRACK: 0, CLASS_NOISE: 1}.get(r["vote"], 1 - row)  # tie -> wrong
        else:
            col = 0 if r["crack_score"] > 0.5 else 1
        cm[row, col] += 1
    detected = cm[:, :2]
    correct = cm[0, 0] + cm[1, 1]
    prec_c = cm[0, 0] / max(detected[:, 0].sum(), 1)
    rec_c = cm[0, 0] / max(cm[0].sum(), 1)
    prec_n = cm[1, 1] / max(detected[:, 1].sum(), 1)
    rec_n = cm[1, 1] / max(cm[1].sum(), 1)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    return {
        "confusion_rows_true_crack_noise_cols_pred_crack_noise_missed": cm.tolist(),
        "n_events": int(cm.sum()),
        "accuracy": float(correct / cm.sum()),
        "discrimination_accuracy": float(correct / max(detected.sum(), 1)),
        "crack_f1": float(f1(prec_c, rec_c)),
        "noise_f1": float(f1(prec_n, rec_n)),
        "macro_f1": float((f1(prec_c, rec_c) + f1(prec_n, rec_n)) / 2),
        "detected_fraction": float(detected.sum() / cm.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--models", nargs="+", default=["lstm", "cnn_lstm", "transformer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--lstm-hidden", type=int, default=None, help="LSTM units per direction (default 128)")
    ap.add_argument("--lstm-layers", type=int, default=None, help="number of LSTM layers (default 2)")
    ap.add_argument("--tag", default="", help="suffix for the output file names, e.g. _h32")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    config_path = pathlib.Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    config["_config_dir"] = str(config_path.parent)
    ds, tr = config["dataset"], config["training"]
    spectrogram, labels = prepare_dataset(config)          # from cache
    ids, placements = frame_event_ids(config, labels)
    train_cfg = {"num_epochs": args.epochs or tr["num_epochs"], "learning_rate": tr["learning_rate"],
                 "batch_size": tr["batch_size"], "chunk_size": ds["chunk_size"]}

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        print(f"\n##### {args.condition} | {model} | seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        t0 = time.time()
        cfg = dict(MODEL_CFGS[model])
        if model == "lstm" and args.lstm_hidden:
            cfg["hidden_size"] = args.lstm_hidden
        if model == "lstm" and args.lstm_layers:
            cfg["num_layers"] = args.lstm_layers
        n_params = sum(p.numel() for p in build_model(model, n_freq=spectrogram.shape[0], n_classes=3,
                                                      **cfg).parameters() if p.requires_grad)
        r = stratified_kfold_cv(model, cfg, (spectrogram, labels), k=5,
                                seed=ds["seed"], train_cfg=train_cfg, save_checkpoints=False,
                                **CONDITIONS[args.condition])
        cm = r.aggregated_cm
        ev = cm[1:, 1:]
        tp, fn, fp = cm[1:, 1:].sum(), cm[1:, 0].sum(), cm[0, 1:].sum()
        det_p, det_r = tp / (tp + fp), tp / (tp + fn)
        out = {
            "condition": args.condition, "model": model, "training_seed": args.seed,
            "seconds": round(time.time() - t0, 1),
            "macro_f1_mean": r.macro_f1_mean,
            "macro_f1_per_fold": [float(fr.f1.mean()) for fr in r.folds],
            "f1_mean": r.f1_mean.tolist(), "f1_std": r.f1_std.tolist(),
            "precision_mean": r.precision_mean.tolist(), "recall_mean": r.recall_mean.tolist(),
            "aggregated_cm": cm.tolist(),
            "frame_detection_f1": float(2 * det_p * det_r / (det_p + det_r)),
            "frame_discrimination_accuracy": float(np.trace(ev) / ev.sum()),
            "best_epochs": [fr.best_epoch for fr in r.folds],
            "model_config": cfg, "n_parameters": int(n_params),
            "final_train_loss_per_fold": [float(fr.train_losses[-1]) for fr in r.folds],
            "final_val_loss_per_fold": [float(fr.val_losses[-1]) for fr in r.folds],
            "train_losses_per_fold": [list(map(float, fr.train_losses)) for fr in r.folds],
            "val_losses_per_fold": [list(map(float, fr.val_losses)) for fr in r.folds],
            # same per-model fields as outputs/rerun_full_library/results.json
            "macro_f1_std": float(np.std([fr.f1.mean() for fr in r.folds], ddof=1)),
            "accuracy_per_fold": [float(fr.accuracy) for fr in r.folds],
            "precision_std": np.std([fr.precision for fr in r.folds], axis=0, ddof=1).tolist(),
            "recall_std": np.std([fr.recall for fr in r.folds], axis=0, ddof=1).tolist(),
            "per_fold_cm": [np.asarray(fr.cm, dtype=int).tolist() for fr in r.folds],
            "n_train_per_fold": [int(fr.n_train) for fr in r.folds],
            "n_val_per_fold": [int(fr.n_val) for fr in r.folds],
        }
        out["f1_std"] = np.std([fr.f1 for fr in r.folds], axis=0, ddof=1).tolist()
        folds = [{"val_chunk_idx": fr.val_chunk_idx, "y_prob": fr.y_prob} for fr in r.folds]
        records = event_decisions(folds, ids, placements, ds["chunk_size"])
        out["event_level"] = event_scores(records, "vote")
        out["event_level_prob"] = event_scores(records, "prob")
        path = out_dir / f"{args.condition}_{model}{args.tag}_seed{args.seed}.json"
        path.write_text(json.dumps(out, indent=2))
        np.savez_compressed(
            path.with_suffix(".npz"),
            **{f"fold{i}_val_chunk_idx": fd["val_chunk_idx"] for i, fd in enumerate(folds)},
            **{f"fold{i}_y_prob": fd["y_prob"] for i, fd in enumerate(folds)},
        )
        e = out["event_level_prob"]
        print(f"saved {path.name}: macro {out['macro_f1_mean']:.4f} | frame disc "
              f"{out['frame_discrimination_accuracy']:.3f} | event disc (vote) "
              f"{out['event_level']['discrimination_accuracy']:.3f} (prob) "
              f"{e['discrimination_accuracy']:.3f} | ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
