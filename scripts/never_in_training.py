#!/usr/bin/env python
"""Event-level balanced accuracy on all scored events vs never-in-training events.

For each model (lstm, cnn_lstm, transformer) the 3-seed ensemble is built by
averaging per-frame validation probabilities from
``outputs/final_labels/baseline_<model>_seed{0,1,2}.npz`` (helpers imported
from ``scripts.event_level_analysis``), and each injected event is scored with
the ``prob`` rule (``crack_score = mean crack prob / (mean crack + mean noise
prob)`` over the event's frames; crack iff detected and score > 0.5), reusing
``frame_event_ids`` / ``event_decisions`` from
``scripts.discrimination_experiments``. Event positions come from
``scripts.augmentation_experiment.simulate_placements`` / ``prepare_data``
via ``frame_event_ids`` with ``config/default_new.yaml``; fold assignment is
the ``stratified_kfold_cv`` / ``chunk_stream`` split (k=5, seed 42, chunks of
256 frames), verified against the saved ``val_chunk_idx`` fold by fold.

An event scored in fold f is 'never partly in training' if none of its frames
lies in a training chunk of fold f (training chunks = all chunks minus the
fold's validation chunks).

Writes ``outputs/final_labels/never_in_training.json``.

    PYTHONPATH=$PWD /opt/anaconda3/envs/ae-crack-detection/bin/python -m scripts.never_in_training
"""

from __future__ import annotations

import json
import sys

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from scripts.constants import CLASS_CRACK, CLASS_NOISE, PROJECT_ROOT
from scripts.prepare_data import prepare_dataset
from scripts.augmentation_experiment import simulate_placements  # noqa: F401  (via frame_event_ids)
from scripts.cross_validation import chunk_stream, chunk_stratify_tag
from scripts.discrimination_experiments import event_decisions, frame_event_ids
from scripts.event_level_analysis import load_folds

CONFIG_PATH = PROJECT_ROOT / "config" / "default_new.yaml"
IN_DIR = PROJECT_ROOT / "outputs" / "final_labels"
OUT_PATH = IN_DIR / "never_in_training.json"
MODELS = ["lstm", "cnn_lstm", "transformer"]
SEEDS = [0, 1, 2]
K, CV_SEED, CHUNK = 5, 42, 256
# Ensemble 'event prob' balanced accuracy (rounded to 3 decimals) from
# event_level_analysis; all_scored must reproduce these.
EXPECTED_BALANCED = {"lstm": 0.980, "cnn_lstm": 0.973, "transformer": 0.980}


def fail(msg: str) -> None:
    raise SystemExit(f"never_in_training FAILED: {msg}")


def subset_stats(records: list[dict]) -> dict:
    n_crack = sum(1 for r in records if r["true"] == CLASS_CRACK)
    n_noise = sum(1 for r in records if r["true"] == CLASS_NOISE)
    crack_correct = sum(1 for r in records if r["true"] == CLASS_CRACK
                        and r["detected"] and r["crack_score"] > 0.5)
    noise_correct = sum(1 for r in records if r["true"] == CLASS_NOISE
                        and r["detected"] and r["crack_score"] <= 0.5)
    if not n_crack or not n_noise:
        fail(f"empty class in subset (n_crack={n_crack}, n_noise={n_noise})")
    balanced = (crack_correct / n_crack + noise_correct / n_noise) / 2
    return {"balanced_accuracy": float(balanced), "n_crack": n_crack, "n_noise": n_noise,
            "crack_correct": crack_correct, "noise_correct": noise_correct}


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["_config_dir"] = str(CONFIG_PATH.parent)
    spectrogram, labels = prepare_dataset(config)  # from cache
    ids, placements = frame_event_ids(config, labels)
    print(f"placements: {len(placements)} "
          f"({sum(1 for p in placements if p[0] == CLASS_CRACK)} crack / "
          f"{sum(1 for p in placements if p[0] == CLASS_NOISE)} noise)")

    # Fold assignment exactly as in stratified_kfold_cv (chunk_stream, k=5,
    # seed 42, 256-frame chunks); verified per fold against the saved splits.
    chunks, chunk_labels = chunk_stream(spectrogram, labels, CHUNK)
    n_chunks = len(chunks)
    tags = np.array([chunk_stratify_tag(cl) for cl in chunk_labels])
    ref_splits = list(StratifiedKFold(n_splits=K, shuffle=True,
                                      random_state=CV_SEED).split(np.arange(n_chunks), tags))

    out = {"description": "Event-level balanced accuracy of the 3-seed ensembles, "
                           "all scored events vs only events with no frame in any "
                           "training chunk of the fold in which they are scored."}
    for model in MODELS:
        runs = [load_folds(IN_DIR / f"baseline_{model}_seed{s}.npz") for s in SEEDS]
        for run in runs[1:]:
            for a, b in zip(runs[0], run):
                if not np.array_equal(a["val_chunk_idx"], b["val_chunk_idx"]):
                    fail(f"{model}: val chunks differ between seeds")
        for i, (tr_idx, va_idx) in enumerate(ref_splits):
            saved = np.asarray(runs[0][i]["val_chunk_idx"])
            if not np.array_equal(np.sort(va_idx), np.sort(saved)):
                fail(f"{model} fold {i}: saved val chunks do not match the "
                     f"stratified_kfold_cv (k=5, seed 42) split")
        ens = [{"val_chunk_idx": fd["val_chunk_idx"],
                "y_prob": np.mean([run[i]["y_prob"] for run in runs], axis=0)}
               for i, fd in enumerate(runs[0])]
        records = event_decisions(ens, ids, placements, CHUNK)

        all_stats = subset_stats(records)
        if round(all_stats["balanced_accuracy"], 3) != EXPECTED_BALANCED[model]:
            fail(f"{model}: all_scored balanced "
                 f"{all_stats['balanced_accuracy']:.6f} (rounded "
                 f"{round(all_stats['balanced_accuracy'], 3)}) != expected "
                 f"{EXPECTED_BALANCED[model]} from event_level_analysis")
        if (all_stats["n_crack"], all_stats["n_noise"]) != (75, 74):
            fail(f"{model}: all_scored counts {all_stats['n_crack']}/"
                 f"{all_stats['n_noise']} != stale-file 75/74 (placements changed?)")

        # Home fold per event (fold holding most of its frames, as in
        # event_decisions); training chunks = all chunks minus val chunks.
        val_of = {f: set(np.asarray(ens[f]["val_chunk_idx"]).tolist())
                  for f in range(K)}
        train_of = {f: set(range(n_chunks)) - val_of[f] for f in range(K)}
        per_fold = [ids[(np.asarray(fd["val_chunk_idx"])[:, None] * CHUNK
                         + np.arange(CHUNK)).reshape(-1)] for fd in ens]
        home: dict[int, tuple[int, int]] = {}
        for f, ev in enumerate(per_fold):
            for k, n in zip(*np.unique(ev[ev >= 0], return_counts=True)):
                if n > home.get(int(k), (-1, 0))[1]:
                    home[int(k)] = (f, int(n))
        home_fold = {k: v[0] for k, v in home.items()}
        never_records = []
        for r in records:
            k = r["event"]
            frames = np.nonzero(ids == k)[0]
            frame_chunks = set((frames[frames < n_chunks * CHUNK] // CHUNK).tolist())
            if not frame_chunks & train_of[home_fold[k]]:
                never_records.append(r)
        never_stats = subset_stats(never_records)
        if (never_stats["n_crack"], never_stats["n_noise"]) != (60, 62):
            fail(f"{model}: never_partly_in_training counts "
                 f"{never_stats['n_crack']}/{never_stats['n_noise']} != stale-file "
                 f"60/62 (placements changed?)")

        out[model] = {"all_scored": all_stats,
                      "never_partly_in_training": never_stats}
        print(f"{model}: all {all_stats['n_crack']}/{all_stats['n_noise']} "
              f"bal {all_stats['balanced_accuracy']:.6f} | never-in-training "
              f"{never_stats['n_crack']}/{never_stats['n_noise']} "
              f"bal {never_stats['balanced_accuracy']:.6f}")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
