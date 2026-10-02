#!/usr/bin/env python
"""Event-level analysis of saved cross-validation probabilities.

For one condition of ``scripts.discrimination_experiments`` and each model, it
reads the per-frame validation probabilities of every training seed (``.npz``)
and reports:

* per seed: frame-level and event-level scores (majority vote and mean
  probability);
* the seed ensemble: per-frame probabilities averaged over the seeds, scored
  the same way;
* the ROC curve of the event-level crack score, its area (AUC), and the
  crack-recall operating points read off that curve. These are descriptive
  points on the curve, not thresholds tuned for deployment.

    python -m scripts.event_level_analysis --condition best_epoch --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import re
import json
import pathlib

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.constants import CLASS_BACKGROUND, CLASS_CRACK, CLASS_NOISE, PROJECT_ROOT
from scripts.prepare_data import event_library_paths, prepare_dataset
from scripts.discrimination_experiments import (
    CONFIG_PATH, OUT_DIR, event_decisions, event_scores, frame_event_ids,
)

LABEL = {"lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "transformer": "Transformer"}
TARGET_RECALLS = (0.80, 0.85, 0.90)
TOP_FRACTIONS = (0.10, 0.20, 0.30)


def load_folds(path: pathlib.Path) -> list[dict]:
    z = np.load(path)
    n = len([k for k in z.files if k.endswith("_y_prob")])
    return [{"val_chunk_idx": z[f"fold{i}_val_chunk_idx"], "y_prob": z[f"fold{i}_y_prob"]}
            for i in range(n)]


def frame_scores(folds: list[dict], labels: np.ndarray, chunk_size: int) -> dict:
    cm = np.zeros((3, 3), dtype=int)
    for fd in folds:
        frames = (np.asarray(fd["val_chunk_idx"])[:, None] * chunk_size
                  + np.arange(chunk_size)).reshape(-1)
        np.add.at(cm, (labels[frames], fd["y_prob"].argmax(axis=1)), 1)
    ev = cm[1:, 1:]
    tp, fn, fp = ev.sum(), cm[1:, 0].sum(), cm[0, 1:].sum()
    p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = []
    for c in range(3):
        pc, rc = cm[c, c] / max(cm[:, c].sum(), 1), cm[c, c] / max(cm[c].sum(), 1)
        f1.append(2 * pc * rc / (pc + rc) if pc + rc else 0.0)
    return {"aggregated_cm": cm.tolist(), "pooled_f1": f1, "pooled_macro_f1": float(np.mean(f1)),
            "detection_f1": float(2 * p * r / (p + r)),
            "discrimination_accuracy": float(np.trace(ev) / ev.sum())}


def roc_points(records: list[dict], experiment_of: dict[int, str] | None = None) -> dict:
    y = np.array([r["true"] == CLASS_CRACK for r in records], dtype=int)
    score = np.array([r["crack_score"] for r in records])
    fpr, tpr, thr = roc_curve(y, score)
    points = {}
    for target in TARGET_RECALLS:
        i = int(np.argmax(tpr >= target))
        points[f"crack_recall_{target:.2f}"] = {
            "crack_recall": float(tpr[i]), "noise_flagged_as_crack": float(fpr[i]),
            "threshold": float(thr[i]),
        }
    # The confident end of the ranking: among the events the model finds most
    # crack-like, what fraction are cracks, and what share of all cracks is that?
    order = np.argsort(-score)
    top = {}
    for frac in TOP_FRACTIONS:
        n = max(1, int(round(frac * len(score))))
        sel = y[order[:n]]
        entry = {"n_events": n, "crack_precision": float(sel.mean()),
                 "share_of_all_cracks": float(sel.sum() / y.sum())}
        if experiment_of is not None:
            entry["from_2024"] = int(sum(experiment_of[records[i]["event"]].startswith("2024")
                                         for i in order[:n]))
        top[f"top_{int(frac * 100)}pct"] = entry
    return {"auc": float(roc_auc_score(y, score)), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "average_precision": float(average_precision_score(y, score)),
            "crack_prevalence": float(y.mean()),
            "operating_points": points, "most_crack_like": top}


def by_experiment(records: list[dict], experiment_of: dict[int, str]) -> dict:
    """Crack-versus-noise separation among the events of one experiment only,
    where knowing the experiment gives no information about the class.
    Balanced accuracy (mean of crack and noise recall, probability rule) and
    AUC; 0.5 is chance for both."""
    out = {}
    for exp in sorted(set(experiment_of.values())):
        rs = [r for r in records if experiment_of[r["event"]] == exp]
        y = np.array([r["true"] == CLASS_CRACK for r in rs], dtype=int)
        pred = np.array([r["detected"] and r["crack_score"] > 0.5 for r in rs], dtype=int)
        n_c, n_n = int(y.sum()), int(len(y) - y.sum())
        entry = {"n_crack": n_c, "n_noise": n_n}
        if n_c and n_n:
            rec_c = float((pred[y == 1] == 1).mean())
            rec_n = float((pred[y == 0] == 0).mean())
            score = np.array([r["crack_score"] for r in rs])
            entry.update({"crack_recall": rec_c, "noise_recall": rec_n,
                          "balanced_accuracy": (rec_c + rec_n) / 2,
                          "auc": float(roc_auc_score(y, score)),
                          "crack_prevalence": float(y.mean())})
            order = np.argsort(-score)
            for frac in TOP_FRACTIONS:
                n = max(1, int(round(frac * len(score))))
                entry[f"top_{int(frac * 100)}pct_crack_precision"] = float(y[order[:n]].mean())
        out[exp] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["lstm", "cnn_lstm", "transformer"])
    ap.add_argument("--figure", default=None, help="write the ensemble ROC figure here")
    ap.add_argument("--in-dir", default=None, help="results folder (default outputs/discrimination)")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    config["_config_dir"] = str(config_path.parent)
    cs = config["dataset"]["chunk_size"]
    _, labels = prepare_dataset(config)                    # from cache
    ids, placements = frame_event_ids(config, labels)
    lib_paths = event_library_paths(config)
    experiment_of = {k: ("2022 (2 V)" if "/2V/" in lib_paths[c][i] else "2024 (1.3 V)")
                     for k, (c, i, _s) in enumerate(placements)}
    specimen_of = {k: re.sub(r"^\d+_|_20\d{6}-.*$", "", pathlib.Path(lib_paths[c][i]).name)
                   for k, (c, i, _s) in enumerate(placements)}
    # only events with frames in the chunked part of the stream are scored
    scored = set(np.unique(ids[: len(labels) // cs * cs])) - {-1}
    shortcut = np.mean([(c == CLASS_CRACK) == (experiment_of[k] == "2024 (1.3 V)")
                        for k, (c, _i, _s) in enumerate(placements) if k in scored])

    in_dir = pathlib.Path(args.in_dir) if args.in_dir else OUT_DIR
    summary = {"condition": args.condition, "seeds": args.seeds, "models": {},
               "shortcut_baseline_experiment_to_class": float(shortcut)}
    print(f"shortcut baseline (class guessed from experiment alone): {shortcut:.3f}")
    for model in args.models:
        runs = [load_folds(in_dir / f"{args.condition}_{model}_seed{s}.npz") for s in args.seeds]
        # the fold split is fixed, so every seed must have identical validation chunks
        for run in runs[1:]:
            for a, b in zip(runs[0], run):
                assert np.array_equal(a["val_chunk_idx"], b["val_chunk_idx"])
        m = {"per_seed": {}}
        for s, run in zip(args.seeds, runs):
            rec = event_decisions(run, ids, placements, cs)
            m["per_seed"][str(s)] = {"frame": frame_scores(run, labels, cs),
                                     "event_vote": event_scores(rec, "vote"),
                                     "event_prob": event_scores(rec, "prob"),
                                     "event_auc": roc_points(rec)["auc"],
                                     "by_experiment": by_experiment(rec, experiment_of)}
        ens = [{"val_chunk_idx": fd["val_chunk_idx"],
                "y_prob": np.mean([run[i]["y_prob"] for run in runs], axis=0)}
               for i, fd in enumerate(runs[0])]
        rec = event_decisions(ens, ids, placements, cs)
        m["ensemble"] = {"frame": frame_scores(ens, labels, cs),
                         "event_vote": event_scores(rec, "vote"),
                         "event_prob": event_scores(rec, "prob"),
                         "roc": roc_points(rec, experiment_of),
                         "by_experiment": by_experiment(rec, experiment_of),
                         "by_specimen": by_experiment(rec, specimen_of)}
        summary["models"][model] = m

        e, r = m["ensemble"], m["ensemble"]["roc"]
        print(f"\n{LABEL[model]} ({args.condition}, seeds {args.seeds})")
        for s in args.seeds:
            ps = m["per_seed"][str(s)]
            print(f"  seed {s}: frame disc {ps['frame']['discrimination_accuracy']:.3f} | "
                  f"event vote {ps['event_vote']['discrimination_accuracy']:.3f} | "
                  f"event prob {ps['event_prob']['discrimination_accuracy']:.3f} | "
                  f"AUC {ps['event_auc']:.3f}")
        print(f"  ensemble: frame disc {e['frame']['discrimination_accuracy']:.3f} | "
              f"detection F1 {e['frame']['detection_f1']:.3f} | "
              f"event prob {e['event_prob']['discrimination_accuracy']:.3f} "
              f"(crack F1 {e['event_prob']['crack_f1']:.3f}, noise F1 {e['event_prob']['noise_f1']:.3f}) | "
              f"AUC {r['auc']:.3f}")
        for exp, v in e["by_experiment"].items():
            if "balanced_accuracy" in v:
                print(f"    within {exp}: {v['n_crack']} crack / {v['n_noise']} noise -> balanced acc "
                      f"{v['balanced_accuracy']:.3f}, AUC {v['auc']:.3f} (chance 0.5); top-10/20/30% "
                      f"crack precision {v['top_10pct_crack_precision']:.2f}/{v['top_20pct_crack_precision']:.2f}/"
                      f"{v['top_30pct_crack_precision']:.2f} vs prevalence {v['crack_prevalence']:.2f}")
        for sp, v in e["by_specimen"].items():
            if "balanced_accuracy" in v and min(v["n_crack"], v["n_noise"]) >= 5:
                print(f"    within specimen {sp}: {v['n_crack']} crack / {v['n_noise']} noise -> balanced acc "
                      f"{v['balanced_accuracy']:.3f}, AUC {v['auc']:.3f}")
        for name, t in r["most_crack_like"].items():
            print(f"    {name} most crack-like events ({t['n_events']}, {t.get('from_2024', '?')} from 2024): "
                  f"{t['crack_precision']:.2f} are cracks, covering {t['share_of_all_cracks']:.2f} of all cracks")
        for name, op in r["operating_points"].items():
            print(f"    {name}: crack recall {op['crack_recall']:.3f}, "
                  f"noise flagged as crack {op['noise_flagged_as_crack']:.3f}")

    out = in_dir / f"event_level_summary_{args.condition}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")

    if args.figure:
        with plt.rc_context({"font.size": 13}):
            fig, ax = plt.subplots(figsize=(6.2, 5.6))
            for model in args.models:
                r = summary["models"][model]["ensemble"]["roc"]
                ax.plot(r["fpr"], r["tpr"], lw=2, label=f"{LABEL[model]} (AUC {r['auc']:.2f})")
            ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Chance")
            ax.set_xlabel("Noise events labelled crack (false-alarm rate)")
            ax.set_ylabel("Crack events labelled crack (recall)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", frameon=False)
            fig.tight_layout()
            fig.savefig(args.figure, dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
