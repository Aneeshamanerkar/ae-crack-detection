#!/usr/bin/env python
"""Build the event cache read by ``screen_full_dataset``.

Reads every extracted event of both experimental series (the directories
``events.event_root``/2022_n/2V and ``events.event_root``/2024_n/1.3V of the
configuration), in sorted path order, and writes

    outputs/cache/screening/event_waveforms.npy   float32 [N, 4096]
    outputs/cache/screening/event_metadata.csv    index, filename, specimen, source_dir

Each waveform is the structure-borne channel of the event file (column ``CH``
after the twelve header lines), as in ``prepare_data``.

    python -m scripts.build_screening_cache --workers 8
"""

from __future__ import annotations

import argparse
import pathlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from scripts.constants import CH, PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "default_new.yaml"
OUT = PROJECT_ROOT / "outputs" / "cache" / "screening"
SERIES_DIRS = ("2022_n/2V", "2024_n/1.3V")
EVENT_LEN = 4096


def read_waveform(path: str) -> np.ndarray:
    w = pd.read_csv(path, sep=",", skiprows=12, header=None).to_numpy()[:, CH].astype(np.float32)
    assert w.shape == (EVENT_LEN,), (path, w.shape)
    return w


def specimen(name: str) -> str:
    return next(p for p in pathlib.Path(name).stem.split("_") if p.startswith("B"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    root = (CONFIG_PATH.parent / config["events"]["event_root"]).resolve()
    paths = sorted(str(p) for d in SERIES_DIRS for p in (root / d).glob("*.csv"))
    print(f"{len(paths)} event files")

    with Pool(args.workers) as pool:
        waves = pool.map(read_waveform, paths, chunksize=500)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "event_waveforms.npy", np.stack(waves))
    pd.DataFrame({
        "index": np.arange(len(paths)),
        "filename": [pathlib.Path(p).name for p in paths],
        "specimen": [specimen(p) for p in paths],
        "source_dir": [pathlib.Path(p).parent.parent.name for p in paths],
    }).to_csv(OUT / "event_metadata.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
