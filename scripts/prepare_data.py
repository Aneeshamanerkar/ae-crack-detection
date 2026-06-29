import os
import yaml
import torch
import hashlib
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .constants import LABEL_MAP, NUM_CLASSES
from .transforms import compute_spectrogram, align_labels_to_frames
# Assuming ae_io has a load_wfs or similar; if not yet built, we use a placeholder
# from .ae_io import load_wfs 

def load_event_library(config):
    """
    Loads isolated waveforms from data/events/ CSV files.
    Expected CSV format: time-series values in a single column or row.
    """
    events_dir = Path(config['paths']['events_dir'])
    library = {LABEL_MAP['crack']: [], LABEL_MAP['mechanical_noise']: []}
    
    # Example: files named 'crack_01.csv' or organized in folders
    for csv_path in events_dir.glob("*.csv"):
        df = pd.read_csv(csv_path, header=None)
        waveform = df.values.flatten().astype(np.float32)
        
        if 'crack' in csv_path.name.lower():
            library[LABEL_MAP['crack']].append(waveform)
        elif 'noise' in csv_path.name.lower():
            library[LABEL_MAP['mechanical_noise']].append(waveform)
            
    print(f"Loaded {len(library[1])} cracks and {len(library[2])} noise events.")
    return library

def select_background(stream_waveform, chunk_size):
    """
    Finds the quietest region (lowest RMS energy) in a long stream to use as background.
    """
    n_chunks = len(stream_waveform) // chunk_size
    best_chunk = None
    min_energy = float('inf')
    
    for i in range(n_chunks):
        chunk = stream_waveform[i*chunk_size : (i+1)*chunk_size]
        energy = np.sqrt(np.mean(chunk**2))
        if energy < min_energy:
            min_energy = energy
            best_chunk = chunk
            
    return best_chunk

def build_synthetic_stream(background, events_by_class, n_per_class, seed=42):
    """
    Injects events into the background using a rejection sampler to prevent overlaps.
    """
    np.random.seed(seed)
    stream = background.copy()
    labels = np.full(len(stream), LABEL_MAP['background'], dtype=np.int64)
    occupied_mask = np.zeros(len(stream), dtype=bool)
    
    for label, event_list in events_by_class.items():
        if not event_list: continue
        
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 10:
            attempts += 1
            event = event_list[np.random.randint(len(event_list))]
            event_len = len(event)
            
            # Pick a random start position
            start = np.random.randint(0, len(stream) - event_len)
            end = start + event_len
            
            # Check if this spot is already taken
            if not np.any(occupied_mask[start:end]):
                stream[start:end] += event
                labels[start:end] = label
                occupied_mask[start:end] = True
                count += 1
                
    return stream, labels

def prepare_dataset(config):
    """
    Main orchestration logic: Load -> Synthesize -> Transform -> Cache.
    """
    # 1. Create a unique hash for this configuration to use as a cache key
    config_str = str(config['stft']) + str(config['injection'])
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    cache_path = Path(config['paths']['output_dir']) / "cache" / f"data_{config_hash}.pkl"
    
    if cache_path.exists():
        print(f"Loading cached dataset: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Building new dataset...")
    # 2. Load background from a sample stream file (Placeholder logic)
    # In practice: stream_waveform = load_wfs(Path(config['paths']['raw_stream_dir']) / "sample.wfs")
    # For now, we simulate a quiet stream if files don't exist yet:
    stream_waveform = np.random.normal(0, 0.01, 1000000) 
    
    bg_chunk = select_background(stream_waveform, config['injection']['background_chunk_size'])
    
    # 3. Load events and inject
    library = load_event_library(config)
    synth_wave, synth_labels = build_synthetic_stream(
        bg_chunk, 
        library, 
        config['injection']['n_per_class'],
        seed=config['cv']['seed']
    )
    
    # 4. Transform to Spectrogram
    spec = compute_spectrogram(
        synth_wave, 
        **config['stft']
    )
    
    # 5. Align labels to frames
    frame_labels = align_labels_to_frames(
        synth_labels, 
        n_frames=spec.shape[1], 
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length']
    )
    
    dataset = {'spectrogram': spec, 'labels': frame_labels}
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data = prepare_dataset(config)
    print(f"Dataset prepared. Spectrogram shape: {data['spectrogram'].shape}")
    print(f"Labels shape: {data['labels'].shape}")