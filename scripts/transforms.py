import torch
import torchaudio
import numpy as np
from .constants import LABEL_MAP

def compute_spectrogram(waveform, n_fft, hop_length, win_length, power=2.0, window='hann', device='cpu'):
    """
    Computes a spectrogram from a waveform.
    Returns: (f_hz, t_s, Sxx)
    Note: In this implementation, we return Sxx as a Tensor. 
    f_hz and t_s can be derived from metadata if needed for plotting.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)

    # Use torchaudio Spectrogram logic
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=power,
        window_fn=torch.hann_window if window == 'hann' else torch.hamming_window,
        center=True,      # Required for the alignment logic below
        pad_mode="reflect"
    ).to(device)

    Sxx = spec_transform(waveform).squeeze(0)
    
    # We return the tensor. f_hz and t_s are usually handled in plotting scripts.
    return Sxx

def align_labels_to_frames(per_sample_labels, n_frames, hop_length, win_length, priority=None):
    """
    Generalizes label alignment from samples to STFT frames.
    
    Args:
        per_sample_labels: 1D array of labels for every raw audio sample.
        n_frames: The exact number of frames in the spectrogram (Sxx.shape[1]).
        hop_length: From config.
        win_length: From config.
        priority: Tuple of class labels in order of importance. 
                 Default: (Crack, Noise, Background)
    """
    if priority is None:
        priority = (LABEL_MAP['crack'], LABEL_MAP['mechanical_noise'], LABEL_MAP['background'])
    
    per_frame_labels = np.zeros(n_frames, dtype=np.int64)
    n_samples = len(per_sample_labels)

    for i in range(n_frames):
        # Formula for STFT frame center with center=True padding:
        # The center of frame 'i' is exactly at sample i * hop_length
        center_sample = i * hop_length
        
        # Calculate window boundaries
        start = max(0, center_sample - win_length // 2)
        end = min(n_samples, center_sample + win_length // 2)
        
        window_labels = per_sample_labels[start:end]
        
        if len(window_labels) == 0:
            per_frame_labels[i] = priority[-1] # Default to lowest priority (Background)
            continue
            
        # Apply explicit priority rule
        assigned = False
        for label_type in priority:
            if label_type in window_labels:
                per_frame_labels[i] = label_type
                assigned = True
                break
        
        if not assigned:
            per_frame_labels[i] = priority[-1]

    return per_frame_labels