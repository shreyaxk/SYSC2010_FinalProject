import numpy as np
from scipy.signal import find_peaks

def respiration_rate(y, fs):
    # Minimum distance between peaks: 2 seconds (adjustable)
    min_distance_samples = int(2 * fs)

    # Find peaks
    peaks, _ = find_peaks(y, distance=min_distance_samples, height=0.3 * np.std(y))

    breath_count = len(peaks)

    if breath_count < 2:
        return 0, 0  # Not enough breaths detected to compute rate

    # calculate breath intervals
    intervals = np.diff(peaks) / fs

    rr = 60 / np.mean(intervals)  # breaths per minute

    return rr, breath_count