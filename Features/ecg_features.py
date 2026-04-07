import numpy as np
from scipy.signal import find_peaks

def compute_heart_rate_ecg(y, fs):
    peaks, _ = find_peaks(y, distance=int(0.6 * fs), height=0.5*np.std(y))

    if len(peaks) < 2:
        return 0, 0

    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)

    return hr, len(peaks)