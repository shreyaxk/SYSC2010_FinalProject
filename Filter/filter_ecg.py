import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, fs, cutoff_frequency=40, order=2):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_frequency / nyquist

    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")

    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal)


def highpass_filter(signal, fs, cutoff_frequency=0.5, order=2):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_frequency / nyquist

    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")

    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal)


def apply_ecg_filter(signal, fs):
    # remove baseline drift
    signal = highpass_filter(signal, fs, cutoff_frequency=0.5, order=2)

    #remove high-frequency noise
    signal = lowpass_filter(signal, fs, cutoff_frequency=40, order=2)

    return signal