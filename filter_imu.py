import numpy as np
from scipy.signal import butter, filtfilt

def imu_filter(signal, fs, cutoff=5, order=3):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal