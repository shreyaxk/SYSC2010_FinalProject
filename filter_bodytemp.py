from scipy.signal import butter, filtfilt

def bodytemperature_filter(signal, fs, cutoff=0.005, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)