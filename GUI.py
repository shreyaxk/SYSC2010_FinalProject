import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from LOADER import load_csv_numeric
from filter_ecg import apply_ecg_filter
from filter_respiration import apply_respiration_filter
from filter_imu import imu_filter
from filter_bodytemp import bodytemperature_filter


class SignalGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Signal Analyzer")
        self.window.geometry("500x500")

        # file input
        tk.Label(self.window, text="CSV File Path").pack()
        self.file_box = tk.Entry(self.window, width=50)
        self.file_box.pack()

        # signal type dropdown
        tk.Label(self.window, text="Signal Type").pack()
        self.signal_type_var = tk.StringVar(value="ECG")
        tk.OptionMenu(
            self.window,
            self.signal_type_var,
            "ECG",
            "Respiration",
            "Temperature",
            "IMU",
        ).pack()

        # apply filter checkbox
        self.filter_var = tk.IntVar(value=0)
        tk.Checkbutton(self.window, text="Apply Filter", variable=self.filter_var).pack()

        # buttons
        tk.Button(self.window, text="Plot Filtered Signal", command=self.plot_time).pack(pady=5)
        tk.Button(self.window, text="Plot FFT", command=self.plot_fft).pack(pady=5)
        tk.Button(self.window, text="Plot Raw Signal", command=self.plot_raw_signal).pack(pady=5)

        # plot limit
        self.PLOT_LIMIT = 3000

        self.window.mainloop()

    # load CSV
    def load_data(self):
        path = self.file_box.get().strip()
        headers, data = load_csv_numeric(path)
        t = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
        return t, y

    # apply filter
    def apply_selected_filter(self, y, fs):
        if not self.filter_var.get():
            return y
        try:
            sig_type = self.signal_type_var.get()
            if sig_type == "ECG":
                return apply_ecg_filter(y, fs)
            elif sig_type == "Respiration":
                return apply_respiration_filter(y, fs)
            elif sig_type == "IMU":
                return imu_filter(y, fs)
            elif sig_type == "Temperature":
                return bodytemperature_filter(y, fs)
            else:
                return y
        except Exception as e:
            print("Filter failed:", e)
            return y

    # compute sampling frequency
    def compute_fs(self, t):
        return 1 / np.mean(np.diff(t))

    # get downsampling indices
    def get_downsample_indices(self, length):
        if length > self.PLOT_LIMIT:
            step = length // self.PLOT_LIMIT
            return np.arange(0, length, step)
        else:
            return np.arange(length)

    # plot time domain
    def plot_time(self):
        t, y = self.load_data()
        fs = self.compute_fs(t)

        y_processed = self.apply_selected_filter(y, fs)

        # use same indices for both signals
        indices = self.get_downsample_indices(len(t))
        t_plot = t[indices]
        y_raw_plot = y[indices] - np.mean(y[indices])      
        y_proc_plot = y_processed[indices] - np.mean(y_processed[indices])

        plt.close('all')
        plt.figure(figsize=(10, 4))
        plt.plot(t_plot, y_raw_plot, label="Raw", alpha=0.5)
        plt.plot(t_plot, y_proc_plot, label="Processed", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Time Domain ({self.signal_type_var.get()})")
        plt.legend()
        plt.show()

    # plot raw signal
    def plot_raw_signal(self):
        t, y = self.load_data()
        y_centered = y - np.mean(y)

        plt.figure(figsize=(10, 4))
        plt.plot(t, y_centered, label="Raw Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Raw Signal ({self.signal_type_var.get()})")
        plt.legend()
        plt.show()

    # plot FFT
    def plot_fft(self):
        t, y = self.load_data()
        fs = self.compute_fs(t)

        y_processed = self.apply_selected_filter(y, fs)

        indices = self.get_downsample_indices(len(y_processed))
        y_processed_ds = y_processed[indices] - np.mean(y_processed[indices])
        fs_ds = fs * len(indices) / len(y_processed)

        N = len(y_processed_ds)
        freqs = np.fft.rfftfreq(N, d=1/fs_ds)
        fft_vals = np.abs(np.fft.rfft(y_processed_ds))

        plt.close('all')
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, fft_vals)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"FFT ({self.signal_type_var.get()})")
        plt.show()


if __name__ == "__main__":
    SignalGUI()