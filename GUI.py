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

        # file
        tk.Label(self.window, text="CSV File Path").pack()
        self.file_box = tk.Entry(self.window, width=50)
        self.file_box.pack()

        # signal type dropdown
        tk.Label(self.window, text="Signal Type").pack()
        self.signal_type_var = tk.StringVar(value="ECG")
        tk.OptionMenu(self.window, self.signal_type_var,
                      "ECG", "Respiration", "Temperature", "IMU").pack()

        # apply filter, to apply the filter 
        self.filter_var = tk.IntVar(value=0)
        tk.Checkbutton(self.window, text="Apply Filter", variable=self.filter_var).pack()

        tk.Button(self.window, text="Plot Filtered Signal", command=self.plot_time).pack(pady=5)
        tk.Button(self.window, text="Plot FFT", command=self.plot_fft).pack(pady=5)
        tk.Button(self.window, text="Plot Raw Signal", command=self.plot_raw_signal).pack(pady=5)

        self.PLOT_LIMIT = 3000

        self.window.mainloop()

    def load_data(self):
        path = self.file_box.get().strip()
        headers, data = load_csv_numeric(path)

        t = data[:, 0].astype(float)
        y = data[:, 1].astype(float)

        return t, y

    def apply_selected_filter(self, y, fs):
        if not self.filter_var.get():
            return y

        try:
            signal_type = self.signal_type_var.get()

            if signal_type == "ECG":
                return apply_ecg_filter(y, fs)
            elif signal_type == "Respiration":
                return apply_respiration_filter(y, fs)
            elif signal_type == "IMU":
                return imu_filter(y, fs)
            elif signal_type == "Temperature":
                return bodytemperature_filter(y, fs)
            else:
                return y

        except Exception as e:
            print("Filter failed:", e)
            return y  #Makes sure the plot works, even if the filter crashes 

    def compute_fs(self, t):
        return 1 / np.mean(np.diff(t))

    def downsample(self, t, y):
        if len(t) > self.PLOT_LIMIT:
            step = len(t) // self.PLOT_LIMIT
            return t[::step], y[::step]
        return t, y

    def plot_time(self):
        t, y = self.load_data()
        fs = self.compute_fs(t)

        print(f"Sampling frequency (fs): {fs:.2f} Hz")

        y_processed = self.apply_selected_filter(y, fs)

        # downsampling so my code does'nt lag
        t_plot, y_raw_plot = self.downsample(t, y)
        _, y_proc_plot = self.downsample(t, y_processed)

        plt.close('all')
        plt.figure(figsize=(10, 4))

        plt.plot(t_plot, y_raw_plot, label="Raw", alpha=0.5)
        plt.plot(t_plot, y_proc_plot, label="Processed", linewidth=1.5)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Time Domain ({self.signal_type_var.get()})")
        plt.legend()
        plt.show()

    def plot_raw_signal(self):
        t, y = self.load_data()

        plt.figure(figsize=(10, 4))
        plt.plot(t, y, label="Raw Signal")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Raw Signal ({self.signal_type_var.get()})")
        plt.legend()
        plt.show()

    def plot_fft(self):
        t, y = self.load_data()
        fs = self.compute_fs(t)

        print(f"Sampling frequency (fs): {fs:.2f} Hz")

        y_processed = self.apply_selected_filter(y, fs)

        # Downsample safely
        if len(y_processed) > self.PLOT_LIMIT:
            step = len(y_processed) // self.PLOT_LIMIT
            y_processed = y_processed[::step]
            fs = fs / step

        # removes DC (the amplitude shift, downwards)
        y_processed = y_processed - np.mean(y_processed)

        N = len(y_processed)
        freqs = np.fft.rfftfreq(N, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(y_processed))

        plt.close('all')
        plt.figure(figsize=(10, 4))

        plt.plot(freqs, fft_vals)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"FFT ({self.signal_type_var.get()})")
        plt.show()


if __name__ == "__main__":
    SignalGUI()