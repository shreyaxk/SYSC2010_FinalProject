import pandas as pd
import numpy as np

def load_csv_numeric(path: str, fs=250):
    df = pd.read_csv(path)

    # CASE 1: ECG-style file (1 column, no time)
    if df.shape[1] == 1:
        signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        signal = signal.dropna().reset_index(drop=True)

        if signal.empty:
            raise ValueError("No valid ECG data found.")

        t = np.arange(len(signal)) / fs

        headers = ["time", "lead_I"]
        data = np.column_stack((t, signal.to_numpy()))
        return headers, data

    # CASE 2: Normal file (time + sensor)
    first_col = df.columns[0]

    # --- Convert time safely ---
    t_numeric = pd.to_numeric(df[first_col], errors='coerce')

    if t_numeric.notna().all():
        t = t_numeric
    else:
        try:
            dt = pd.to_datetime(df[first_col], errors='coerce')
            t = (dt - dt.iloc[0]).dt.total_seconds()
        except Exception:
            t = t_numeric

    df[first_col] = t


    df = df.dropna(subset=[first_col])

    if df.empty:
        raise ValueError("No valid data after cleaning.")

    #ensure increasing time
    df = df.sort_values(by=first_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[first_col]).reset_index(drop=True)

    # convert ALL columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any NaNs 
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid numeric data after cleaning.")

    headers = list(df.columns)
    data = df.to_numpy(dtype=float)

    return headers, data