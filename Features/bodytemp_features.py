import numpy as np

def average_body_temp(y):
    if len(y) == 0:
        return 0, 0

    avg_temp = np.mean(y)
    num_samples = len(y)

    return avg_temp, num_samples