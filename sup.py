from datetime import datetime, timedelta
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
from functools import partial
from scipy.signal import butter, lfilter

def generate_date_array(n, start_date='2024-01-01'):
    """
    Generate an array of dates starting from the given start_date, incrementing by one day for n elements.

    Args:
    - start_date: A string representing the start date in the format 'YYYY-MM-DD'.
    - n: An integer representing the number of elements in the array.

    Returns:
    - An array of datetime objects representing dates.
    """
    # Convert the start_date string to a datetime object
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Initialize an empty list to store the generated dates
    date_array = []
    
    # Generate dates by incrementing start_date by one day for n elements
    for i in range(n):
        # Append the date in the desired format to the date_array
        date_array.append(start_date.strftime('%Y-%m-%d'))
        start_date += timedelta(days=1)
    
    return np.array(date_array)

def get_moving_avg(y, window_size):
    # window_size = 50
    return pd.Series(y).rolling(window=window_size).mean()


def block_mean(original_array, k):
    original_array = np.array(original_array)
    num_blocks = len(original_array) // k
    blocks = original_array[:num_blocks*k].reshape(num_blocks, k)
    mean_array = np.mean(blocks, axis=1)
    return mean_array

def plot(x, y, label='', fig_size=15):
    fig = plt.figure(figsize=(fig_size, fig_size/2))
    plt.plot(x, y, label=label)
    plt.show()
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass(df, fs, lowcut=500.0, highcut=1250.0):
    # lowcut = 500.0
    # highcut = 1250.0
    return butter_bandpass_filter(df.y, lowcut, highcut, fs = 2*fs)