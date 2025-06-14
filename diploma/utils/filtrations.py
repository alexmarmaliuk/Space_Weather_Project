import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os

def fft_filters_plot(df, col='Byc', time_col='Time', fs=1, figure=(10, 8), is_plot=True, title='', path='', save=True):
    """
    Extracts and plots daily, fast, and band-pass components from a time series using FFT.

    Parameters:
        df (pd.DataFrame): DataFrame with time and signal columns.
        col (str): Name of the signal column.
        time_col (str): Name of the time column.
        fs (float): Sampling frequency in Hz (samples per second).
    """
    signal = np.array(df[col])
    time = np.array(df[time_col])
    N = len(signal)

    fft_data = fft(signal)
    freqs = fftfreq(N, d=1/fs)

    daily = (np.abs(freqs) > 1/(24*60*60)) & (np.abs(freqs) < 1/(2*60*60))  # 2h–24h
    fast = np.abs(freqs) > 1/3600                                            # <1h
    bandpass = (np.abs(freqs) >= 1/600) & (np.abs(freqs) <= 1/10)            # 10–600s

    fft_daily = np.zeros_like(fft_data)
    fft_fast = np.zeros_like(fft_data)
    fft_band = np.zeros_like(fft_data)

    fft_daily[daily] = fft_data[daily]
    fft_fast[fast] = fft_data[fast]
    fft_band[bandpass] = fft_data[bandpass]

    daily_component = np.real(ifft(fft_daily))
    fast_component = np.real(ifft(fft_fast))
    band_component = np.real(ifft(fft_band))

    if (is_plot):
        plt.figure(figsize=figure)
        plt.suptitle(title, fontsize=21)
        plt.subplot(3, 1, 1)
        plt.plot(time, daily_component)
        plt.title("Daily variations (2–24h)")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(time, fast_component)
        plt.title("Fast variations (<1h)")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(time, band_component)
        plt.title("10–600s variations")
        plt.grid()

        plt.tight_layout()
        
        if (save):
            plt.savefig(path)

        plt.show()

    return {
        'signal': signal,
        'daily': daily_component,
        'fast': fast_component,
        'band': band_component,
        'freqs': freqs,
        'fft_data': fft_data
    }
