import numpy as np
import matplotlib.pyplot as plt
from tftb.processing import WignerVilleDistribution  # Ensure tftb is installed

def plot_wvd_positive(signal, timestamps, dt=1.0, figsize=(12, 6), title='Wigner-Ville Transform (positive frequencies only)', path='', save=True):
    """
    Computes and plots the Wigner-Ville Distribution (WVD) of a signal,
    displaying only the positive frequency components.

    Parameters:
        signal (array-like): Input signal.
        timestamps (array-like): Corresponding time values.
        dt (float): Time resolution (sampling interval).
        figsize (tuple): Figure size for plotting.
        title (str): Plot title.
    """
    # Compute WVD
    wvd = WignerVilleDistribution(signal, timestamps=timestamps)
    tfr, _, _ = wvd.run()

    # Compute shifted frequency axis
    f = np.fft.fftshift(np.fft.fftfreq(tfr.shape[0], d=2 * dt))
    df = f[1] - f[0]

    # Only keep positive frequencies
    pos_mask = f > 0
    f_pos = f[pos_mask]
    tfr_pos = np.fft.fftshift(tfr, axes=0)[pos_mask, :]

    # Time axis
    t_plot = np.linspace(timestamps[0] - dt / 2, timestamps[-1] + dt / 2, tfr.shape[1])
    f_plot = np.linspace(f_pos[0] - df / 2, f_pos[-1] + df / 2, len(f_pos))

    # Plot
    plt.figure(figsize=figsize)
    plt.pcolormesh(t_plot, f_plot, tfr_pos, shading='auto', cmap='jet')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    if (save):
        plt.savefig(path)
    plt.show()

    return tfr_pos, t_plot, f_pos



from tftb.processing import PseudoWignerVilleDistribution

def plot_pwvd_positive(signal, timestamps, dt=1.0, figsize=(12, 6), title='Pseudo Wigner-Ville Transform (positive frequencies only)'):
    """
    Computes and plots the Pseudo Wigner-Ville Distribution (PWVD) of a signal,
    showing only the positive frequencies.

    Parameters:
        signal (array-like): Input signal.
        timestamps (array-like): Time axis.
        dt (float): Sampling interval.
        figsize (tuple): Figure size.
        title (str): Title of the plot.
    """
    pwvd = PseudoWignerVilleDistribution(signal, timestamps=timestamps)
    tfr, _, _ = pwvd.run()

    f = np.fft.fftshift(np.fft.fftfreq(tfr.shape[0], d=2 * dt))
    df = f[1] - f[0]

    # Keep only positive frequencies
    pos_mask = f > 0
    f_pos = f[pos_mask]
    tfr_pos = np.fft.fftshift(tfr, axes=0)[pos_mask, :]

    t_plot = np.linspace(timestamps[0] - dt / 2, timestamps[-1] + dt / 2, tfr.shape[1])
    f_plot = np.linspace(f_pos[0] - df / 2, f_pos[-1] + df / 2, len(f_pos))

    plt.figure(figsize=figsize)
    plt.pcolormesh(t_plot, f_plot, tfr_pos, shading='auto', cmap='jet')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return tfr_pos, t_plot, f_pos
