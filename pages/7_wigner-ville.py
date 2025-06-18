import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tftb.processing import WignerVilleDistribution, PseudoWignerVilleDistribution

st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Wigner-Ville Distributions Demo')

# Load data info
with open('pages/app_data/data_choice.txt', 'r') as file:
    data_choice = file.read()

st.markdown(f'#### Working with **{data_choice}** data.')

# Load data
datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

# User input: downsampling & slicing
sr = st.number_input('Decimation factor (integer)', min_value=1, max_value=100, value=5, step=1)
interval_start, interval_end = st.slider(
    'Select data interval (start and end indices)',
    min_value=0,
    max_value=len(data),
    value=(0, len(data)),
    step=1
)

interval = slice(interval_start, interval_end)
timestamps = np.array(data.x)[interval][::sr]
signal = np.array(data.y)[interval][::sr]

# Calculate dt from timestamps
if len(timestamps) > 1:
    dt = np.mean(np.diff(timestamps))
else:
    dt = 1.0

def plot_wvd_positive(signal, timestamps, dt=1.0, figsize=(12,6), title='Wigner-Ville Transform (positive frequencies only)'):
    wvd = WignerVilleDistribution(signal, timestamps=timestamps)
    tfr, _, _ = wvd.run()

    f = np.fft.fftshift(np.fft.fftfreq(tfr.shape[0], d=2 * dt))
    df = f[1] - f[0]

    pos_mask = f > 0
    f_pos = f[pos_mask]
    tfr_pos = np.fft.fftshift(tfr, axes=0)[pos_mask, :]

    t_plot = np.linspace(timestamps[0] - dt / 2, timestamps[-1] + dt / 2, tfr.shape[1])
    f_plot = np.linspace(f_pos[0] - df / 2, f_pos[-1] + df / 2, len(f_pos))

    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(t_plot, f_plot, tfr_pos, shading='auto', cmap='jet')
    fig.colorbar(c, ax=ax, label='Amplitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    fig.tight_layout()

    return fig

def plot_pwvd_positive(signal, timestamps, dt=1.0, figsize=(12,6), title='Pseudo Wigner-Ville Transform (positive frequencies only)'):
    pwvd = PseudoWignerVilleDistribution(signal, timestamps=timestamps)
    tfr, _, _ = pwvd.run()

    f = np.fft.fftshift(np.fft.fftfreq(tfr.shape[0], d=2 * dt))
    df = f[1] - f[0]

    pos_mask = f > 0
    f_pos = f[pos_mask]
    tfr_pos = np.fft.fftshift(tfr, axes=0)[pos_mask, :]

    t_plot = np.linspace(timestamps[0] - dt / 2, timestamps[-1] + dt / 2, tfr.shape[1])
    f_plot = np.linspace(f_pos[0] - df / 2, f_pos[-1] + df / 2, len(f_pos))

    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(t_plot, f_plot, tfr_pos, shading='auto', cmap='jet')
    fig.colorbar(c, ax=ax, label='Amplitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    fig.tight_layout()

    return fig


# Generate plots
fig1 = plot_wvd_positive(signal, timestamps, dt=dt, title='Classic Wigner-Ville Distribution')
st.pyplot(fig1)

fig2 = plot_pwvd_positive(signal, timestamps, dt=dt, title='Pseudo Wigner-Ville Distribution')
st.pyplot(fig2)
