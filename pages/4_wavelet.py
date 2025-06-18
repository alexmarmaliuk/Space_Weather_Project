# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import waipy
# import altair as alt

# from pages.additional.sup import *
# from pages.additional.plotting import *

# st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
# st.title('📊 Demo')

# st.sidebar.header("Wavelet Sprectrum")

# with open(('pages/app_data/data_choice.txt'), 'r') as file:
#     data_choice = file.read()

# st.markdown(f'#### Working with **{data_choice}** data.')

# datapath = 'pages/app_data/current_data.csv'
# data = pd.read_csv(datapath)

# s0 = st.slider('Smallest scale', min_value=1, max_value=4000, value=401, step=1)
# s1 = st.slider('Largest scale', min_value=1, max_value=4000, value=2*s0, step=1)
# ds = st.slider('Scale step', min_value=0.1, max_value=100., value=0.1)
# nvoice = st.slider('octaves / scale', min_value=3, max_value=20, value=6)
# lag = st.number_input(label='autocorrelation', value=1)

# alpha = 6
# mother = 'Morlet'

# result = waipy.cwt(
#     data=data.y,
#     dt=s0,
#     pad=1,
#     dj=ds,
#     s0=s1, 
#     j1=(s1-s0)/ds,
#     lag1=lag,
#     param=alpha,
#     mother='Morlet',
#     name='Morlet'
# )


# # import matplotlib.cm as cm
# # import waipy.cwt.wavetest as wavetest  # ← This is where wavelet_plot is located

# # # Patch waipy's broken reference to mpl.cm.get_cmap
# # if not hasattr(wavetest.mpl.cm, "get_cmap"):
# #     wavetest.mpl.cm.get_cmap = cm.get_cmap


# label = 'Morlet wavelet'
# fig = plt.figure(figsize=(10, 6))
# waipy.wavelet_plot(label, data.x, data.y, ds, result, plot_power=False, contour_levels=20, yscale='log', ylabel='Period')
# st.pyplot(fig)


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from pages.additional.sup import *
from pages.additional.plotting import *

st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Wavelet Spectrogram")

# Load data choice info
with open('pages/app_data/data_choice.txt', 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

# Load data
datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

# --- User parameters ---
sr = st.number_input('Decimation factor (integer)', min_value=1, max_value=100, value=5, step=1)

# Limit the domain slice here
interval_start, interval_end = st.slider(
    'Select data interval (start and end indices)',
    min_value=0,
    max_value=len(data),
    value=(0, len(data)),
    step=1
)

# Slice and downsample domain and signal
interval = slice(interval_start, interval_end)
timestamps = np.array(data.x)[interval][::sr]
signal = np.array(data.y)[interval][::sr]

# Wavelet kernel choice
kernel = st.selectbox("Select wavelet kernel", ['morl', 'gaus1'])

# Scales: user can provide max scale or use default
max_scale = st.number_input('Max scale', min_value=10, max_value=500, value=128)
scales = np.arange(1, max_scale + 1)

# Sampling interval (time between samples)
# Calculate dt from timestamps or use fixed (assume uniform)
if len(timestamps) > 1:
    dt = np.mean(np.diff(timestamps))
else:
    dt = 1.0

def plot_wavelet_spectrogram(signal, timestamps, wavelet='morl', scales=None, dt=1.0, figsize=(12, 6), title=None, path=''):
    if scales is None:
        max_scale = 128
        scales = np.arange(1, max_scale + 1)

    # Continuous wavelet transform
    coefficients, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    power = np.abs(coefficients) ** 2

    if title is None:
        title = f"Wavelet Spectrogram using '{wavelet}'"

    plt.figure(figsize=figsize)
    extent = [timestamps[0], timestamps[-1], freqs[-1], freqs[0]]  # freq axis flipped
    plt.imshow(power, extent=extent, aspect='auto', cmap='jet')
    plt.colorbar(label='Power')
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()

    if path:
        plt.savefig(path)

    st.pyplot(plt.gcf())
    plt.close()

    return power, freqs

# Plot wavelet spectrogram
power, freqs = plot_wavelet_spectrogram(signal, timestamps, wavelet=kernel, scales=scales, dt=dt,
                                        title=f"Wavelet Spectrogram of {data_choice}", figsize=(12, 6))
