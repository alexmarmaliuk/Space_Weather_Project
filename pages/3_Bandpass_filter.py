# import streamlit as st
# import numpy as np
# import pandas as pd
# import altair as alt

# from pages.additional.preprocessing import *
# from pages.additional.sup import *
# from pages.additional.plotting import *


# # Page title
# st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
# st.title('📊 Demo')

# st.sidebar.header("Bandpass filter")


# with open(('pages/app_data/data_choice.txt'), 'r') as file:
#     data_choice = file.read()
    
# st.markdown(f'#### Working with **{data_choice}** data.')

# datapath = 'pages/app_data/current_data.csv'
# data = pd.read_csv(datapath)

# st.markdown('''
#             ### Nyquist-Shannon sampling criteria (Kotelnikov theorem)
            
#             If our signal has the highest frequency $f_{max}$ then we need to sample it with 
#             $$f_s > 2f_{max}$$
            
#             ### Plot params
#             ''')
            
            
# # plot 2
# # Assume 1 sample per minute ⇒ fs = 1/60 Hz
# sr = 1 / 60  # Hz
# st.markdown(f"**Sampling frequency:** {sr:.5f} Hz (1 sample/minute)")

# # Periods in seconds (e.g., 2 minutes = 120s to 24 hours = 86400s)
# min_period = 121         # 2 minutes
# max_period = 43200    # 0.5 day

# low_period, high_period = st.slider(
#     'Select Low and High Periods (in seconds)',
#     min_value=min_period,
#     max_value=max_period,
#     value=(600, 7200),  # Example: 10 min to 2 hours
#     step=60
# )

# # Convert periods (seconds) to frequencies (Hz) for filtering
# lowcut = 1 / high_period
# highcut = 1 / low_period

# plot_type = st.selectbox('Plotting lib', ['plotly', 'pyplot'])

# filtered_signal = bandpass(data.y, highcut=highcut, lowcut=lowcut, fs=sr)

# plot_single(
#     data.x,
#     filtered_signal,
#     f"Periods: {low_period}s to {high_period}s → Frequencies: {lowcut:.5f}Hz–{highcut:.5f}Hz",
#     plot_type,
#     x_label='Time',
#     y_label='Signal'
# )


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# from pages.additional.preprocessing import bandpass
from pages.additional.sup import *
from pages.additional.plotting import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Bandpass filter")

with open('pages/app_data/data_choice.txt', 'r') as file:
    data_choice = file.read().strip()
st.markdown(f'#### Working with **{data_choice}** data.')

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

st.markdown('''
### Nyquist-Shannon sampling criteria (Kotelnikov theorem)

If our signal has the highest frequency $f_{max}$ then we need to sample it with 

$$f_s > 2f_{max}$$

### Plot params
''')

# Sampling frequency: 1 sample per minute = 1/60 Hz
sr = 1 / 60  # Hz
st.markdown(f"**Sampling frequency:** {sr:.5f} Hz (1 sample/minute)")

# User chooses periods in seconds
min_period = 121         # 2 minutes
max_period = 43200       # 0.5 day

# low_period, high_period = st.slider(
#     'Select Low and High Periods (in seconds)',
#     min_value=min_period,
#     max_value=max_period,
#     value=(600, 7200),  # default 10 min to 2 hours
#     step=60
# )

# Convert periods to cutoff frequencies for bandpass filter

plot_type = st.selectbox('Plotting lib for bandpass filtered signal', ['plotly', 'pyplot'])

# --- Part 1: Show fixed 3-subplot FFT filtered components plot (daily, fast, band) ---

def fft_filters_plot(df, col='y', time_col='x', fs=sr, figure=(10, 8), is_plot=True, title='FFT Filtered Components'):
    signal = np.array(df[col])
    time = np.array(df[time_col])
    N = len(signal)

    fft_data = fft(signal)
    freqs = fftfreq(N, d=1/fs)

    # Fixed frequency bands for daily, fast, band
    daily = (np.abs(freqs) > 1/(24*3600)) & (np.abs(freqs) < 1/(2*3600))    # 2h–24h
    fast = np.abs(freqs) > 1/3600                                          # <1h
    bandpass = (np.abs(freqs) >= 1/600) & (np.abs(freqs) <= 1/10)          # 10–600s

    fft_daily = np.zeros_like(fft_data)
    fft_fast = np.zeros_like(fft_data)
    fft_band = np.zeros_like(fft_data)

    fft_daily[daily] = fft_data[daily]
    fft_fast[fast] = fft_data[fast]
    fft_band[bandpass] = fft_data[bandpass]

    daily_component = np.real(ifft(fft_daily))
    fast_component = np.real(ifft(fft_fast))
    band_component = np.real(ifft(fft_band))

    if is_plot:
        fig, axs = plt.subplots(3, 1, figsize=figure, sharex=True)
        fig.suptitle(title, fontsize=18)

        axs[0].plot(time, daily_component)
        axs[0].set_title("Daily variations (2–24h)")
        axs[0].grid()

        axs[1].plot(time, fast_component)
        axs[1].set_title("Fast variations (<1h)")
        axs[1].grid()

        axs[2].plot(time, band_component)
        axs[2].set_title("10–600s variations")
        axs[2].grid()
        axs[2].set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    return None

fig = fft_filters_plot(data)
st.pyplot(fig)



low_period, high_period = st.slider(
    'Select Low and High Periods (in seconds)',
    min_value=min_period,
    max_value=max_period,
    value=(600, 7200),  # default 10 min to 2 hours
    step=60
)

lowcut = 1 / high_period
highcut = 1 / low_period

def flexible_bandpass_fft(signal, fs, low_period, high_period):
    N = len(signal)
    fft_data = fft(signal)
    freqs = fftfreq(N, d=1/fs)

    low_freq = 1 / high_period  # convert period to frequency
    high_freq = 1 / low_period

    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)

    filtered_fft = np.zeros_like(fft_data)
    filtered_fft[mask] = fft_data[mask]

    filtered_signal = np.real(ifft(filtered_fft))
    return filtered_signal

# --- Part 2: User controlled flexible bandpass filter and plot ---

filtered_signal = flexible_bandpass_fft(data.y.values, sr, low_period, high_period)

plot_single(
    data.x,
    filtered_signal,
    f"User Bandpass Filter: Periods {low_period}s to {high_period}s → Frequencies {lowcut:.5f}Hz to {highcut:.5f}Hz",
    plot_type,
    x_label='Time',
    y_label='Signal'
)

