import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from pages.additional.preprocessing import *
from pages.additional.sup import *
from pages.additional.plotting import *


# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Bandpass filter")


with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

st.markdown('''
            ### Nyquist-Shannon sampling criteria (Kotelnikov theorem)
            
            If our signal has the highest frequency $f_{max}$ then we need to sample it with 
            $$f_s > 2f_{max}$$
            
            ### Plot params
            ''')
            
            
# plot 2
# Assume 1 sample per minute ⇒ fs = 1/60 Hz
sr = 1 / 60  # Hz
st.markdown(f"**Sampling frequency:** {sr:.5f} Hz (1 sample/minute)")

# Periods in seconds (e.g., 2 minutes = 120s to 24 hours = 86400s)
min_period = 121         # 2 minutes
max_period = 43200    # 0.5 day

low_period, high_period = st.slider(
    'Select Low and High Periods (in seconds)',
    min_value=min_period,
    max_value=max_period,
    value=(600, 7200),  # Example: 10 min to 2 hours
    step=60
)

# Convert periods (seconds) to frequencies (Hz) for filtering
lowcut = 1 / high_period
highcut = 1 / low_period

plot_type = st.selectbox('Plotting lib', ['plotly', 'pyplot'])

filtered_signal = bandpass(data.y, highcut=highcut, lowcut=lowcut, fs=sr)

plot_single(
    data.x,
    filtered_signal,
    f"Periods: {low_period}s to {high_period}s → Frequencies: {lowcut:.5f}Hz–{highcut:.5f}Hz",
    plot_type,
    x_label='Time',
    y_label='Signal'
)
