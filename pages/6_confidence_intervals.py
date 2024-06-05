import numpy as np
import scipy as sp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
# import waipy
import pandas as pd
import altair as alt

from pages.additional.preprocessing import *
from pages.additional.plotting import *
from pages.additional.sup import *

with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

var = st.number_input(label='noise variance', value=1)
number_of_samples = st.number_input(label='sampling size', value=100, min_value=1, step=10)
noise_type = st.radio("Noise type for samples", 
                      ['white', 'pink', 'Brownian', 'blue', 'violet']
                      )




data = pd.read_csv('pages/app_data/current_data.csv')
t = data.x
signal = data.y

# Generate a sample signal: a sum of two sine waves with noise
sampling_rate = len(t) 
T = 1.0 / sampling_rate  # Sampling interval


fft_result = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(t.size, T)


fft_magnitude = 2.0 / t.size * np.abs(fft_result[:t.size // 2])
fft_freqs = fft_freqs[:t.size // 2]
samples, ffts = generate_samples(data.x, data.y, number_of_samples, var, noise_type)
plot_samples(samples=samples, x=data.x, num=50)
plot_confidence_intervals(ffts, fft_freqs, fft_magnitude)