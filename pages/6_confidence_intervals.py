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

# datapath = 'pages/app_data/current_data.csv'
# data = pd.read_csv(datapath)
# x, y = data.x, data.y
# size = len(x)
# sample_rate = 24 * 60
# xf = sp.fft.fftfreq(size, 1 / sample_rate)
# yf = sp.fft.fft(np.array(y))

var = st.number_input(label='noise variance', value=1)
number_of_samples = st.number_input(label='sampling size', value=100, min_value=1, step=10)
# # number_of_samples = st.select_slider(label='sampling size', options=[1,])

# samples, ffts = generate_samples(x, y, 1000, var)
# plot_samples(samples=samples, x=x, num=number_of_samples)
# plot_confidence_intervals(ffts, xf, yf)


data = pd.read_csv('pages/app_data/current_data.csv')
t = data.x
signal = data.y

# Generate a sample signal: a sum of two sine waves with noise
sampling_rate = len(t)  # Sampling rate in Hz
T = 1.0 / sampling_rate  # Sampling interval

# Compute the Fourier Transform using FFT
fft_result = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(t.size, T)

# Take the magnitude of the FFT and keep only the positive frequencies
fft_magnitude = 2.0 / t.size * np.abs(fft_result[:t.size // 2])
fft_freqs = fft_freqs[:t.size // 2]
samples, ffts = generate_samples(data.x, data.y, number_of_samples, var)
plot_samples(samples=samples, x=data.x, num=50)
# print(f'{fft_freqs.shape = }, {fft_magnitude.shape = }')
plot_confidence_intervals(ffts, fft_freqs, fft_magnitude)