import numpy as np
import scipy as sp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import waipy
import pandas as pd
import altair as alt

from pages.additional.preprocessing import *
from pages.additional.plotting import *
from pages.additional.sup import *

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)
x, y = data.x, data.y
size = len(x)
sample_rate = 24 * 60
xf = sp.fft.fftfreq(size, 1 / sample_rate)
yf = sp.fft.fft(np.array(y))

var = st.number_input(label='noise variance', value=1)
number_of_samples = st.number_input(label='sampling size', value=100, min_value=1, step=500)
# number_of_samples = st.select_slider(label='sampling size', options=[1,])

samples, ffts = generate_samples(x, y, 1000, var)
plot_samples(samples=samples, x=x, num=number_of_samples)
plot_confidence_intervals(ffts, xf, yf)