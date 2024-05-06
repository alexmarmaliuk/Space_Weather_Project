import streamlit as st
import numpy as np
import pandas as pd
# import pywt
import waipy
import altair as alt

from pages.additional.sup import *
from pages.additional.plotting import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Wavelets")

with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

s0 = st.slider('Smallest scale', min_value=1, max_value=4000, value=401, step=1)
s1 = st.slider('Largest scale', min_value=1, max_value=4000, value=2*s0, step=1)
ds = st.slider('Scale step', min_value=0.1, max_value=100., value=0.1)
nvoice = st.slider('octaves / scale', min_value=3, max_value=20, value=6)
lag = st.number_input(label='autocorrelation', value=1)
alpha = 6  # Alpha parameter for Morlet wavelet
mother = 'Morlet'  # Wavelet type

# result = waipy.cwt(data.y, s0, s1, ds, nvoice, (s1-s0)/ds, alpha, 6, mother=mother, name="Morlet")
result = waipy.cwt(
    data=data.y,
    # Time-sample of the vector. Example: Hourly, daily, monthly, etc...
    dt=s0,
    # Pad the time series with zeroes to next pow of two length (recommended).
    pad=1,
    # Divide octave in sub-octaves. If dj = 0.25 this will do 4 sub-octaves per octave
    dj=ds,
    # The maximum frequency resolution
    s0=s1, 
    j1=(s1-s0)/ds,
    # autocoorelation for red noise backgorund
    lag1=lag,
    # parameter for Morlet wavelet
    param=alpha,
    mother='Morlet',
    name='Morlet'
)

label = 'Morlet wavelet'
plt.figure(figsize=(10, 6))
waipy.wavelet_plot(label, data.x, data.y, ds, result, plot_power=False, contour_levels=20, yscale='log', ylabel='Period')
plt.show()
st.pyplot()

