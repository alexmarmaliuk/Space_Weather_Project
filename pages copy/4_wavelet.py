import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import waipy
import altair as alt

from pages.additional.sup import *
from pages.additional.plotting import *

st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Wavelet Sprectrum")

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

alpha = 6
mother = 'Morlet'

result = waipy.cwt(
    data=data.y,
    dt=s0,
    pad=1,
    dj=ds,
    s0=s1, 
    j1=(s1-s0)/ds,
    lag1=lag,
    param=alpha,
    mother='Morlet',
    name='Morlet'
)


# import matplotlib.cm as cm
# import waipy.cwt.wavetest as wavetest  # ← This is where wavelet_plot is located

# # Patch waipy's broken reference to mpl.cm.get_cmap
# if not hasattr(wavetest.mpl.cm, "get_cmap"):
#     wavetest.mpl.cm.get_cmap = cm.get_cmap


label = 'Morlet wavelet'
fig = plt.figure(figsize=(10, 6))
waipy.wavelet_plot(label, data.x, data.y, ds, result, plot_power=False, contour_levels=20, yscale='log', ylabel='Period')
st.pyplot(fig)
