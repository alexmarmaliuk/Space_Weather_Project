import streamlit as st
import numpy as np
import pandas as pd
import pywt
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

s0 = st.slider('Smallest scale', min_value=1, max_value=4001, value=401, step=200)
s1 = st.slider('Largest scale', min_value=1, max_value=4001, value=1001, step=200)
ds = st.slider('Scale step', min_value=0.1, max_value=100., value=0.1)
nvoice = st.slider('octaves / scale', min_value=3, max_value=20, value=6)
nvoice = 6  # Number of octaves per scale
alpha = 6  # Alpha parameter for Morlet wavelet
mother = 'Morlet'  # Wavelet type

result = waipy.cwt(data.y, s0, s1, ds, nvoice, (s1-s0)/ds, alpha, 6, mother=mother, name="Morlet")

label = 'Morlet wavelet'
plt.figure(figsize=(10, 6))
waipy.wavelet_plot(label, data.x, data.y, ds, result, plot_power=False, contour_levels=20, yscale='log', ylabel='Period')
plt.show()
st.pyplot()

