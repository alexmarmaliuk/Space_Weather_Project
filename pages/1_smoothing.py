import os
import streamlit as st
import altair as alt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import savgol_filter

from pages.additional.preprocessing import *
from pages.additional.plotting import *

# turn off empty st.pyplot() warning
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Smoothing Types')
st.sidebar.header("Smoothing")
# st.markdown('##### *TODO*: smoothing params for each type.')

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)

with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

    
selected_option = st.selectbox('Select Smoothing to Apply', ['Savitzky-Golay', 'Exponential', 'Moving Average'])
plot_type = st.selectbox('Plotting lib', ['plotly', 'pyplot'])
st.markdown('Note: Plotting lib influences only representation.')

if (selected_option == 'Moving Average'):
    window_size = st.number_input("Enter window size:", min_value=1, max_value=1000, value=100, step=50)

    data['y'] = pd.Series(data.y_orig).rolling(window=window_size).mean()
    st.write(f'type = {plot_type}')
    plot_smooth(data.x, data.y, data.y_orig, selected_option, plot_type)
    data.to_csv(datapath)

elif (selected_option == 'Exponential'):

    st.markdown(r'Parameter $$\alpha = factor \cdot 10^{\beta}$$.')

    beta = st.slider(r"$$\beta$$", min_value=-10., max_value=0., step=0.05, value=-1.)
    factor = st.slider(r"$$factor$$", min_value=1., max_value=10., step=0.01)
    a = factor * 10**beta

    model = ExponentialSmoothing(data.y_orig)
    fit_model = model.fit(smoothing_level=a)
    data['y'] = fit_model.fittedvalues
    st.write(f'type = {plot_type}')
    plot_smooth(data.x, data.y, data.y_orig, selected_option, plot_type)
    data.to_csv(datapath)

    
elif (selected_option == 'Savitzky-Golay'):
    window_size = st.number_input("Enter window size:", min_value=1, max_value=1000, value=100, step=10)
    ord = st.number_input("Enter polynom order:", min_value=1, max_value=5, value=3, step=1)
    data['y'] = savgol_filter(data.y_orig, window_length=window_size, polyorder=ord)
    st.write(f'type = {plot_type}')
    plot_smooth(data.x, data.y, data.y_orig, selected_option, plot_type)
    data.to_csv(datapath)

