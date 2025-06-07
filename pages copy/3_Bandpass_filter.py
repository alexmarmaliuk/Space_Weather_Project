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

# Adjusting the sample rate slider
sr = st.slider('Sample Rate', min_value=200, max_value=20000, value=1000)

low, high = st.slider(
    'Select Low and High Cut Frequencies',
    min_value=500, max_value=20000, value=(50, round(sr/2 - 1)), step=10)

plot_type = st.selectbox('Plotting lib', ['plotly', 'pyplot'])

plot_single(data.x, bandpass(data.y, highcut=high, lowcut=low, fs=sr), f"{sr=}, {low=}, {high=}", plot_type, x_label='Time', y_label='Signal')