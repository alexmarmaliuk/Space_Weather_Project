import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from pages.additional.sup import *
from pages.additional.plotting import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

st.sidebar.header("FFT")

plot_type = st.selectbox('Plotting lib', ['plotly', 'pyplot'])
st.markdown('Note: Plotting lib influences only representation.')

df = pd.read_csv('pages/app_data/current_data.csv')
# st.write(df.columns)

# Plot 1
size = len(df.x)
# st.write(size)
yf = sp.fft.fft(np.array(df.y))
duration = df['x'].iloc[-1] - df['x'].iloc[0]
# st.write(duration)
sample_rate = 24 * 60
xf = sp.fft.fftfreq(size, 1 / sample_rate)


plot_single(x=xf, y=np.abs(yf),label='FFT', type=plot_type, y_scale='log', x_scale='log', y_label='Power', x_label='Frequency')

# TODO: 3-dim transform