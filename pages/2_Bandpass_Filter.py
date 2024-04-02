import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from sup import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

st.sidebar.header("Bandpass filter")

df = pd.read_csv('filtered.csv')
df.y = df.y_xg

st.markdown('''
            ### Nyquist-Shannon sampling criteria (Kotelnikov theorem)
            
            If our signal has the highest frequency $f_{max}$ then we need to sample it with 
            $$f_s > 2f_{max}$$
            
            ### Plot params
            ''')
            
            
# plot 2
low, high = st.slider(
    'Select Low and High Cut Frequencies',
    min_value=500, max_value=4000, value=(50, 2000), step=10)

# Adjusting the sample rate slider
sr = st.slider('Sample Rate', min_value=200, max_value=4000, value=1000)

# Plotting (simplified example)
fig, ax = plt.subplots()
ax.plot(df['x'], bandpass(df, highcut=high, lowcut=low, fs=sr))
ax.set_title(f"Low={low}, High={high}, Sample Rate={sr}")

# Display the plot
st.pyplot(fig)