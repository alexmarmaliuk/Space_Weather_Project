import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from sup import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

df = pd.read_csv('filtered.csv')
df.y = df.y_xg

# Plot 1
yf = sp.fft.fft(np.array(df.y))
duration = 14
sample_rate = 24 * 60
xf = sp.fft.fftfreq(20160, 1 / sample_rate)
fig_size = 15

fig, ax = plt.subplots()
# ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
# ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
# plt.plot(xf, np.abs(yf))
plt.plot(xf[20160//2 - 100000 + 10 : 20160//2], np.abs(yf)[20160//2 - 100000 + 10 : 20160//2 ])
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.yscale('log')
plt.xscale('log')
# plt.show()
st.pyplot(fig)


# plot 2
low, high = st.slider(
    'Select Low and High Cut Frequencies',
    min_value=500, max_value=4000, value=(850, 900), step=10)

# Adjusting the sample rate slider
sr = st.slider('Sample Rate', min_value=200, max_value=2000, value=1000)

# Plotting (simplified example)
fig, ax = plt.subplots()
ax.plot(df['x'], bandpass(df, highcut=high, lowcut=low, fs=sr))
ax.set_title(f"Low={low}, High={high}, Sample Rate={sr}")

# Display the plot
st.pyplot(fig)