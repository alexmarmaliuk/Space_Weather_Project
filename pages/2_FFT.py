import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from sup import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')

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


fig_size = 15
fig, ax = plt.subplots()
plt.plot(xf, np.abs(yf))
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.yscale('log')
plt.xscale('log')
# plt.show()
st.pyplot(fig)

# plt.plot(xf[size//2 - 100000 + 10 : size//2], np.abs(yf)[size//2 - 100000 + 10 : size//2 ])
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# plt.yscale('log')
# plt.xscale('log')
# # plt.show()
# st.pyplot(fig)