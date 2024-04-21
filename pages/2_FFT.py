import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from sup import *
from pages.additional.plotting import *

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


plot_single(x=xf, y=np.abs(yf),label='FFT', type=plot_type, y_scale='log', x_scale='log', y_label='Power', x_label='Frequency')

# TODO: 4-dim transform


# fig_size = 15
# fig, ax = plt.subplots()
# plt.plot(xf, np.abs(yf))
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# plt.yscale('log')
# plt.xscale('log')
# # # plt.show()
# # st.pyplot(fig)
# st.write('plotting with plotly...')
# xf  = xf#np.array([1,10,100,1000])
# yf  = np.abs(yf) #np.array([1,10,100,1000])

# width = 800
# height = 600
# trace1 = go.Scatter(x=xf, y=yf, mode='lines', )

# layout = go.Layout(
#     title='label',
#     xaxis_title='x_label',
#     yaxis_title='y_label',
#     width=width,
#     height=height,
#     legend_title="Legend",
#     xaxis_type='log',
#     yaxis_type='log',
# )
# fig = go.Figure(data=[trace1], layout=layout)
# # width = 800
# # height = 600
# # trace1 = go.Scatter(x=xf, y=yf, mode='lines', )

# # layout = go.Layout(
# #     title='label',
# #     xaxis_title='x_label',
# #     yaxis_title='y_label',
# #     width=width,
# #     height=height,
# #     legend_title="Legend",
# #     xaxis_type='log',
# #     yaxis_type='log',
# # )
# # fig = go.Figure(data=[trace1], layout=layout)
# st.plotly_chart(fig)