import streamlit as st
import altair as alt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels
import pandas as pd

from sup import *

# turn off empty st.pyplot() warning

st.set_option('deprecation.showPyplotGlobalUse', False)

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Smoothing Types')

st.sidebar.header("Smoothing")
st.markdown('##### *TODO*: smoothing params for each type.')

base_data = pd.read_csv('filtered.csv')

def plot_data(selected_data, plot_title=''):
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 5))
    
    if (selected_data == 'Savitzky-Golay'):
        Y = base_data.y_xg
    elif (selected_data == 'Exponential'):
        Y = base_data.y_exp
    elif (selected_data == 'Moving Average'):
        Y = base_data.y_ma
        
    plt.plot(base_data.x, Y, label='Filtered Data')
    plt.plot(base_data.x, base_data.y, label='Noisy Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.legend()
    plt.title(plot_title)
    plt.show()
    st.pyplot()
    
selected_option = st.selectbox('Select Smoothing to Apply', ['Savitzky-Golay', 'Exponential', 'Moving Average'])
    
    # Plot selected data
plot_data(selected_option, selected_option)
