import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pages.additional.sup import *

def plot(x, y, label='', fig_size=15):
    fig = plt.figure(figsize=(fig_size, fig_size/2))
    plt.plot(x, y, label=label)
    plt.show()

def plot_smooth(x, y, y_orig, label='Plot', type='pyplotly', fig_size=(16, 8)):
    if (type=='pyplot'):
        # plt.figure(figsize=fig_size)
        plt.plot(x, y_orig, label='Noisy Data')
        plt.plot(x, y, label='Filtered Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(label)
        plt.show()
        st.pyplot()
    elif type=='plotly':
        width = 800
        height = 600
        trace1 = go.Scatter(x=x, y=y, mode='lines', name='Filtered Data')
        trace2 = go.Scatter(x=x, y=y_orig, mode='lines', name='Noisy Data')
        layout = go.Layout(
            title=label,
            xaxis_title='X',
            yaxis_title='Y',
            width=width,
            height=height,
            legend_title="Legend"
        )
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        st.plotly_chart(fig)

def plot_single(x, y, label='Plot', type='plotly', y_scale=None, x_scale=None, x_label='X', y_label='Y'):
    if (type=='pyplot'):
        plt.plot(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if (x_scale is not None):
            plt.xscale(x_scale)
        if (y_scale is not None):
            plt.yscale(y_scale)
        plt.legend()
        plt.grid()
        plt.title(label)
        plt.show()
        st.pyplot()
    elif type=='plotly':
        st.write('plotting with plotly...')
        width = 800
        height = 600
        trace1 = go.Scatter(x=x, y=y, mode='lines', )

        layout = go.Layout(
            title=label,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            legend_title="Legend",
            xaxis_type=x_scale,
            yaxis_type=y_scale,
        )
        fig = go.Figure(data=[trace1], layout=layout)
        st.plotly_chart(fig)
        
# def plotWavelet():
    
    
    
def plot_samples(
    samples,
    x,
    num=5,
):
    elements = np.random.choice(samples.shape[0], size=num, replace=False)
    ys = samples[elements, :]
    # print(f'{ys.shape=}')
    # print(f'{ys[0]=}')
    plt.figure(figsize=(15, 5))
    for sample in ys:
        plt.plot(x, sample, color=random_color_hex())
        
    plt.show()
    st.pyplot()
    
    
def plot_confidence_intervals(
    samples,
    x,
    original_fft=None,
):
    means = np.mean(np.abs(samples), axis=0)
    stds = np.std(np.abs(samples), axis=0)
    lower = means - 1.96 * stds
    upper = means + 1.96 * stds
    plt.figure(figsize=(15, 5))
    plt.plot(x, np.abs(means), label='Mean Amplitude')
    plt.fill_between(x, np.abs(lower), np.abs(upper), alpha=0.5, label='95% Confidence Interval')
    
    if (original_fft is not None):
        plt.plot(x, np.abs(original_fft), color='red', label='Original')
    
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    st.pyplot()
