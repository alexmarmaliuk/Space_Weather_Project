import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def load_data(file_path):
    # file_path = 'materials/Для студентов (Геомагнитное поле)/Для студентов (Геомагнитное поле)/1-14 Jan 2024, Surlari, B-x.txt'
    x = []
    y = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Time'):
                break
        for line in file:
            if line.strip(): 
                parts = line.split()  
                x.append(float(parts[0]))
                y.append(float(parts[-1]))

    x = np.array(x)
    y = np.array(y)
    return x, y
    
def center_data(y):
    return y - y.mean()

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

def plot_single(x, y, label='Plot', type='plotly'):
        if (type=='pyplot'):
            plt.plot(x, y)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title(label)
            plt.show()
            st.pyplot()
        elif type=='plotly':
            st.write('dergdrg')
            width = 800
            height = 600
            trace1 = go.Scatter(x=x, y=y, mode='lines')

            layout = go.Layout(
                title=label,
                xaxis_title='X',
                yaxis_title='Y',
                width=width,
                height=height,
                legend_title="Legend"
            )
            fig = go.Figure(data=[trace1], layout=layout)
            st.plotly_chart(fig)