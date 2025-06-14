import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import pandas as pd
import numpy as np

def load_data(file_path):
    # Initialize lists to store data
    data = []
    header = ['Time', 'By']  # Default header if none found in the file

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Time'):
                header = line.strip().split()  # Extract column names
                break
        else:
            # No header line found; rewind file and read all lines as data
            file.seek(0)

        for line in file:
            if line.strip(): 
                parts = line.split()  # Split line into values
                data.append([float(value) for value in parts])

    df = pd.DataFrame(data, columns=header)
    return df

    
def center_data(y):
    return y - y.mean()

def drop_value(df, column, target_value=99999):
    by = df[column].values.copy()
    mask = np.isclose(by, target_value)
    new_by = by.copy()
    avg_neighbors = (by[:-2] + by[2:]) / 2
    replacement_indices = np.where(mask[1:-1])[0] + 1
    new_by[replacement_indices] = avg_neighbors[mask[1:-1]]
    df[column] = new_by
    return df

