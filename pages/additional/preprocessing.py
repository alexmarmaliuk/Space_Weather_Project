import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from csaps import csaps
from sklearn.model_selection import KFold

import re

def load_data(file_path):
    # file_path = 'materials/Для студентов (Геомагнитное поле)/Для студентов (Геомагнитное поле)/1-14 Jan 2024, Surlari, B-x.txt'
    x = []
    y = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Time'):
                break
            if re.match(r'^\d', line):
                if line.strip(): 
                    parts = line.split()  
                    x.append(float(parts[0]))
                    y.append(float(parts[-1]))
                break
        for line in file:
            if line.strip(): 
                parts = line.split()  
                x.append(float(parts[0]))
                y.append(float(parts[-1]))

    x = np.array(x)
    y = drop_value(np.array(y))
    return x, y
    
def center_data(y):
    return y - y.mean()

def drop_value(data, column=None, target_value=99999):
    if isinstance(data, pd.DataFrame):
        by = data[column].values.copy()
    else:
        by = data.copy()

    mask = np.isclose(by, target_value)
    new_by = by.copy()
    avg_neighbors = (by[:-2] + by[2:]) / 2
    replacement_indices = np.where(mask[1:-1])[0] + 1
    new_by[replacement_indices] = avg_neighbors[mask[1:-1]]
    if isinstance(data, pd.DataFrame):
        data[column] = new_by
        return data
    else:
        return new_by
    
    
def cross_validate_smoothing(x, y, smoothing_values, n_splits=5, shuffle=False, rs=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)
    best_smoothing = None
    best_error = float('inf')
    for s in smoothing_values:
        cv_errors = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_pred = csaps(x_train, y_train, x_test, smooth=s)
            error = np.mean((y_pred - y_test)**2)
            cv_errors.append(error)
        
        avg_error = np.mean(cv_errors)
        if avg_error < best_error:
            best_error = avg_error
            best_smoothing = s

    return best_smoothing

def tricube_weight(d, bandwidth):
    """Tricube kernel function."""
    u = np.abs(d / bandwidth)
    return (1 - u**3)**3 * (u < 1)

def gaussian_weight(d, bandwidth):
    """Gaussian kernel function."""
    return np.exp(-0.5 * (d / bandwidth)**2)

def loess_smoothing(x, y, x_eval, bandwidth=1, kernel='tricube', degree=1):
    y_smooth = np.zeros_like(x_eval)

    for i, x0 in enumerate(x_eval):
        distances = np.abs(x - x0)
        
        if kernel == 'tricube':
            weights = tricube_weight(distances, bandwidth)
        elif kernel == 'gaussian':
            weights = gaussian_weight(distances, bandwidth)
        else:
            raise ValueError("Unknown kernel")

        W = np.diag(weights)
        X_mat = np.vstack([x**p for p in range(degree + 1)]).T
        XTWX = X_mat.T @ W @ X_mat
        XTWY = X_mat.T @ W @ y

        theta = np.linalg.pinv(XTWX) @ XTWY  # Solve weighted least squares
        y_smooth[i] = sum(theta[p] * x0**p for p in range(degree + 1))

    return y_smooth