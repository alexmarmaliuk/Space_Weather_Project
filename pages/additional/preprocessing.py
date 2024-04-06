import numpy as np
import scipy as sp
import pandas as pd

def load_data(file_path):
    # file_path = 'materials/Для студентов (Геомагнитное поле)/Для студентов (Геомагнитное поле)/1-14 Jan 2024, Surlari, B-x.txt'
    x = []
    y = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Time (Days)'):
                break
        for line in file:
            if line.strip(): 
                parts = line.split()  
                x.append(float(parts[0]))
                y.append(float(parts[-1]))

    x = np.array(x)
    y = np.array(y)
    
def center_data(y):
    return y - y.mean()