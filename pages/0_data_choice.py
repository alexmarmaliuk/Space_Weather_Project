import os
import sys
import streamlit as st
import matplotlib.pyplot as plt

st.write(sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# from additional import 
from additional.preprocessing import *

def list_files(directory):
    print(directory)
    filenames = os.listdir(directory)
    return filenames


data_directory = os.path.abspath("./data")
# st.write(data_directory)
filenames = list_files(data_directory)

st.sidebar.header("Smoothing")
st.markdown(r'##### File choice.')
st.markdown(r'It will be centered ($$data := data - \overline{data}$$).')
selected_file = st.selectbox("Select file with data:", filenames)

x, y = load_data(selected_file)
y = center_data(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title(selected_file)
st.pyplot(fig)