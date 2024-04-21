import os
import sys
import streamlit as st
import matplotlib.pyplot as plt

from pages.additional.preprocessing import *
from pages.additional.plotting import *



def list_files(directory):
    print(directory)
    filenames = os.listdir(directory)
    return filenames


data_directory = os.path.abspath("./data")
filenames = list_files(data_directory)

st.sidebar.header("Data Choice")
st.markdown(r'##### File choice.')
st.markdown(r'Note 1: It will be centered ($$data := data - \overline{data}$$).')
st.markdown(r'Note 2: File must inlude a line which starts with "Time" just above data lines.')
selected_file = st.selectbox("Select file with data:", filenames)

x, y = load_data(os.path.join(data_directory, selected_file))
y = center_data(y)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title(selected_file)
plt.xlabel("Value")
plt.ylabel("Time")
st.pyplot(fig)
st.write(f'number of points is {len(x)}')

df = pd.DataFrame({
    'x': x,
    'y_orig': y,
})
st.write(f'x from {x[0]} to {x[-1]}')
df.to_csv('pages/app_data/current_data.csv')