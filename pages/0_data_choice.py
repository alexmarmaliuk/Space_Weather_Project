import os
import sys
import streamlit as st
import matplotlib.pyplot as plt

from pages.additional.preprocessing import *
from pages.additional.plotting import *



# def list_files(directory):
#     print(directory)
#     filenames = os.listdir(directory)
#     return filenames


def list_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Create a relative path like 'subfolder/subfile'
            rel_dir = os.path.relpath(root, directory)
            rel_file = os.path.join(rel_dir, filename) if rel_dir != '.' else filename
            all_files.append(rel_file)
    return all_files


data_directory = os.path.abspath("./data")
filenames = list_files(data_directory)

st.sidebar.header("Data Choice")
st.markdown(r'##### File choice.')
st.markdown(r'Note 1: It will be centered ($$data := data - \overline{data}$$).')
st.markdown(r'Note 2: File must inlude a line which starts with "Time" just above data lines.')
selected_file = st.selectbox("Select file with data:", filenames)

with open(('pages/app_data/data_choice.txt'), 'w') as file:
    file.write(selected_file)
    
st.write(selected_file)

x, y = load_data(os.path.join(data_directory, selected_file))
y = center_data(y)

##########
# st.markdown(r'$sin(2 \pi at) + sin(2 \pi bt)$.')
# a = st.slider('a', min_value=0, max_value=150, value=50)
# b = st.slider('b', min_value=0, max_value=150, value=120)

# domain = np.linspace(0, 9, 1000)
# f = lambda t : np.sin(2*np.pi*a*t) + np.sin(2*np.pi*b*t)
# y = f(domain)
# x = domain
##########

fig, ax = plt.subplots()
# plt.figure(figsize=(20, 8))
ax.plot(x, y)
ax.set_title(selected_file)
fig.set_size_inches(20,8)
plt.ylabel("Signal")
plt.xlabel("Time")
st.pyplot(fig)
st.write(f'number of points is {len(x)}')

df = pd.DataFrame({
    'x': x,
    'y_orig': y,
    'y': y,
})
st.write(f'x from {x[0]} to {x[-1]}')
df.to_csv('pages/app_data/current_data.csv')