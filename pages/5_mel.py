import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import altair as alt


with open(('pages/app_data/data_choice.txt'), 'r') as file:
    data_choice = file.read()
    
st.markdown(f'#### Working with **{data_choice}** data.')

datapath = 'pages/app_data/current_data.csv'
data = pd.read_csv(datapath)


y = np.array(data.y)
x = np.array(data.x)
plt.figure(figsize=(15, 5))
plt.plot(x,y)
plt.show()
st.pyplot()
sr = st.number_input('Sample rate', value=22050, step=1000)
fft_n = st.select_slider(label='FFT window',options=np.power(2, np.arange(25)), value=1024)

# Compute the mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_n)

# Convert to dB (log scale)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plotting
plt.figure(figsize=(20, 4))
# plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
st.pyplot()
