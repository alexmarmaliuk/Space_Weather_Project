import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from pages.additional.sup import *

# Page title
st.set_page_config(page_title='Interactive Geomagnetic Field Variations Data Explorer', page_icon='📊')
st.title('📊 Demo')
# st.sidebar.success("Select a demo.")