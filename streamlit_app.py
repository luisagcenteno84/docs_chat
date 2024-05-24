import streamlit as st
import numpy as np
import pandas as pd

st.title("test")

with st.chat_message("assistant"):
    st.write("Hello human")
    st.bar_chart(np.random.randn(30,3))


data = st.file_uploader("Upload a Dataset", type=["pdf"])

st_df = ''
df = None

if data is not None:
	df = pd.read_csv(data)
	st_df = st.dataframe(df.head())
                        
print(st_df)

st.write(df)