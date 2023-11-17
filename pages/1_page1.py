import streamlit as st
import requests
import json

st.title("This is first page")

col1,col2 = st.columns([1,1])

with col1:
    text_input = st.text_input("enter the text")


data = {"text":text_input}


with col2 :
    submit_button = st.button("Submit")

    
    if submit_button:
        res = requests.post(url = "http://127.0.0.1:8000/Image_Predict",data = json.dumps(data))

        st.text(res.text)
