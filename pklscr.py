import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

"""
# COVID-19 Survival Predictor :mask:
![keep calm](https://mir-s3-cdn-cf.behance.net/project_modules/disp/b8d3cd96548519.5eb10b3277638.gif)
"""
with open('prediction_model.pkl', 'rb') as file:
    Pickled_lm =pickle.load(file)
    age=st.slider("Age Selector",min_value=1,max_value=110, value=20)
    k1=st.button("MALE");
    k0=st.button("FEMALE")
    if(k1==True):
        gender=1
        k="MALE"
    else:
        gender=0
        k="FEMALE"
    preds=Pickled_lm.predict([[gender,age]])[0][0]

    x_high=Pickled_lm.predict([[0,1]])[0][0] #Children have highest probs
    x_low=Pickled_lm.predict([[1,150]])[0][0] #Adults have lower probs
    final=(preds-x_low)/(x_high-x_low) #Normalizing
    if(k1==True or k0==True):
        st.subheader("Predictions for \n Age: {age} and Gender: {gender}".format(age=age,gender=k))
        st.header(round(final*100,2))