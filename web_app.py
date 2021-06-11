#! /usr/bin/env python3
# coding=utf-8


import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

from churn_model_train import missing_values_table

st.set_page_config(layout = "wide")

st.title('Customer Churn')
st.markdown("<h3></h3>", unsafe_allow_html=True)
st.image('images/customer_churn.png', caption=None, width=None, use_column_width=True)

_FEATURES=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
           'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges']

model = joblib.load('model.joblib')

def predict_churn(gender, SeniorCitizen, Partner,tenure, PhoneService, 
           InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
           StreamingMovies, Contract, MonthlyCharges, TotalCharges):
    
    """Let's Authenticate Churn Prediction 
    responses:
        200:
            description: The output values
        
    """
   
    prediction=model.predict([[gender, SeniorCitizen, Partner,tenure, PhoneService, 
           InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
           StreamingMovies, Contract, MonthlyCharges, TotalCharges]])
    print(prediction)
    return prediction

# Selecting the mode of prediction
predict_mode = st.sidebar.radio(
    "Choose mode to predict?",
    ("Online", "Batch"))


def run():
    if predict_mode == 'Online':
    
        customerID = st.text_input('customerID')
        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            gender = st.selectbox('gender', ['Male', 'Female'])
        with col2:
            SeniorCitizen = st.selectbox('SeniorCitizen', [0, 1])
        with col3:
            Partner = st.selectbox('Partner', ['Yes', 'No'])   
        with col4:
            Dependents = st.selectbox('Dependents', ['Yes', 'No'])
        col5, col6, col7, col8 = st.beta_columns(4) 
        with col5:
            tenure = st.slider('tenure', 0, 50, 10)
        with col6:
            PhoneService = st.selectbox('PhoneService', ['Yes', 'No'])
        with col7:
            MultipleLines = st.selectbox('MultipleLines', ['No', 'Yes', 'No phone service'])
        with col8:
            InternetService = st.selectbox('InternetService', ['Fiber optic', 'DSL', 'No'])
        col9, col10, col11, col12 = st.beta_columns(4)
        with col9:
            OnlineSecurity =  st.selectbox('OnlineSecurity', ['No', 'Yes', 'No phone service'])
        with col10:
            OnlineBackup = st.selectbox('OnlineBackup', ['No', 'Yes', 'No internet service']) 
        with col11:
            DeviceProtection = st.selectbox('DeviceProtection', ['No', 'Yes', 'No internet service']) 
        with col12:
            TechSupport = st.selectbox('TechSupport', ['No', 'Yes', 'No internet service'])
        col13, col14, col15, col16= st.beta_columns(4)
        with col13:
            StreamingTV = st.selectbox('StreamingTV', ['No', 'Yes', 'No internet service'])
        with col14:
            StreamingMovies = st.selectbox('StreamingMovies', ['No', 'Yes', 'No internet service'])
        with col15:
            Contract = st.selectbox('Contract', ['Month-to-month', 'Two Year', 'One Year'])
        with col16:
            PaperlessBilling= st.selectbox('PaperlessBilling', ['Yes', 'No'])
        col17, col18, col19 = st.beta_columns(3)
        with col17:
            PaymentMethod = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        with col18:
            MonthlyCharges = st.slider('MonthlyCharges', 0, 200, 50)
        with col19:
            TotalCharges = st.slider('TotalCharges', 0, 10000, 2000) 

        predict=""
           
        if st.button("Predict"):
            predict = predict_churn(gender, SeniorCitizen, Partner,tenure, PhoneService, 
           InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
           StreamingMovies, Contract, MonthlyCharges, TotalCharges)
        st.success('Customer will churn:  {}'.format(predict))
            
    if predict_mode == 'Batch':

        file_upload = st.file_uploader("Upload file", type=["csv"])

        if file_upload is not None:
            input_df = pd.read_csv(file_upload)
            #input_df = missing_values_table(input_df)
            input_df = input_df.drop(['customerID','Dependents','PhoneService','MultipleLines', 'PaperlessBilling','PaymentMethod'], axis = 1)
            predict = model.predict(input_df)
            st.write(predict)
            

if __name__ == '__main__':
    run()
