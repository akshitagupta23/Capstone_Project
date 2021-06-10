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

from churn_model_train import missing_values_table, remove_cols

st.set_page_config(layout = "wide")

st.title('Customer Churn')
st.markdown("<h3></h3>", unsafe_allow_html=True)
st.image('images/customer_churn.png', caption=None, width=None, use_column_width=True)

FEATURES=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
           'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges']

model = joblib.load('model.joblib')

def run():
    
    

    file_upload = st.file_uploader("Upload file", type=["csv"])

    if file_upload is not None:
        input_df = pd.read_csv(file_upload)
        #input_df = missing_values_table(input_df)
        input_df = input_df.drop(['customerID','Dependents','PhoneService','MultipleLines', 'PaperlessBilling','PaymentMethod'], axis = 1)
        predict = model.predict(input_df)
        st.write(predict)
            

if __name__ == '__main__':
    run()
