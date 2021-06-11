#! /usr/bin/env python3
# coding=utf-8

import pandas as pd
import joblib
import numpy as np
import logging
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


df=pd.read_csv("Data/Customer-Churn.csv")

def remove_cols(df):
    '''
    Check and remove the unwanted columns
    input: df
    output: cleaned dataframe    
    '''
    Constant_Values = df.columns[df.eq(df.iloc[0]).all()].tolist()
    Duplicate_Columns = df.columns[df.T.duplicated(keep='first').T]  # Only report second column as duplicate
    
    df = df.drop(Constant_Values, axis=1)
    df = df.drop(Duplicate_Columns, axis=1)
    df = df.drop(['customerID','Dependents','PhoneService','MultipleLines', 'PaperlessBilling','PaymentMethod'], axis = 1)
    return df

df = remove_cols(df)

def missing_values_table(df):
    '''
    Check the missing values in the data columns
    input: df
    output: Dataframe of columns and their missing value percent    
    '''
    
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    missing_val = df.isnull().sum()
    missing_val_percent = 100 * missing_val/ len(df)
    mis_val_table = pd.concat([missing_val, missing_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    return mis_val_table_ren_columns

missing_values = missing_values_table(df)

# droping the columns
train_features = df.drop(columns=['Churn'])
label = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(train_features, label, test_size=0.20, random_state=42)


cat_cols = list(X_train.select_dtypes('object').columns)
num_cols = list(X_train.select_dtypes('number').columns)

num_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                  ('scaler', StandardScaler())])
cat_transform = Pipeline(steps=[('onehotenc', OneHotEncoder(handle_unknown='ignore', sparse=False))
                                         ])
col_transformer = ColumnTransformer(transformers=[  ('num_transform',num_transform, num_cols),
                                                    ('cat_transform', cat_transform, num_cols)
                                                  ], remainder='drop')


print("Model building with hyperparameter tunned during Research Phase")

def build_model():
    '''
    Machine Learning classification model function that executes following steps:
      1. Building Machine Learning pipeline
      2. Running GridSearchCV for Hyper-parameter tunning
      
      input: None
    output: RandomSearch best model.
    '''
    pipeline_clf = Pipeline([
                     ('transform_column', col_transformer),
                     ('clf', RandomForestClassifier(random_state=42,n_estimators=100, min_samples_split=2, max_depth=90, criterion='gini', class_weight='balanced', bootstrap=True ))
                     ]) 

    best_clf = pipeline_clf
    return best_clf
    
model = build_model()
model.fit(X_train,y_train)

joblib.dump(model, 'model.joblib')
