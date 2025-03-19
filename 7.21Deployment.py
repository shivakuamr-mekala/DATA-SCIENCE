#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[ ]:

train = pd.read_csv('Titanic.csv')
@st.cache_data(show_spinner="Fetching data...")
def read_data():
    df = pd.read_csv('Titanic.csv')
    return df
@st.cache_resource
def get_features_encoder(data):
    oe = OneHotEncoder(dtype=int)
    train2 = oe.fit_transform(data[['Sex','Embarked']])
    train2=pd.DataFrame(train2.toarray(),columns=['Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])
    df = pd.concat([data,train2],axis=1)
    df.drop(['Sex','Embarked'],axis=1,inplace=True)
    return df
@st.cache_resource
def get_encoder(data,x_pre):
    oe = OneHotEncoder(dtype=int)
    oe.fit(data[['Sex','Embarked']])
    train2= oe.transform(x_pre[['Sex','Embarked']])
    train2=pd.DataFrame(train2.toarray(),columns=['Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])
    df = pd.concat([x_pre,train2],axis=1)
    df.drop(['Sex','Embarked'],axis=1,inplace=True)
    return df
@st.cache_resource(show_spinner="Training model...")
def train_model(data):
    x=data.drop(['Survived'],axis=1)
    y=data['Survived']
    model=LogisticRegression(max_iter=350)
    model.fit(x,y)
    return model
st.cache_data(show_spinner="Making a prediction...")
def make_prediction(_model,data,X_pred):
    encoded_features= get_encoder(data,X_pred)
    pred = _model.predict(encoded_features)
    prob = _model.predict_proba(encoded_features)[0][1]
    return pred[0],prob


# In[ ]:


if __name__ == "__main__":
    st.title('Model Deployment: Logistic Regression: Titanic')
    st.subheader("Step 1: Select the values for prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Choose a Passenger class & Age")
        Pclass = st.selectbox("class", options=train.Pclass.unique().tolist())
        Age = st.slider("Age", min_value= 1, max_value= 100, value=25, step=1)
    with col2:
        st.write("Choose SibSp & Parch")
        SibSp = st.slider("SibSp", min_value=0, max_value=20, value=0,step=1)
        Parch = st.slider("Parch", min_value= 0, max_value= 20, value=0, step=1)
    with col3:
        st.write("Choose Gender & City")
        Sex = st.selectbox("Gender", options=train.Sex.unique().tolist())
        Embarked = st.selectbox("City", options=train.Embarked.unique().tolist())    
    st.subheader("Step 2: Ask the model for a prediction")
    
    pred_btn = st.button("Predict", type="primary")
    if pred_btn:
        df = read_data()
        en = get_features_encoder(df)
        LR = train_model(en)
        data={'Pclass':[Pclass],'Age':[Age],'SibSp':[SibSp],'Parch':[Parch],'Sex':[Sex],'Embarked':[Embarked]}

        x_predict = pd.DataFrame(data)
        
        predict,probability = make_prediction(LR,df,x_predict)

        nice_pred = "this person survived" if predict == 1 else "this person not survived"
        st.write(nice_pred)
        st.write(f'this persons survival probability is : {round(probability,2)}')
    

