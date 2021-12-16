#Basic
import numpy as np
import pandas as pd

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Model
import pickle

import streamlit as st 

filename = 'knn.pkl'
clf = pickle.load(open(filename, 'rb'))

dataset = pd.read_csv("heart.csv")

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


def predict_note_authentication(X):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: cp
        in: query
        type: number
        required: true
      - name: trestbps
        in: query
        type: number
        required: true
      - name: chol
        in: query
        type: number
        required: true
      - name: fbs
        in: query
        type: number
        required: true
      - name: thalach
        in: query
        type: number
        required: true
      - name: exang
        in: query
        type: number
        required: true
      - name: oldpeak
        in: query
        type: number
        required: true
      - name: slope
        in: query
        type: number
        required: true
      - name: ca
        in: query
        type: number
        required: true
      - name: thal
        in: query
        type: number
        required: true                                        
    responses:
        200:
            description: The output values
        
    """
   
    prediction=clf.predict(X)
    #print(prediction)

    outputstr = 'No Presence of Heart Disease'

    #print(prediction[0])
    if(int(prediction[0]) != 0):
    	outputstr = 'Presence of Heart Disease'
    return outputstr



def main():
    st.title("Heart Disease Predict")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Predict ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.number_input("age",step=1.,format="%.2f") 
    sex= st.number_input("sex",step=1.,format="%.2f")
    cp= st.number_input("cp",step=1.,format="%.2f")
    trestbps = st.number_input("trestbps",step=1.,format="%.2f")
    chol = st.number_input("chol",step=1.,format="%.2f")
    fbs = st.number_input("fbs",step=1.,format="%.2f")
    restecg = st.number_input("restecg",step=1.,format="%.2f")
    thalach = st.number_input("thalach",step=1.,format="%.2f")
    exang = st.number_input("exang",step=1.,format="%.2f")
    oldpeak = st.number_input("oldpeak",step=1.,format="%.2f")
    slope = st.number_input("slope",step=1.,format="%.2f")
    ca = st.number_input("ca",step=1.,format="%.2f")
    thal = st.number_input("thal",step=1.,format="%.2f")

    inpu = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal]]
    test_dataset = pd.DataFrame(inpu, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    test_dataset = pd.get_dummies(test_dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    test_dataset[columns_to_scale] = standardScaler.transform(test_dataset[columns_to_scale])
    test_dataset = test_dataset.reindex(columns = dataset.columns, fill_value=0)

    #print(test_dataset.columns)
    X = test_dataset.drop(['target_0', 'target_1'], axis = 1)

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(X)
    st.success(result)
    if st.button("About"):
        st.text("Heart Disease Predictor")
        st.text("Final Year Project")

if __name__=='__main__':
    main()