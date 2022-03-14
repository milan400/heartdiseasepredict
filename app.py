#Basic
import numpy as np
import pandas as pd
import os

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Model
import pickle

import streamlit as st 

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, 'knn.pkl')


clf = pickle.load(open(filename, 'rb'))

dataset = pd.read_csv(os.path.join(dirname,'heart.csv'))

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


def predict_note_authentication(X):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: Sex
        in: query
        type: number
        required: true
      - name: Chest Pain Type
        in: query
        type: number
        required: true
      - name: Resting Blood Pressure
        in: query
        type: number
        required: true
      - name: Serum Cholestoral
        in: query
        type: number
        required: true
      - name: Fasting Blood Sugar
        in: query
        type: number
        required: true
      - name: Resting Electrocardiographic Results
        in: query
        type: number
        required: true
      - name: Maximum Heart Rate Achieved
        in: query
        type: number
        required: true
      - name: Exercise Induced Angina
        in: query
        type: number
        required: true
      - name: ST depression induced by exercise relative to rest 
        in: query
        type: number
        required: true
      - name: Slope of the peak exercise ST segment
        in: query
        type: number
        required: true
      - name: Number of major vessels (0-3) colored by flourosopy
        in: query
        type: number
        required: true
      - name: Thal
        in: query
        type: number
        required: true                                        
    responses:
        200:
            description: The output values
        
    """
   
    prediction=clf.predict(X)
    #print(prediction)

    pred_prob = clf.predict_proba(X)*100
    value_likely = pred_prob[0][1]

    outputstr = None

    #print(prediction[0])
    if(int(prediction[0]) != 0):
      outputstr = 'Presence of Heart Disease'+'('+str(value_likely)+'%)'
    else:
      outputstr='No Presence of Heart Disease'+'('+str(value_likely)+'%)'
    return outputstr



def main():
    st.title("Heart Disease Predict")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Predict ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    age = st.number_input("Age",step=1.,format="%.0f") 

    sex = 1 if (st.radio("Sex", ('Male', 'Female')))=='Male' else 0

    cp = st.radio("Chest Pain Type", ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'))

    if(cp=='Typical Angina'):
      cp= 0
    elif(cp=='Atypical Angina'):
      cp= 1
    elif(cp=='Non-Anginal Pain'):
      cp= 2
    else:
      cp= 3 

    trestbps = st.number_input("Resting Blood Pressure",step=1.,format="%.0f")
    chol = st.number_input("Serum Cholestoral",step=1.,format="%.0f")


    fbs = 1 if (st.radio("Fasting Blood Sugar > 120 mg/dl", ('True', 'False')))=='True' else 0


    restecg = st.radio("Resting Electrocardiographic Results", ('Normal', 'ST-T Wave Abnormality', 'Probable or Definite Left Ventricular Hypertrophy'))

    if(restecg=='Normal'):
      restecg= 0
    elif(restecg=='ST-T Wave Abnormality'):
      restecg= 1
    else:
      restecg= 2


    thalach = st.number_input("Maximum Heart Rate Achieved",step=1.,format="%.0f")

    exang = 1 if (st.radio("Exercise Induced Angina", ('Yes', 'No')))=='Yes' else 0

    oldpeak = st.number_input("ST depression induced by exercise relative to rest",step=1.,format="%.2f")    

    slope = st.radio("Slope of the peak exercise ST segment", ('Upsloping', 'Flat', 'Downsloping'))

    if(slope=='Upsloping'):
      slope= 0
    elif(slope=='Flat'):
      slope= 1
    else:
      slope= 2


    #ca = st.number_input("Number of major vessels (0-3) colored by flourosopy",step=1.,format="%.2f")
    
    ca = st.radio("Number of major vessels (0-3) colored by flourosopy", (0,1,2,3))

    if(ca==0):
      ca= 0
    elif(ca==1):
      ca= 1
    elif(ca==2):
      ca= 2
    else:
      ca= 3
    

    thal = st.radio("Thal", ('Normal', 'Fixed Defect', 'Reversable Defect'))

    if(thal=='Normal'):
      thal= 3
    elif(thal=='Fixed Defect'):
      thal= 6
    else:
      thal= 7
    

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
