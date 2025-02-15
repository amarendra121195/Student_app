import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with  open("student_lr_final_model.pkl",'rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocesssing_input_data(data, scaler,le):
    # transform the categorical data from zeorth index
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    # make the dataframe 
    df = pd.DataFrame([data])
    # transform all data
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocesssing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("student performnce perdiction")
    st.write("enter your data to get a prediction for your performance")
    
    #set min value max and default
    hour_sutdied = st.number_input("Hours studied",min_value = 1, max_value = 10 , value = 5)
    prvious_score = st.number_input("previous score",min_value = 40, max_value = 100 , value = 70)
    extra = st.selectbox("extra curri activity" , ['Yes',"No"])
    sleeping_hour = st.number_input("sleeping hours",min_value = 4, max_value = 10 , value = 7)
    number_of_peper_solved = st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    
    if st.button("predict-your_score"):
        user_data = {
            "Hours Studied":hour_sutdied,
            "Previous Scores":prvious_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_peper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"your prediciotn result: {prediction}")

#python execution block   
if __name__ == "__main__":
    main()
    

