import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


#-------Streamlit App------
st.title("Wine Quality Prediction App")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url, sep=';')

st.image('wine quality.jpeg')

st.caption("Wine Quality Dataset Explorer and RandomForest Classifier")

st.subheader("Dataset", divider="violet")

st.write(data.head())

st.subheader("Exploratory Data Analysis(EDA)", divider="violet")


if st.button("Column names"):
    st.write("Dataset Columns", data.columns)
    
    
if st.button("Missing Values"):
    st.write("Sum of missin gvalues", data.isnull().sum())
    

st.subheader("Data Visualization", divider="violet")


if st.checkbox("Bar Chart of Residual Sugar against Quality"):
    st.bar_chart(x="residual sugar" , y="quality" , data=data)
    
    
if st.checkbox("Line Chart of Residual Sugar against Quality"):
    st.line_chart(x="residual sugar" , y="quality" , data=data)
    
    
#PREPARE DATA (drop y because its the dependent variable)
x = data.drop("quality", axis=1)
y = data["quality"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#CREATE AND FIT OUR MODEL
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)


#USER INPUT
st.sidebar.header("Slide your values")

fixed_acidity = st.sidebar.slider("Fixed Acidity", (data['fixed acidity']).min(), (data['fixed acidity']).max(), (data['fixed acidity']).mean())
volatile_acidity = st.sidebar.slider("Volatile Acidity", (data['volatile acidity']).min(), (data['volatile acidity']).max(), (data['volatile acidity']).mean())
citric_acid = st.sidebar.slider("Citric Acid", (data['citric acid']).min(), (data['citric acid']).max(), (data['citric acid']).mean())
residual_sugar = st.sidebar.slider("Residual Sugar", (data['residual sugar']).min(), (data['residual sugar']).max(), (data['residual sugar']).mean())
chlorides = st.sidebar.slider("Chlorides", (data['chlorides']).min(), (data['chlorides']).max(), (data['chlorides']).mean())
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", (data['free sulfur dioxide']).min(), (data['free sulfur dioxide']).max(), (data['free sulfur dioxide']).mean())
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", (data['total sulfur dioxide']).min(), (data['total sulfur dioxide']).max(), (data['total sulfur dioxide']).mean())
density = st.sidebar.slider("Density", (data['density']).min(), (data['density']).max(), (data['density']).mean())
pH = st.sidebar.slider("pH", (data['pH']).min(), (data['pH']).max(), (data['pH']).mean())
sulphates = st.sidebar.slider("Sulphates", (data['sulphates']).min(), (data['sulphates']).max(), (data['sulphates']).mean())
alcohol = st.sidebar.slider("Alcohol", (data['alcohol']).min(), (data['alcohol']).max(), (data['alcohol']).mean())


#PREDICT BUTTON
if st.sidebar.button("Predict"):
    user_input = pd.DataFrame(
        {
           'fixed acidity':[fixed_acidity],
           'volatile acidity':[volatile_acidity],
           'citric acid':[citric_acid],
           'residual sugar':[residual_sugar],
           'chlorides':[chlorides],
           'free sulfur dioxide':[free_sulfur_dioxide],
           'total sulfur dioxide':[total_sulfur_dioxide],
           'density':[density],
           'pH':[pH],
           'sulphates':[sulphates],
           'alcohol':[alcohol]
        }   
    )

#predict the quality of wine
prediction = rf.predict(user_input)

#display the prediction
st.sidebar.subheader('prediction')
st.sidebar.write(f"From the information provided the wine quality is {prediction[0]}")
