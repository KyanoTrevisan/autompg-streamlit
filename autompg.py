# Import necessary libraries for the Streamlit app
import streamlit as st
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Fetch the dataset
auto_mpg = fetch_ucirepo(id=9) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# App title and intro
st.title("Auto MPG Prediction App")
st.write("Predict Miles per Gallon (MPG) based on car features.")

# Displaying metadata and dataset overview
st.write("### Dataset Metadata")
st.write(auto_mpg.metadata)

st.write("### Dataset Variables")
st.write(auto_mpg.variables)

# Filling missing values
X['horsepower'].fillna(X['horsepower'].mean(), inplace=True)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

# SVR
svr_model = SVR()
svr_model.fit(X_train, y_train.values.ravel())
svr_predictions = svr_model.predict(X_test)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))

# Visualizing the distribution of MPG
st.write("### Distribution of MPG")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y, bins=20, color='skyblue', edgecolor='black')
ax.set_title('Distribution of MPG')
ax.set_xlabel('MPG')
ax.set_ylabel('Number of Cars')
ax.grid(True)
st.pyplot(fig)

# User input for prediction
st.write("### Predict MPG")
model_option = st.selectbox("Select a Model", ["Linear Regression", "SVR", "Ridge Regression"])

# Getting input features from the user
input_features = {}
for feature in X.columns:
    input_features[feature] = st.slider(f"Select value for {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

# Predicting using the chosen model
if model_option == "Linear Regression":
    prediction = lr_model.predict([list(input_features.values())])[0]
elif model_option == "SVR":
    prediction = svr_model.predict([list(input_features.values())])[0]
else:
    prediction = ridge_model.predict([list(input_features.values())])[0]

st.write(f"Predicted MPG for {model_option}: {prediction}")

# Displaying RMSE for each model
st.write("### Model Performance (RMSE)")
st.write(f"Linear Regression: {lr_rmse}")
st.write(f"SVR: {svr_rmse}")
st.write(f"Ridge Regression: {ridge_rmse}")
