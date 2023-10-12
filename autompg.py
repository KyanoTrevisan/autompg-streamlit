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

# Dataset overview
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

# Visualization Function
def visualize_data(option):
    if option == "Displacement vs. MPG":
        fig, ax = plt.subplots()
        ax.scatter(X['displacement'], y, color='blue')
        ax.set_title('Displacement vs. MPG')
        ax.set_xlabel('Displacement')
        ax.set_ylabel('MPG')
        st.pyplot(fig)
        
    elif option == "Horsepower vs. MPG":
        fig, ax = plt.subplots()
        ax.scatter(X['horsepower'], y, color='red')
        ax.set_title('Horsepower vs. MPG')
        ax.set_xlabel('Horsepower')
        ax.set_ylabel('MPG')
        st.pyplot(fig)

    elif option == "MPG across Different Cylinders":
        fig, ax = plt.subplots()
        data_to_plot = [y[X['cylinders'] == i].values for i in sorted(X['cylinders'].unique())]
        ax.boxplot(data_to_plot)
        ax.set_xticklabels(sorted(X['cylinders'].unique()))
        ax.set_title('MPG across Different Cylinders')
        ax.set_xlabel('Cylinders')
        ax.set_ylabel('MPG')
        st.pyplot(fig)

    elif option == "Weight vs. MPG":
        fig, ax = plt.subplots()
        ax.scatter(X['weight'], y, color='green')
        ax.set_title('Weight vs. MPG')
        ax.set_xlabel('Weight')
        ax.set_ylabel('MPG')
        st.pyplot(fig)

    elif option == "Average MPG by Model Year":
        fig, ax = plt.subplots()
        combined = pd.concat([X, y], axis=1)
        avg_mpg = combined.groupby('model_year')['mpg'].mean()
        ax.bar(avg_mpg.index, avg_mpg.values, color='purple')
        ax.set_title('Average MPG by Model Year')
        ax.set_xlabel('Model Year')
        ax.set_ylabel('Average MPG')
        st.pyplot(fig)

    elif option == "Average MPG by Origin":
        fig, ax = plt.subplots()
        combined = pd.concat([X, y], axis=1)
        avg_mpg = combined.groupby('origin')['mpg'].mean()
        ax.bar(avg_mpg.index, avg_mpg.values, color='orange')
        ax.set_title('Average MPG by Origin')
        ax.set_xlabel('Origin')
        ax.set_ylabel('Average MPG')
        st.pyplot(fig)


# Adding visualization dropdown to Streamlit
visualization_options = ["Displacement vs. MPG", "Horsepower vs. MPG", "MPG across Different Cylinders", 
                         "Weight vs. MPG", "Average MPG by Model Year", "Average MPG by Origin"]
selected_visualization = st.selectbox("Choose a Visualization", visualization_options)
visualize_data(selected_visualization)

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

st.write(f"### Predicted MPG for {model_option}: {prediction}")

# Displaying RMSE for each model
st.write("### Model Performance (RMSE)")
st.write(f"Linear Regression: {lr_rmse}")
st.write(f"SVR: {svr_rmse}")
st.write(f"Ridge Regression: {ridge_rmse}")
