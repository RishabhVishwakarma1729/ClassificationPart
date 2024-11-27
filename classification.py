import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Function to encode target labels
def encode_labels(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

# Function to preprocess the data
def preprocess_data(df, target_column, feature_columns, scaler_option):
    X = df[feature_columns]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    
    scaler = StandardScaler() if scaler_option == "Standard Scaler" else None
    if scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    return X_scaled, y_encoded, label_encoder, scaler

# Function to plot Logistic Regression decision boundary using np.linspace
def plot_logistic_regression(X_train, y_train, model, feature_columns, scaler=None):
    # Generate a range of values using np.linspace for the decision boundary
    x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
    x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    x1_values = np.linspace(x1_min, x1_max, 100)
    x2_values = np.linspace(x2_min, x2_max, 100)
    
    # Create the decision boundary by predicting for each combination of x1 and x2 values
    xx, yy = np.meshgrid(x1_values, x2_values)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid_points).reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.RdBu)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k', marker='o')
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.title("Logistic Regression Decision Boundary")
    return plt

# Streamlit app
st.title("Logistic Regression with Multiple Features")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.write(df.head())

    # Select target and feature columns
    target_column = st.selectbox("Select the target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", df.columns[df.columns != target_column])

    scaler_option = st.selectbox("Select Scaler", ["Standard Scaler", "None"])

    # Preprocess the data
    X_scaled, y_encoded, label_encoder, scaler = preprocess_data(df, target_column, feature_columns, scaler_option)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Plotting the decision boundary
    fig = plot_logistic_regression(X_train, y_train, model, feature_columns, scaler)
    st.pyplot(fig)
    
    # Show model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # New data prediction input
    new_data = st.text_input("Enter new data point for prediction (comma-separated values)")
    if new_data:
        new_data = np.array([list(map(float, new_data.split(',')))])
        if scaler:
            new_data = scaler.transform(new_data)
        new_pred = model.predict(new_data)
        st.write(f"Predicted class: {label_encoder.inverse_transform(new_pred)[0]}")
