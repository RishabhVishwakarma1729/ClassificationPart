import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data function
def load_data(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    return None

# Function to encode target labels
def encode_labels(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

# Function to preprocess the data
def preprocess_data(df, target_column, feature_columns):
    X = df[feature_columns]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    X_scaled = StandardScaler().fit_transform(X)  # Scaling the features
    return X_scaled, y_encoded, label_encoder

# Streamlit App
st.title("ML Classification App")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    # Load and display the dataset
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset loaded successfully!")
        st.write(df.head())

        # Select target and feature columns
        target_column = st.selectbox("Select the target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", df.columns[df.columns != target_column])

        if feature_columns:
            # Preprocess the data
            X_scaled, y_encoded, label_encoder = preprocess_data(df, target_column, feature_columns)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            # Train Logistic Regression
            if st.button("Train Logistic Regression Model"):
                model_lr = LogisticRegression(max_iter=1000)
                model_lr.fit(X_train, y_train)
                y_pred_lr = model_lr.predict(X_test)
                accuracy_lr = accuracy_score(y_test, y_pred_lr)
                st.write(f"Logistic Regression Test Accuracy: {accuracy_lr * 100:.2f}%")

            # Train KNN Model
            if st.button("Train KNN Model"):
                k_value = st.slider("Select number of neighbors for KNN", min_value=1, max_value=20, value=5)
                model_knn = KNeighborsClassifier(n_neighbors=k_value)
                model_knn.fit(X_train, y_train)
                y_pred_knn = model_knn.predict(X_test)
                accuracy_knn = accuracy_score(y_test, y_pred_knn)
                st.write(f"KNN Test Accuracy: {accuracy_knn * 100:.2f}%")

            # Train SVM Model
            if st.button("Train SVM Model"):
                kernel_type = st.selectbox("Select SVM Kernel Type", ["linear", "poly", "rbf"])
                model_svm = SVC(kernel=kernel_type)
                model_svm.fit(X_train, y_train)
                y_pred_svm = model_svm.predict(X_test)
                accuracy_svm = accuracy_score(y_test, y_pred_svm)
                st.write(f"SVM Test Accuracy: {accuracy_svm * 100:.2f}%")

            # Train Decision Tree Model
            if st.button("Train Decision Tree Model"):
                model_dt = DecisionTreeClassifier(random_state=42)
                model_dt.fit(X_train, y_train)
                y_pred_dt = model_dt.predict(X_test)
                accuracy_dt = accuracy_score(y_test, y_pred_dt)
                st.write(f"Decision Tree Test Accuracy: {accuracy_dt * 100:.2f}%")

            # New data prediction (column-wise input)
            new_data = {}
            for feature in feature_columns:
                new_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

            if all(value != 0.0 for value in new_data.values()):  # Make sure data is entered for all columns
                new_data_values = np.array(list(new_data.values())).reshape(1, -1)
                new_data_scaled = StandardScaler().fit(X_train).transform(new_data_values)
                
                # Logistic Regression Prediction
                if model_lr:
                    new_pred_lr = model_lr.predict(new_data_scaled)
                    st.write(f"Logistic Regression Predicted class: {label_encoder.inverse_transform(new_pred_lr)[0]}")
                
                # KNN Prediction
                if model_knn:
                    new_pred_knn = model_knn.predict(new_data_scaled)
                    st.write(f"KNN Predicted class: {label_encoder.inverse_transform(new_pred_knn)[0]}")
                
                # SVM Prediction
                if model_svm:
                    new_pred_svm = model_svm.predict(new_data_scaled)
                    st.write(f"SVM Predicted class: {label_encoder.inverse_transform(new_pred_svm)[0]}")
                
                # Decision Tree Prediction
                if model_dt:
                    new_pred_dt = model_dt.predict(new_data_scaled)
                    st.write(f"Decision Tree Predicted class: {label_encoder.inverse_transform(new_pred_dt)[0]}")
