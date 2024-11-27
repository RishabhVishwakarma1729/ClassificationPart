import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load data
def load_data(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    return None

# Function to clean data
def clean_data(df):
    # Remove rows with missing values
    df_cleaned = df.dropna()
    
    # Optionally: Convert columns to appropriate types (e.g., numeric)
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
        except:
            pass
    
    # You can add other cleaning steps as needed (e.g., encoding categorical variables, etc.)
    return df_cleaned

# Function to encode target labels
def encode_labels(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

# Preprocess data
def preprocess_data(df, target_column, feature_columns):
    X = df[feature_columns]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y_encoded, label_encoder

# Streamlit App
st.title("ML Classification App")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    # Load and clean data
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset loaded successfully!")

        # Data Cleaning
        df_cleaned = clean_data(df)
        st.write("Cleaned Data:")
        st.write(df_cleaned.head())  # Show cleaned data for review

        # Select target and feature columns
        target_column = st.selectbox("Select the target column", df_cleaned.columns)
        feature_columns = st.multiselect("Select feature columns", df_cleaned.columns[df_cleaned.columns != target_column])

        if feature_columns:
            # Preprocess data
            X_scaled, y_encoded, label_encoder = preprocess_data(df_cleaned, target_column, feature_columns)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            # Train Logistic Regression
            if st.button("Train Logistic Regression Model"):
                model_lr = LogisticRegression(max_iter=1000)
                model_lr.fit(X_train, y_train)
                y_pred_lr = model_lr.predict(X_test)
                accuracy_lr = accuracy_score(y_test, y_pred_lr)
                st.write(f"Logistic Regression Test Accuracy: {accuracy_lr * 100:.2f}%")

            # New data prediction (column-wise input)
            if st.button("Predict"):
                new_data = {}
                for feature in feature_columns:
                    new_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                if all(value != 0.0 for value in new_data.values()):  # Ensure all columns are filled
                    new_data_values = np.array(list(new_data.values())).reshape(1, -1)
                    new_data_scaled = StandardScaler().fit(X_train).transform(new_data_values)

                    if model_lr is not None:
                        new_pred_lr = model_lr.predict(new_data_scaled)
                        st.write(f"Predicted class: {label_encoder.inverse_transform(new_pred_lr)[0]}")
                else:
                    st.warning("Please enter valid values for all features before predicting.")
