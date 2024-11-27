import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initializing session state
if "models" not in st.session_state:
    st.session_state.models = {}
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "accuracies" not in st.session_state:
    st.session_state.accuracies = {}

# Function to load data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

# Function to preprocess data
def preprocess_data(df, target_column, feature_columns):
    X = df[feature_columns]
    y = df[target_column]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded, scaler, label_encoder

# Streamlit app setup
st.title("Machine Learning Classification App")

# Uploading dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset Preview:")
        st.write(df.head())

        # Selecting target and feature columns
        target_column = st.selectbox("Select target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])

        if feature_columns:
            X, y, scaler, label_encoder = preprocess_data(df, target_column, feature_columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Storing data in session state
            st.session_state.scaler = scaler
            st.session_state.label_encoder = label_encoder
            st.session_state.feature_columns = feature_columns
            st.success("Data preprocessed successfully!")

            # Training models with default parameters (no user inputs)
            st.write("### Train Models")
            
            # Logistic Regression
            if st.button("Train Logistic Regression"):
                model_lr = LogisticRegression(max_iter=1000)
                model_lr.fit(X_train, y_train)
                y_pred = model_lr.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
                st.session_state.models["Logistic Regression"] = model_lr
                st.session_state.accuracies["Logistic Regression"] = accuracy * 100

            # KNN
            if st.button("Train KNN"):
                model_knn = KNeighborsClassifier()  # Using default K=5
                model_knn.fit(X_train, y_train)
                y_pred = model_knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"KNN Accuracy: {accuracy * 100:.2f}%")
                st.session_state.models["KNN"] = model_knn
                st.session_state.accuracies["KNN"] = accuracy * 100

            # SVM
            if st.button("Train SVM"):
                model_svm = SVC()  # Using default 'rbf' kernel
                model_svm.fit(X_train, y_train)
                y_pred = model_svm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"SVM Accuracy: {accuracy * 100:.2f}%")
                st.session_state.models["SVM"] = model_svm
                st.session_state.accuracies["SVM"] = accuracy * 100

            # Decision Tree
            if st.button("Train Decision Tree"):
                model_dt = DecisionTreeClassifier(random_state=42)  # Using default parameters
                model_dt.fit(X_train, y_train)
                y_pred = model_dt.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
                st.session_state.models["Decision Tree"] = model_dt
                st.session_state.accuracies["Decision Tree"] = accuracy * 100

            # Display all accuracy scores
            if st.session_state.accuracies:
                st.write("### Model Accuracies")
                for model_name, accuracy in st.session_state.accuracies.items():
                    st.write(f"{model_name}: {accuracy:.2f}%")

            # Predicting new data
            st.write("### Predict with New Data")
            new_data = {feature: st.number_input(f"Enter value for {feature}", value=0.0) for feature in feature_columns}

            if st.button("Predict"):
                if not st.session_state.models:
                    st.warning("Please train at least one model before predicting.")
                else:
                    # Validating input dimensions
                    if len(new_data) != len(feature_columns):
                        st.error(f"Number of features must match the trained model: {len(feature_columns)}.")
                    else:
                        # Scaling input data
                        new_data_scaled = st.session_state.scaler.transform([list(new_data.values())])

                        # Making predictions
                        for model_name, model in st.session_state.models.items():
                            pred = model.predict(new_data_scaled)
                            predicted_class = st.session_state.label_encoder.inverse_transform(pred)[0]
                            st.write(f"{model_name} Prediction: {predicted_class}")
