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
    """Load the dataset based on file type (CSV or Excel)."""
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    return None

# Function to encode target labels
def encode_labels(y):
    """Encode target labels using LabelEncoder."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

# Function to preprocess the data
def preprocess_data(df, target_column, feature_columns):
    """Preprocess data by scaling features and encoding labels."""
    X = df[feature_columns]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    X_scaled = StandardScaler().fit_transform(X)  # Scaling the features
    return X_scaled, y_encoded, label_encoder

# Function to train model
def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and display accuracy results."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")
    return model  # Return the trained model

# Streamlit App
st.title("ML Classification App")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

# Initialize models and state for session
if 'models' not in st.session_state:
    st.session_state.models = {
        "Logistic Regression": None,
        "KNN": None,
        "SVM": None,
        "Decision Tree": None
    }

if 'trained' not in st.session_state:
    st.session_state.trained = False

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

            # Model Training and Accuracy Display
            model_mapping = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }

            for model_name, model in model_mapping.items():
                if st.button(f"Train {model_name} Model"):
                    if model_name == "KNN":
                        k_value = st.slider("Select number of neighbors for KNN", min_value=1, max_value=20, value=5)
                        model.set_params(n_neighbors=k_value)
                    if model_name == "SVM":
                        kernel_type = st.selectbox("Select SVM Kernel Type", ["linear", "poly", "rbf"])
                        model.set_params(kernel=kernel_type)

                    # Train and evaluate model
                    trained_model = train_model(model, X_train, y_train, X_test, y_test, model_name)
                    st.session_state.models[model_name] = trained_model  # Save model in session state
                    st.session_state.trained = True  # Mark models as trained

            # New data prediction (column-wise input)
            if st.session_state.trained:
                new_data = {}
                for feature in feature_columns:
                    new_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                # Predict with all models when the button is clicked
                if st.button("Predict"):
                    if all(value != 0.0 for value in new_data.values()):  # Ensure all columns are filled
                        new_data_values = np.array(list(new_data.values())).reshape(1, -1)
                        new_data_scaled = StandardScaler().fit(X_train).transform(new_data_values)

                        # Prediction with all models, only if they have been trained
                        for model_name, model in st.session_state.models.items():
                            if model is not None:
                                new_pred = model.predict(new_data_scaled)
                                st.write(f"{model_name} Predicted class: {label_encoder.inverse_transform(new_pred)[0]}")
                    else:
                        st.warning("Please enter valid values for all features before predicting.")
