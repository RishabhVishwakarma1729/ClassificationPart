import streamlit as st
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

# Streamlit App
st.title("ML Classification App")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

# Initialize model variables
models = {
    "Logistic Regression": None,
    "KNN": None,
    "SVM": None,
    "Decision Tree": None
}
model_fitted = {key: False for key in models}

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

            # Train Models
            for model_name in models.keys():
                if st.button(f"Train {model_name} Model"):
                    if model_name == "Logistic Regression":
                        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
                        models["Logistic Regression"].fit(X_train, y_train)
                    elif model_name == "KNN":
                        models["KNN"] = KNeighborsClassifier()
                        models["KNN"].fit(X_train, y_train)
                    elif model_name == "SVM":
                        models["SVM"] = SVC()
                        models["SVM"].fit(X_train, y_train)
                    elif model_name == "Decision Tree":
                        models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
                        models["Decision Tree"].fit(X_train, y_train)

                    # Mark model as fitted
                    model_fitted[model_name] = True

                    # Evaluate model
                    y_pred = models[model_name].predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")

            # New data prediction
            new_data = {}
            for feature in feature_columns:
                new_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

            # Predict with all models when the button is clicked
            if st.button("Predict"):
                if all(value != 0.0 for value in new_data.values()):  # Ensure all columns are filled
                    new_data_values = pd.DataFrame([new_data])
                    new_data_scaled = StandardScaler().fit(X_train).transform(new_data_values)

                    for model_name, model in models.items():
                        if model_fitted[model_name]:  # Only predict with trained models
                            new_pred = model.predict(new_data_scaled)
                            st.write(f"{model_name} Predicted class: {label_encoder.inverse_transform(new_pred)[0]}")
                        else:
                            st.warning(f"{model_name} is not trained yet.")
                else:
                    st.warning("Please enter valid values for all features before predicting.")
