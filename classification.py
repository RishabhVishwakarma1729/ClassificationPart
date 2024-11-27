import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initializing session state for models and data
if "models" not in st.session_state:
    st.session_state.models = {}
if "X_train" not in st.session_state:
    st.session_state.X_train, st.session_state.X_test = None, None
    st.session_state.y_train, st.session_state.y_test = None, None
    st.session_state.label_encoder = None

# Loading the dataset
def load_data(file):
    # Loading CSV or Excel files
    if file:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    return None

# Preprocessing the dataset
def preprocess_data(df, target_column, feature_columns):
    # Splitting into features and target
    X = df[feature_columns]
    y = df[target_column]
    # Encoding target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Scaling feature values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y_encoded, label_encoder

# Streamlit App
st.title("Simplified ML Classification App")

# Uploading dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    # Loading and displaying dataset
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataset loaded successfully!")
        st.write(df.head())

        # Selecting target and feature columns
        target_column = st.selectbox("Select the target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", df.columns[df.columns != target_column])

        if feature_columns:
            # Preprocessing the data
            X, y, label_encoder = preprocess_data(df, target_column, feature_columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Storing preprocessed data in session state
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            st.session_state.label_encoder = label_encoder

            # Training models
            if st.button("Train All Models"):
                # Training Logistic Regression
                model_lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                st.session_state.models["Logistic Regression"] = model_lr

                # Training KNN
                model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
                st.session_state.models["KNN"] = model_knn

                # Training SVM
                model_svm = SVC(kernel="linear").fit(X_train, y_train)
                st.session_state.models["SVM"] = model_svm

                # Training Decision Tree
                model_dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
                st.session_state.models["Decision Tree"] = model_dt

                st.success("All models trained successfully!")

            # Predicting on new data
            new_data = {feature: st.number_input(f"Enter value for {feature}", value=0.0) for feature in feature_columns}

            if st.button("Predict"):
                if all(value != 0.0 for value in new_data.values()):
                    # Scaling new input data
                    new_data_values = pd.DataFrame([new_data])
                    new_data_scaled = StandardScaler().fit(X_train).transform(new_data_values)

                    # Making predictions with each trained model
                    for model_name, model in st.session_state.models.items():
                        if model:
                            pred = model.predict(new_data_scaled)
                            predicted_class = st.session_state.label_encoder.inverse_transform(pred)[0]
                            st.write(f"{model_name} Prediction: {predicted_class}")
                        else:
                            st.warning(f"{model_name} is not trained yet!")
                else:
                    st.warning("Please fill all feature values for prediction.")
