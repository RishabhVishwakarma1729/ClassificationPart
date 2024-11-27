import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.prepimport streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

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

# Function to plot Logistic Regression decision boundary
def plot_logistic_regression_boundary(X_train, y_train, model, feature_columns, scaler=None):
    # Plotting decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.75)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
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

    # Train Logistic Regression model
    if st.button("Train Logistic Regression Model"):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Logistic Regression Test Accuracy: {accuracy * 100:.2f}%")
        
        # Plot the logistic regression decision boundary
        if len(feature_columns) == 2:  # Plot decision boundary for 2 features
            fig = plot_logistic_regression_boundary(X_train, y_train, model, feature_columns)
            st.pyplot(fig)
        else:  # For more than 2 features, use PCA to reduce dimensionality for plotting
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)

            fig = plot_logistic_regression_boundary(X_train_pca, y_train, model, feature_columns)
            st.pyplot(fig)

        # New data prediction
        new_data = st.text_input("Enter new data for prediction (comma separated values for selected features)")

        if new_data:
            try:
                new_data = np.array([list(map(float, new_data.split(',')))]).reshape(1, -1)
                if scaler:
                    new_data_scaled = scaler.transform(new_data)  # Scale the new data if scaling is applied
                else:
                    new_data_scaled = new_data
                new_pred = model.predict(new_data_scaled)
                predicted_class = label_encoder.inverse_transform(new_pred)[0]
                st.write(f"Predicted class: {predicted_class}")

                # Visualizing new data point
                if len(feature_columns) == 2:  # Plot new data point on the decision boundary for 2 features
                    fig_with_new_data = plot_logistic_regression_boundary(X_train, y_train, model, feature_columns)
                    fig_with_new_data.scatter(new_data[0][0], new_data[0][1], color='green', label='New Prediction', s=100, marker='x')
                    fig_with_new_data.legend()
                    st.pyplot(fig_with_new_data)
                else:
                    pca_new_data = pca.transform(new_data)
                    fig_with_new_data = plot_logistic_regression_boundary(X_train_pca, y_train, model, feature_columns)
                    fig_with_new_data.scatter(pca_new_data[0][0], pca_new_data[0][1], color='green', label='New Prediction', s=100, marker='x')
                    fig_with_new_data.legend()
                    st.pyplot(fig_with_new_data)

            except Exception as e:
                st.error(f"Error with new data prediction: {e}")
rocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Function to encode target labels
def encode_labels(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y), label_encoder

# Function to preprocess the data
def preprocess_data(df, target_column, feature_column, scaler_option):
    X = df[[feature_column]]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    
    scaler = StandardScaler() if scaler_option == "Standard Scaler" else None
    if scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    return X_scaled, y_encoded, label_encoder, scaler

# Function to plot Logistic Regression curve
def plot_logistic_regression_curve(X_train, y_train, model, feature_column):
    # Create a range of values for the feature (for smooth curve plotting)
    X_range = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_range)[:, 1]
    
    # Scatter plot
    plt.scatter(X_train, y_train, color='blue', label='Data Points')
    plt.plot(X_range, y_prob, color='red', label='Logistic Regression Curve')
    plt.xlabel(feature_column)
    plt.ylabel('Probability')
    plt.title('Logistic Regression: S-shaped Curve')
    plt.legend()
    
    return plt

# Streamlit app
st.title("Logistic Regression Model")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.write(df.head())

    # Select target and feature columns
    target_column = st.selectbox("Select the target column", df.columns)
    feature_column = st.selectbox("Select the feature column", df.columns[df.columns != target_column])

    scaler_option = st.selectbox("Select Scaler", ["Standard Scaler", "None"])

    # Preprocess the data
    X_scaled, y_encoded, label_encoder, scaler = preprocess_data(df, target_column, feature_column, scaler_option)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    if st.button("Train Logistic Regression Model"):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Logistic Regression Test Accuracy: {accuracy * 100:.2f}%")
        
        # Plot the logistic regression curve
        fig = plot_logistic_regression_curve(X_train, y_train, model, feature_column)
        st.pyplot(fig)

        # New data prediction
        new_data = st.text_input("Enter new data for prediction (single value)")

        if new_data:
            try:
                new_data = np.array([[float(new_data)]])  # Reshape for prediction
                if scaler:
                    new_data_scaled = scaler.transform(new_data)  # Scale the new data if scaling is applied
                else:
                    new_data_scaled = new_data
                new_pred = model.predict(new_data_scaled)
                predicted_class = label_encoder.inverse_transform(new_pred)[0]
                st.write(f"Predicted class: {predicted_class}")

                # Visualizing new data point
                fig_with_new_data = plot_logistic_regression_curve(X_train, y_train, model, feature_column)
                fig_with_new_data.scatter(new_data, new_pred, color='green', label='New Prediction', s=100, marker='x')
                fig_with_new_data.legend()
                st.pyplot(fig_with_new_data)

            except Exception as e:
                st.error(f"Error with new data prediction: {e}")
