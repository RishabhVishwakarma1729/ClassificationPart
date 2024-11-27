import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import os

# Cache file loading to avoid re-reading
@st.cache_data
def load_data(file):
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
def preprocess_data(df, target_column, feature_columns, scaler_option):
    X = df[feature_columns]
    y = df[target_column]
    y_encoded, label_encoder = encode_labels(y)
    
    scaler = StandardScaler() if scaler_option == "Standard Scaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, label_encoder

# Train-test split
def split_data(X_scaled, y_encoded):
    return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Function to plot KNN accuracy curve
def plot_knn_accuracy_curve(X_train, y_train, metric, k_range):
    knn_accuracy = []
    best_k = 1
    best_score = 0

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        knn_accuracy.append(scores.mean())
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_k = k

    fig, ax = plt.subplots()
    ax.plot(k_range, knn_accuracy, marker='o')
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title(f"KNN Accuracy for Different k Values ({metric} Metric)")
    return fig, best_k, best_score

# Function for 3D SVM Visualization
def plot_svm_3d(X_train, y_train, kernel_type, feature_columns):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', s=30)
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_zlabel(feature_columns[2])
    ax.set_title(f"SVM Decision Boundary with {kernel_type} kernel")
    return fig

# Save plot to file
def save_plot(fig, filename):
    plot_path = os.path.join("plots", filename)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig(plot_path, format='jpg')
    return plot_path

# Main Streamlit app
st.title("Classification and Visualisation App")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is None:
        st.error("Unsupported file format! Please upload a CSV or Excel file.")
    else:
        st.write("Dataset loaded successfully!")
        st.write(df.head())

        target_column = st.selectbox("Select the target column", df.columns)
        feature_columns = st.multiselect("Select the feature columns", df.columns[df.columns != target_column])

        if not feature_columns:
            st.error("Please select at least one feature column.")

        scaler_option = st.selectbox("Select Scaler", ["Standard Scaler", "Min-Max Scaler"])

        X_scaled, y_encoded, label_encoder = preprocess_data(df, target_column, feature_columns, scaler_option)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)

        # KNN Model
        if st.button("Train KNN Model"):
            knn_metric = st.selectbox("Select KNN Metric", ["euclidean", "manhattan", "minkowski"])
            k_range = range(1, 21)
            fig_knn, best_k, best_score = plot_knn_accuracy_curve(X_train, y_train, knn_metric, k_range)
            st.pyplot(fig_knn)
            plot_path = save_plot(fig_knn, f"knn_accuracy_curve_{knn_metric}.jpg")
            st.download_button(label="Download Plot", data=open(plot_path, 'rb'), file_name=f"knn_accuracy_curve_{knn_metric}.jpg", mime='image/jpeg')

            st.write(f"Best K: {best_k} with Accuracy: {best_score * 100:.2f}%")
            
            knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=knn_metric)
            knn_model.fit(X_train, y_train)
            y_pred_knn = knn_model.predict(X_test)
            knn_accuracy_test = accuracy_score(y_test, y_pred_knn)
            st.write(f"KNN Test Accuracy: {knn_accuracy_test * 100:.2f}%")

            new_data = st.text_input("Enter new data for KNN prediction (comma separated values)")
            if new_data:
                new_data = np.array([list(map(float, new_data.split(',')))]).reshape(1, -1)
                new_pred_knn = knn_model.predict(new_data)
                st.write(f"Predicted class: {label_encoder.inverse_transform(new_pred_knn)[0]}")

        # SVM Model
        if st.button("Train SVM Model"):
            kernel_type = st.selectbox("Select SVM Kernel Type", ["linear", "poly", "rbf"])

            svm_model = SVC(kernel=kernel_type)
            svm_model.fit(X_train, y_train)
            y_pred_svm = svm_model.predict(X_test)
            svm_accuracy = accuracy_score(y_test, y_pred_svm)
            st.write(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")

            if X_train.shape[1] >= 3:
                fig_svm_3d = plot_svm_3d(X_train, y_train, kernel_type, feature_columns)
                st.pyplot(fig_svm_3d)
                plot_path = save_plot(fig_svm_3d, f"svm_decision_boundary_{kernel_type}_3d.jpg")
                st.download_button(label="Download 3D SVM Plot", data=open(plot_path, 'rb'), file_name=f"svm_decision_boundary_{kernel_type}_3d.jpg", mime='image/jpeg')

                new_data_svm = st.text_input("Enter new data for SVM prediction (comma separated values)")
                if new_data_svm:
                    new_data_svm = np.array([list(map(float, new_data_svm.split(',')))]).reshape(1, -1)
                    new_pred_svm = svm_model.predict(new_data_svm)
                    st.write(f"Predicted class: {label_encoder.inverse_transform(new_pred_svm)[0]}")

        # Decision Tree Model
        if st.button("Train Decision Tree Model"):
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_train, y_train)
            y_pred_dt = dt_model.predict(X_test)
            dt_accuracy = accuracy_score(y_test, y_pred_dt)
            st.write(f"Decision Tree Test Accuracy: {dt_accuracy * 100:.2f}%")

            fig_dt, ax_dt = plt.subplots(figsize=(15, 10))
            plot_tree(dt_model, filled=True, feature_names=feature_columns, class_names=label_encoder.classes_, ax=ax_dt)
            plt.title("Decision Tree Classifier")
            st.pyplot(fig_dt)

            new_data_dt = st.text_input("Enter new data for Decision Tree prediction (comma separated values)")
            if new_data_dt:
                new_data_dt = np.array([list(map(float, new_data_dt.split(',')))]).reshape(1, -1)
                new_pred_dt = dt_model.predict(new_data_dt)
                st.write(f"Predicted class: {label_encoder.inverse_transform(new_pred_dt)[0]}")
