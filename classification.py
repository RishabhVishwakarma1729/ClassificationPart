import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# Cache the file loading to avoid re-reading the file each time the user interacts with the UI
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return None

# Function to save plots
def save_plot(fig, filename):
    plot_path = os.path.join("plots", filename)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig(plot_path, format='jpg')
    st.download_button(label="Download Plot", data=open(plot_path, 'rb'), file_name=filename, mime='image/jpeg')

# Function to encode target labels and return encoding mapping
def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return label_encoder, y_encoded

# Data preprocessing
def preprocess_data(df, target_column, feature_columns, scaler_option):
    X = df[feature_columns]
    y = df[target_column]
    
    # Label Encoding for categorical target
    label_encoder, y_encoded = encode_labels(y)
    
    # Scaling the features
    scaler = StandardScaler() if scaler_option == "Standard Scaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, label_encoder

# Train-Test split function
def split_data(X_scaled, y_encoded):
    return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Plotting KNN Accuracy Curve
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

# 3D SVM Visualization
def plot_svm_3d(X_train, y_train, kernel_type, feature_columns):
    fig_svm_3d = plt.figure()
    ax_svm_3d = fig_svm_3d.add_subplot(111, projection='3d')
    ax_svm_3d.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', s=30)
    ax_svm_3d.set_xlabel(feature_columns[0])
    ax_svm_3d.set_ylabel(feature_columns[1])
    ax_svm_3d.set_zlabel(feature_columns[2])
    ax_svm_3d.set_title(f"SVM Decision Boundary with {kernel_type} kernel")
    return fig_svm_3d

# Main App
st.title("ML Classification App with Custom Dataset")

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

        # Preprocessing data
        scaler_option = st.selectbox(
            "Select Scaler", ["Standard Scaler", "Min-Max Scaler"], 
            help="Standard Scaler works well for most models, Min-Max Scaler is useful for distance-based models like KNN."
        )

        X_scaled, y_encoded, label_encoder = preprocess_data(df, target_column, feature_columns, scaler_option)

        # Train-Test split
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)

        # KNN Model
        if st.button("Train KNN Model"):
            knn_metric = st.selectbox(
                "Select KNN Metric", ["euclidean", "manhattan", "minkowski"],
                help="Euclidean for continuous features, Manhattan for sparse data, Minkowski for general cases."
            )
            k_range = range(1, 21)
            fig_knn, best_k, best_score = plot_knn_accuracy_curve(X_train, y_train, knn_metric, k_range)
            st.pyplot(fig_knn)
            save_plot(fig_knn, f"knn_accuracy_curve_{knn_metric}.jpg")

            st.write(f"Best value of n_neighbors: {best_k} with accuracy: {best_score * 100:.2f}%")
            
            knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=knn_metric)
            knn_model.fit(X_train, y_train)
            y_pred_knn = knn_model.predict(X_test)
            knn_accuracy_test = accuracy_score(y_test, y_pred_knn)
            st.write(f"KNN Test Accuracy: {knn_accuracy_test * 100:.2f}%")

        # SVM Model
        if st.button("Train SVM Model"):
            kernel_type = st.selectbox(
                "Select SVM Kernel Type", ["linear", "poly", "rbf"],
                help="Linear is best for linearly separable data, Poly for non-linear, RBF for complex decision boundaries."
            )

            svm_model = SVC(kernel=kernel_type)
            svm_model.fit(X_train, y_train)
            y_pred_svm = svm_model.predict(X_test)
            svm_accuracy = accuracy_score(y_test, y_pred_svm)
            st.write(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")

            if X_train.shape[1] >= 3:
                fig_svm_3d = plot_svm_3d(X_train, y_train, kernel_type, feature_columns)
                st.pyplot(fig_svm_3d)
                save_plot(fig_svm_3d, f"svm_decision_boundary_{kernel_type}_3d.jpg")
            else:
                st.write("Data doesn't have enough features (at least 3) for 3D visualization.")

        # Decision Tree Model
        if st.button("Train Decision Tree Model"):
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_train, y_train)
            y_pred_dt = dt_model.predict(X_test)
            dt_accuracy = accuracy_score(y_test, y_pred_dt)
            st.write(f"Decision Tree Test Accuracy: {dt_accuracy * 100:.2f}%")

            fig_dt, ax_dt = plt.subplots(figsize=(12, 8))
            from sklearn.tree import plot_tree
            plot_tree(dt_model, feature_names=feature_columns, class_names=label_encoder.classes_, filled=True, ax=ax_dt)
            plt.title("Decision Tree Visualization")
            st.pyplot(fig_dt)
            save_plot(fig_dt, "decision_tree_visualization.jpg")
