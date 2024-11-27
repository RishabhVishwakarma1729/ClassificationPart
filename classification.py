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
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D

# Function to save visualizations
def save_plot(fig, filename):
    plot_path = os.path.join("plots", filename)
    if not os.path.exists("plots"):
        os.makedirs("plots")  # Creating 'plots' directory if not existing
    fig.savefig(plot_path, format='jpg')  # Saving plot as JPG
    st.download_button(label="Download Plot", data=open(plot_path, 'rb'), file_name=filename, mime='image/jpeg')  # Allowing the user to download the plot

# File upload functionality
st.title("ML Classification App with Custom Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Loading dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)  # Reading CSV file
    else:
        df = pd.read_excel(uploaded_file)  # Reading Excel file
    
    st.write("Dataset loaded successfully!")
    st.write("Preview of the dataset:")
    st.write(df.head())  # Displaying dataset preview

    # Selecting target and features
    target_column = st.selectbox("Select the target column", df.columns)
    feature_columns = st.multiselect("Select the feature columns", df.columns[df.columns != target_column])

    if not feature_columns:
        st.error("Please select at least one feature column.")  # Displaying error if no feature columns are selected

    # Preparing data
    X = df[feature_columns]  # Extracting feature columns
    y = df[target_column]  # Extracting target column

    # Label Encoding for categorical target
    label_encoder = LabelEncoder()  # Initializing LabelEncoder
    y_encoded = label_encoder.fit_transform(y)  # Encoding target values
    st.write("Label Encoding done. Below are the classes and their respective encoded values:")
    st.write(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))  # Displaying encoded target classes

    # Selecting Scaler
    scaler_option = st.selectbox(
        "Select Scaler", 
        ["Standard Scaler", "Min-Max Scaler"], 
        help="Standard Scaler works well with algorithms that assume Gaussian distributions. Min-Max Scaler is useful for algorithms that rely on distance measures (e.g., KNN)."
    )

    # Scaling the data based on user selection
    if scaler_option == "Standard Scaler":
        scaler = StandardScaler()  # Initializing StandardScaler
    elif scaler_option == "Min-Max Scaler":
        scaler = MinMaxScaler()  # Initializing Min-Max Scaler

    X_scaled = scaler.fit_transform(X)  # Scaling feature data
    
    # Train-test split only when needed (based on selected algorithm)
    def split_data():
        return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)  # Splitting data into training and testing sets

    # KNN Model
    knn_model = None
    if st.button("Train KNN Model"):
        X_train, X_test, y_train, y_test = split_data()  # Splitting data for KNN

        knn_metric = st.selectbox(
            "Select KNN Metric", 
            ["euclidean", "manhattan", "minkowski"], 
            help="Euclidean is ideal for continuous features, Manhattan works better for sparse data, Minkowski is a generalization of both."
        )

        # Finding the best number of neighbors
        st.write("Finding the best value for n_neighbors...")
        k_range = list(range(1, 21))  # Creating a range for k values
        best_score = 0
        best_k = 1
        knn_accuracy = []

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)  # Initializing KNN classifier
            scores = cross_val_score(knn, X_train, y_train, cv=5)  # Performing cross-validation to find the best k
            knn_accuracy.append(scores.mean())  # Storing cross-validation scores
            if scores.mean() > best_score:
                best_score = scores.mean()  # Updating best score
                best_k = k  # Updating best k

        # Plotting KNN accuracy curve
        fig, ax = plt.subplots()
        ax.plot(k_range, knn_accuracy, marker='o')
        ax.set_xlabel("Number of Neighbors (k)")  # Labeling x-axis
        ax.set_ylabel("Cross-Validated Accuracy")  # Labeling y-axis
        ax.set_title(f"KNN Accuracy for Different k Values ({knn_metric} Metric)")  # Adding title
        st.pyplot(fig)  # Displaying plot
        save_plot(fig, f"knn_accuracy_curve_{knn_metric}.jpg")  # Saving plot

        st.write(f"Best value of n_neighbors: {best_k} with accuracy: {best_score * 100:.2f}%")  # Displaying best k value and accuracy

        # Training KNN with best k
        knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=knn_metric)
        knn_model.fit(X_train, y_train)  # Fitting KNN model to the data
        y_pred_knn = knn_model.predict(X_test)  # Making predictions on test data
        knn_accuracy_test = accuracy_score(y_test, y_pred_knn)  # Calculating accuracy
        st.write(f"KNN Test Accuracy: {knn_accuracy_test * 100:.2f}%")  # Displaying accuracy

        # Option for prediction
        new_data = []
        for feature in feature_columns:
            value = st.number_input(f"Enter value for {feature}")  # Collecting user input for prediction
            new_data.append(value)

        new_data_scaled = scaler.transform([new_data])  # Scaling new data
        prediction = knn_model.predict(new_data_scaled)  # Making prediction
        prediction_label = label_encoder.inverse_transform(prediction)  # Converting encoded prediction back to original label
        st.write(f"Predicted class for the entered data point: {prediction_label[0]}")  # Displaying prediction

    # SVM Model
    svm_model = None
    if st.button("Train SVM Model"):
        X_train, X_test, y_train, y_test = split_data()  # Splitting data for SVM

        kernel_type = st.selectbox(
            "Select SVM Kernel Type", 
            ["linear", "poly", "rbf"], 
            help="Linear is ideal for linearly separable data, Poly for non-linear, and RBF for complex decision boundaries."
        )

        svm_model = SVC(kernel=kernel_type)  # Initializing SVM model
        svm_model.fit(X_train, y_train)  # Fitting SVM model to the data
        y_pred_svm = svm_model.predict(X_test)  # Making predictions on test data
        svm_accuracy = accuracy_score(y_test, y_pred_svm)  # Calculating accuracy
        st.write(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")  # Displaying accuracy

        # Plotting 3D SVM Decision Boundary
        if X_train.shape[1] >= 3:
            st.write("3D SVM decision boundary visualization:")
            fig_svm_3d = plt.figure()
            ax_svm_3d = fig_svm_3d.add_subplot(111, projection='3d')
            
            ax_svm_3d.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', s=30)  # Plotting 3D scatter
            ax_svm_3d.set_xlabel(feature_columns[0])  # Labeling x-axis
            ax_svm_3d.set_ylabel(feature_columns[1])  # Labeling y-axis
            ax_svm_3d.set_zlabel(feature_columns[2])  # Labeling z-axis
            ax_svm_3d.set_title(f"SVM Decision Boundary with {kernel_type} kernel")  # Adding title
            
            st.pyplot(fig_svm_3d)  # Displaying plot
            save_plot(fig_svm_3d, f"svm_decision_boundary_{kernel_type}_3d.jpg")  # Saving plot
        else:
            st.write("Data doesn't have enough features (at least 3) for 3D visualization.")  # Error message if there are not enough features

        # Option for prediction
        new_data = []
        for feature in feature_columns:
            value = st.number_input(f"Enter value for {feature}")  # Collecting user input for prediction
            new_data.append(value)

        new_data_scaled = scaler.transform([new_data])  # Scaling new data
        prediction = svm_model.predict(new_data_scaled)  # Making prediction
        prediction_label = label_encoder.inverse_transform(prediction)  # Converting encoded prediction back to original label
       
