import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Function for cleaning the data
def clean_data(data):
    # Filling missing values with mean or mode
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)
    return data

# Function for encoding categorical variables
def encode_labels(data):
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    return data, label_encoders

# Displaying the title and description
st.title("Advanced ML Model Visualizer")
st.write("Uploading a dataset, cleaning it, encoding labels, choosing an ML algorithm, and visualizing predictions.")

# Allowing the user to upload a dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Reading and cleaning the dataset
    data = pd.read_csv(uploaded_file)
    data = clean_data(data)
    data, label_encoders = encode_labels(data)

    # Displaying cleaned dataset
    st.write("Cleaned Dataset preview:")
    st.write(data.head())

    # Allowing the user to select features and target
    st.write("Selecting features and target variable for model training:")
    features = st.multiselect("Selecting features (independent variables)", options=data.columns)
    target = st.selectbox("Selecting target (dependent variable)", options=data.columns)

    if features and target:
        # Preparing the data
        X = data[features]
        y = data[target]

        # Standardizing features for algorithms that need it
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Allowing the user to choose a machine learning algorithm
        st.write("Choosing a machine learning algorithm:")
        algorithm = st.selectbox(
            "Select an algorithm",
            ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"]
        )

        # Training the selected model
        if algorithm == "Logistic Regression":
            model = LogisticRegression()
        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algorithm == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif algorithm == "Support Vector Machine":
            model = SVC(probability=True, kernel="linear")

        # Fitting the model to the training data
        model.fit(X_train, y_train)

        # Calculating and displaying model accuracy
        accuracy = model.score(X_test, y_test)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Allowing the user to input values for a new data point
        st.write("Entering values for the new data point:")
        new_data = []
        for feature in features:
            value = st.number_input(f"Value for {feature}", value=0.0)
            new_data.append(value)

        if st.button("Predict and Visualize"):
            # Making predictions on the new data point
            new_data_scaled = scaler.transform([new_data])
            prediction = model.predict(new_data_scaled)[0]
            prediction_prob = model.predict_proba(new_data_scaled)[0]

            # Displaying the prediction and probabilities
            st.write(f"Prediction: {prediction}")
            st.write(f"Prediction Probabilities: {prediction_prob}")

            # Plotting the results
            if algorithm == "Logistic Regression" or algorithm == "K-Nearest Neighbors":
                # Scatterplot for Logistic Regression and KNN
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="viridis", label="Data Points")
                ax.scatter(new_data_scaled[0][0], new_data_scaled[0][1], c="red", label="New Data Point", marker="X", s=100)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.legend()
                st.pyplot(fig)

            elif algorithm == "Decision Tree":
                # Plotting the decision tree
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(model, feature_names=features, class_names=str(model.classes_), filled=True, ax=ax)
                st.pyplot(fig)

            elif algorithm == "Support Vector Machine":
                # Plotting decision boundaries for SVM
                if len(features) == 2:
                    # 2D decision boundary
                    xx, yy = np.meshgrid(
                        np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                        np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100),
                    )
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots()
                    ax.contourf(xx, yy, Z, alpha=0.8, cmap="viridis")
                    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors="k", cmap="viridis")
                    ax.scatter(new_data_scaled[0][0], new_data_scaled[0][1], c="red", label="New Data Point", marker="X", s=100)
                    ax.set_xlabel(features[0])
                    ax.set_ylabel(features[1])
                    ax.legend()
                    st.pyplot(fig)

                elif len(features) > 2:
                    # High-dimensional rotation visualization (3D)
                    st.write("Visualizing 3D projection (using the first 3 features):")
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, cmap="viridis")
                    ax.scatter(new_data_scaled[0][0], new_data_scaled[0][1], new_data_scaled[0][2], c="red", marker="X", s=100)
                    ax.set_xlabel(features[0])
                    ax.set_ylabel(features[1])
                    ax.set_zlabel(features[2])
                    st.pyplot(fig)
