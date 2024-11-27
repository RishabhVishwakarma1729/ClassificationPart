import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

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
    encoded_info = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
            # Storing original and encoded mappings
            encoded_info[col] = dict(zip(le.classes_, range(len(le.classes_))))
    return data, label_encoders, encoded_info

# Function to download images
def download_image(image, filename):
    # Save the plot to a file
    image_path = f"/tmp/{filename}"
    image.savefig(image_path, format='jpg')
    # Provide the option to download the file
    with open(image_path, "rb") as file:
        st.download_button(label="Download Image", data=file, file_name=filename)

# Displaying the title and description
st.title("Advanced ML Model Visualizer")
st.write("Upload a dataset, clean it, encode labels, train ML models, and visualize predictions.")

# Allowing the user to upload a dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Reading and cleaning the dataset
    data = pd.read_csv(uploaded_file)
    data = clean_data(data)
    data, label_encoders, encoded_info = encode_labels(data)

    # Displaying cleaned dataset
    st.write("Cleaned Dataset preview:")
    st.write(data.head())

    # Displaying encoded target labels
    st.write("Encoded Labels Information:")
    for col, mapping in encoded_info.items():
        st.write(f"Column: **{col}**")
        st.write(mapping)

    # Allowing the user to select features and target
    st.write("Select features and target variable for model training:")
    features = st.multiselect("Select features (independent variables)", options=data.columns)
    target = st.selectbox("Select target (dependent variable)", options=data.columns)

    if features and target:
        # Preparing the data
        X = data[features]
        y = data[target]

        # Allowing the user to choose a scaling method
        scaler_choice = st.selectbox(
            "Select a scaling method:",
            ["Standard Scaler", "MinMax Scaler"]
        )

        # Display information about scalers
        st.write("Scaler Information:")
        st.write("""
            - **Standard Scaler**: Best suited for algorithms like Logistic Regression, SVM, and Decision Trees, which assume data has a normal distribution and/or benefit from zero mean and unit variance.
            - **MinMax Scaler**: Best suited for algorithms like KNN, which are sensitive to the range of features, and ensures the feature values are scaled between 0 and 1.
        """)

        # Applying the selected scaler
        if scaler_choice == "Standard Scaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Allowing the user to choose a machine learning algorithm
        st.write("Choose a machine learning algorithm:")
        algorithm = st.selectbox(
            "Select an algorithm",
            ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"]
        )

        # Training the selected model
        if algorithm == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Plotting the logistic regression decision boundary (S-curve)
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
            Z = model.predict_proba(np.c_[xx, yy])[:, 1]
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, 25, cmap="RdBu", alpha=0.8)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", cmap="RdBu", s=30)
            ax.set_title("Logistic Regression S-Curve Decision Boundary")
            st.pyplot(fig)

            # Allowing the user to download the logistic regression plot
            download_image(fig, "logistic_regression_decision_boundary.jpg")

        elif algorithm == "K-Nearest Neighbors":
            # Finding the best value for n_neighbors using cross-validation
            st.write("Finding the best value for n_neighbors...")
            k_range = list(range(1, 21))
            best_score = 0
            best_k = 1

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=5)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_k = k

            st.write(f"Best value of n_neighbors: {best_k} with accuracy: {best_score * 100:.2f}%")

            # Train the model with the best n_neighbors
            model = KNeighborsClassifier(n_neighbors=best_k)
            model.fit(X_train, y_train)

            # Visualizing KNN
            fig, ax = plt.subplots(figsize=(10, 6))
            X_set, y_set = X_train, y_train
            X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min(), X_set[:, 0].max(), 0.01),
                                 np.arange(X_set[:, 1].min(), X_set[:, 1].max(), 0.01))
            Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
            ax.contourf(X1, X2, Z, alpha=0.8, cmap="RdBu")
            ax.scatter(X_set[:, 0], X_set[:, 1], c=y_set, edgecolors="k", cmap="RdBu", s=30)
            ax.set_title(f"KNN Decision Boundary (n_neighbors={best_k})")
            st.pyplot(fig)

            # Allowing the user to download the KNN plot
            download_image(fig, f"knn_decision_boundary_{best_k}.jpg")

        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            # Plotting the decision tree
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(model, feature_names=features, class_names=str(model.classes_), filled=True, ax=ax)
            st.pyplot(fig)

            # Allowing the user to download the decision tree plot
            download_image(fig, "decision_tree_plot.jpg")

        elif algorithm == "Support Vector Machine":
            # User selects kernel type for SVM
            kernel_type = st.selectbox(
                "Select SVM Kernel type:",
                ["Linear", "Polynomial", "Radial Basis Function (RBF)"]
            )

            st.write("""
                - **Linear Kernel**: Best suited for linearly separable data (works well for problems like text classification).
                - **Polynomial Kernel**: Works well for problems where the relationship between data points is non-linear but still simple enough to be represented as a polynomial function.
                - **RBF (Radial Basis Function) Kernel**: Suitable for data that is highly non-linear and when the data cannot be separated by a linear boundary. It is the most commonly used kernel.
            """)

            # Train the model with the selected kernel
            model = SVC(kernel=kernel_type.lower(), probability=True)
            model.fit(X_train, y_train)

            # Visualizing the Support Vector Machine decision boundary
            fig, ax = plt.subplots(figsize=(10, 6))
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
            Z = model.predict_proba(np.c_[xx, yy])[:, 1]
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, 25, cmap="RdBu", alpha=0.8)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", cmap="RdBu", s=30)
            ax.set_title(f"SVM Decision Boundary ({kernel_type} Kernel)")
            st.pyplot(fig)

            # Allowing the user to download the SVM plot
            download_image(fig, f"svm_decision_boundary_{kernel_type}.jpg")

        # Evaluation and prediction text
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
        
        # Making a prediction on new data point
        st.write("Enter a new data point for prediction:")
        new_data = []
        for feature in features:
            value = st.number_input(f"Enter value for {feature}")
            new_data.append(value)

        # Scaling the new data point and making the prediction
        new_data_scaled = scaler.transform([new_data])
        prediction = model.predict(new_data_scaled)
        prediction_label = label_encoders[target].inverse_transform(prediction)
        st.write(f"Predicted class for the entered data point: {prediction_label[0]}")
