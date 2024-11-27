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
from sklearn.datasets import load_iris
import seaborn as sns
import os

# Function to save visualizations
def save_plot(fig, filename):
    plot_path = os.path.join("plots", filename)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig(plot_path, format='jpg')
    st.download_button(label="Download Plot", data=open(plot_path, 'rb'), file_name=filename, mime='image/jpeg')

# Load dataset
data = load_iris()  # You can replace this with your dataset
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X = df.drop('target', axis=1)
y = df['target']

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
st.write("Label Encoding done. Below are the classes and their respective encoded values:")
st.write(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scaler selection
scaler_option = st.selectbox(
    "Select Scaler", 
    ["Standard Scaler", "Min-Max Scaler"], 
    help="Standard Scaler works well with algorithms that assume Gaussian distributions. Min-Max Scaler is useful for algorithms that rely on distance measures (e.g., KNN)."
)

if scaler_option == "Standard Scaler":
    scaler = StandardScaler()
elif scaler_option == "Min-Max Scaler":
    scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
st.subheader("K-Nearest Neighbors (KNN) Model")
knn_metric = st.selectbox(
    "Select KNN Metric", 
    ["euclidean", "manhattan", "minkowski"], 
    help="Euclidean is ideal for continuous features, Manhattan works better for sparse data, Minkowski is a generalization of both."
)

# Finding the best number of neighbors
st.write("Finding the best value for n_neighbors...")
k_range = list(range(1, 21))
best_score = 0
best_k = 1
knn_accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric=knn_metric)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    knn_accuracy.append(scores.mean())
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

# Plot the KNN accuracy curve
fig, ax = plt.subplots()
ax.plot(k_range, knn_accuracy, marker='o')
ax.set_xlabel("Number of Neighbors (k)")
ax.set_ylabel("Cross-Validated Accuracy")
ax.set_title(f"KNN Accuracy for Different k Values ({knn_metric} Metric)")
st.pyplot(fig)
save_plot(fig, f"knn_accuracy_curve_{knn_metric}.jpg")

st.write(f"Best value of n_neighbors: {best_k} with accuracy: {best_score * 100:.2f}%")

# Training KNN with best k
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=knn_metric)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
knn_accuracy_test = accuracy_score(y_test, y_pred_knn)
st.write(f"KNN Test Accuracy: {knn_accuracy_test * 100:.2f}%")

# SVM Model
st.subheader("Support Vector Machine (SVM) Model")
kernel_type = st.selectbox(
    "Select SVM Kernel Type", 
    ["linear", "poly", "rbf"], 
    help="Linear is ideal for linearly separable data, Poly for non-linear, and RBF for complex decision boundaries."
)

svm_model = SVC(kernel=kernel_type)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
st.write(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")

# Plot SVM Decision Boundary
fig_svm, ax_svm = plt.subplots()
sns.scatterplot(x=X_test_scaled[:, 0], y=X_test_scaled[:, 1], hue=y_test, palette="viridis", ax=ax_svm)
ax_svm.set_title(f"SVM Decision Boundary with {kernel_type} kernel")
plt.tight_layout()
st.pyplot(fig_svm)
save_plot(fig_svm, f"svm_decision_boundary_{kernel_type}.jpg")

# Decision Tree Model
st.subheader("Decision Tree Model")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
st.write(f"Decision Tree Test Accuracy: {dt_accuracy * 100:.2f}%")

# Plot Decision Tree
fig_dt, ax_dt = plt.subplots(figsize=(12, 8))
from sklearn.tree import plot_tree
plot_tree(dt_model, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, ax=ax_dt)
plt.title("Decision Tree Visualization")
st.pyplot(fig_dt)
save_plot(fig_dt, "decision_tree_visualization.jpg")

# Evaluation
model_choice = st.selectbox(
    "Select Model for Prediction", 
    ["KNN", "SVM", "Decision Tree"]
)

if model_choice == "KNN":
    model = knn_model
elif model_choice == "SVM":
    model = svm_model
else:
    model = dt_model

st.write("Enter a new data point for prediction:")

# User inputs for prediction
new_data = []
for feature in X.columns:
    value = st.number_input(f"Enter value for {feature}")
    new_data.append(value)

# Scaling and prediction
new_data_scaled = scaler.transform([new_data])
prediction = model.predict(new_data_scaled)
prediction_label = label_encoder.inverse_transform(prediction)
st.write(f"Predicted class for the entered data point: {prediction_label[0]}")
