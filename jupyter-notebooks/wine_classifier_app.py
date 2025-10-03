import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
col_names = ["Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
             "Total_phenols", "Flavanoids", "NonFlavanoid_phenols", "Proanthocyanins",
             "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"]

df = pd.read_csv("wine.data", names=col_names)

# Features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Sidebar - choose model
st.sidebar.title("Wine Class Predictor")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "KNN"])

st.title("Wine Dataset Classification")

# Sidebar - user inputs for all features
st.sidebar.subheader("Input Wine Features")
inputs = {}
for feature in X.columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    inputs[feature] = st.sidebar.number_input(feature, min_value=min_val, max_value=max_val, value=mean_val)

# Convert inputs to array
features = np.array([list(inputs.values())])

# Standardize features for models that need it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
features_scaled = scaler.transform(features)

# Train models
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_scaled, y)

dtree = DecisionTreeClassifier()
dtree.fit(X, y)  # Decision Tree does not require scaling

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Predict button
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = log_reg.predict(features_scaled)
    elif model_choice == "Decision Tree":
        prediction = dtree.predict(features)
    else:  # KNN
        prediction = knn.predict(features_scaled)

    st.write("### Predicted Wine Class:", int(prediction[0]))
