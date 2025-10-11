import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Load dataset
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, "wine.data")

col_names = ["Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
             "Total_phenols", "Flavanoids", "NonFlavanoid_phenols", "Proanthocyanins",
             "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"]

df = pd.read_csv(dataset_path, names=col_names)

# Features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

# Sidebar - model selection
st.sidebar.title("Wine Class Predictor")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
)

st.title("🍷 Ensemble Learning on Wine Dataset")

# Sidebar - input features
st.sidebar.subheader("Input Wine Features")
inputs = {}
for feature in X.columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    inputs[feature] = st.sidebar.number_input(feature, min_value=min_val, max_value=max_val, value=mean_val)

# Convert to array
features = np.array([list(inputs.values())])
features_scaled = scaler.transform(features)

# Model selection for prediction
if model_choice == "Logistic Regression":
    model = log_reg
    input_data = features_scaled
elif model_choice == "Decision Tree":
    model = dtree
    input_data = features
elif model_choice == "KNN":
    model = knn
    input_data = features_scaled
elif model_choice == "Random Forest":
    model = rf
    input_data = features
else:
    model = xgb
    input_data = features

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"### Predicted Wine Class: {int(prediction[0])}")

# --- Model Performance Comparison ---
st.subheader("📊 Model Performance Comparison")

models = {
    "Decision Tree": dtree,
    "Random Forest": rf,
    "XGBoost": xgb
}

metrics = []
for name, m in models.items():
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    metrics.append([name, acc, prec, rec])

metrics_df = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall"])
st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

# --- Feature Importance Visualization ---
st.subheader("🔍 Feature Importance Comparison")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
axes[0].barh(rf_importances.index[:8], rf_importances.values[:8])
axes[0].set_title("Random Forest Importance")

# XGBoost
xgb_importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
axes[1].barh(xgb_importances.index[:8], xgb_importances.values[:8], color='orange')
axes[1].set_title("XGBoost Importance")

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.caption("Ensemble Learning using Wine Dataset")