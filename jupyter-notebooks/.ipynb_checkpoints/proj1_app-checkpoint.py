# Import libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

# App Header

st.title("Heart Disease Classification Dashboard")
st.markdown("This dashboard lets you explore the **Heart Disease UCI Dataset**, perform **Exploratory Data Analysis (EDA)**, and compare different **Machine Learning Models**. You can also try the interactive sidebar to predict whether a patient has **Heart Disease**.") 

# Load dataset

@st.cache_data 
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Data Cleaning 

## Fix numeric conversions: look for columns that should be numeric but are object 

for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except:
            # If conversion fails, leave it as object (it's categorical)
            pass

## Drop unnecessary/sparse columns
df.drop(columns=["fbs", "exang", "ca", "thal"], inplace=True, errors="ignore")

## Impute median for continuous variables 
for col in ["trestbps", "chol", "thalch"]:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val) 

## oldpeak: drop missing rows, fix negatives 
df = df.dropna(subset=["oldpeak"])
df["oldpeak"] = df["oldpeak"].abs()

## slope: drop rows where missing 
df = df.dropna(subset=["slope"])

## Reset index after cleaning
df.reset_index(drop=True, inplace=True) 

df.to_csv("heart_disease_cleaned.csv", index=False)

# Preprocessing 

## Select features 

numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak"]
categ_features = ["sex", "cp", "restecg", "slope"]

X = df[numeric_features + categ_features].copy()
y = df["num"].apply(lambda x: 1 if x > 0 else 0)

## Convert to float
X[numeric_features] = X[numeric_features].astype(float)

## Scale numeric features
scaler = MinMaxScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

## Encode categorical features 
X = pd.get_dummies(X, columns=categ_features, drop_first=True)

## Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Comparison 

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

st.subheader("Model Performance Comparison") 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    st.write(f"{name}") 
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred))) 
    if y_prob is not None:
        st.write("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 3))
    st.markdown("---")

# EDA Section 

st.header("Exploratory Data Analysis")

# 1. Basic Statistical Analysis 

st.subheader("Basic Statistical Analysis")
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns 

for col in numeric_cols:
    st.write(f"\nColumn:{col}") 
    st.write("Mean:", df[col].mean())
    st.write("Median:", df[col].median())
    st.write("Mode:", df[col].mode()[0])
    st.write("Variance:", df[col].var())
    st.write("Standard Deviation:", df[col].std()) 
    st.markdown("---")

# 2. Pie Chart for Gender Distribution 
st.subheader("Gender Distribution")
fig1, ax1 = plt.subplots()
gender_counts = df['sex'].value_counts()
ax1.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['skyblue', 'lightpink'])
ax1.set_title("Gender Distribution")
st.pyplot(fig1)

# 3. Pie Chart for Disease Distribution 
st.subheader("Heart Disease Distribution")
fig2, ax2 = plt.subplots()
target_counts = df['num'].apply(lambda x: 1 if x>0 else 0).value_counts()
ax2.pie(target_counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
ax2.set_title("Disease Distribution")
st.pyplot(fig2)

# 4. Histograms of Numeric Features
st.subheader("Histograms of Numeric Features")
fig3 = df[numeric_features].hist(figsize=(12,6), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
st.pyplot(plt.gcf())

# 5. Bar Charts for Categorical Features
st.subheader("Categorical Feature Distributions")
for col in ["cp", "restecg", "slope"]:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, palette='pastel', ax=ax)
    ax.set_title(f"Bar Chart of {col}")
    st.pyplot(fig)

# 6. Scatter Plots for Relationships with Disease Status 

st.subheader("Scatter Plots by Disease Status")

fig4, ax4 = plt.subplots()
ax4.scatter(df['age'], df['thalch'], c=df['num'].apply(lambda x: 1 if x>0 else 0), cmap='coolwarm', alpha=0.7)
ax4.set_xlabel("Age")
ax4.set_ylabel("Max Heart Rate (thalch)")
ax4.set_title("Age vs Max Heart Rate by Disease Status")
st.pyplot(fig4)

fig5, ax5 = plt.subplots()
ax5.scatter(df['chol'], df['oldpeak'], c=df['num'].apply(lambda x: 1 if x>0 else 0), cmap='coolwarm', alpha=0.7)
ax5.set_xlabel("Cholesterol (chol)")
ax5.set_ylabel("ST Depression (oldpeak)")
ax5.set_title("Cholesterol vs ST Depression by Disease Status")
st.pyplot(fig5)

# Interative Prediction 

st.header("Heart Disease Prediction")

## Pick best model (Random Forest)
best_model=RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

st.sidebar.header("Patient Input Features")

## Collect user input
user_input = {}
for col in numeric_features:
    user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))

for col in categ_features:
    user_input[col] = st.sidebar.selectbox(f"{col}", df[col].unique())

# Convert to dataframe 
input_df = pd.DataFrame([user_input])

# Scale numeric
input_df[numeric_features] = scaler.transform(input_df[numeric_features])
# Encode categorical
input_df = pd.get_dummies(input_df, columns=categ_features, drop_first=True)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = best_model.predict(input_df)[0]
prediction_proba = best_model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ The model predicts: **Heart Disease** (Probability: {prediction_proba:.2f})")
else:
    st.success(f"✅ The model predicts: **No Heart Disease** (Probability: {prediction_proba:.2f})")
