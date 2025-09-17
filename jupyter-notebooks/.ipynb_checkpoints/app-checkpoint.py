import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.mode_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

# App Header 

st.title("House Price Prediction Dashboard")
st.markdown("This dashboard lets you explore the **House Prices Dataset**, run EDA, and compare ML models.")

# Load Dataset

@st.cache_data 
def load_data():
    df = pd.read_csv("house_prices_cleaned.csv")
    return df
    
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing 

numeric_features = ["LotArea", "OverallQual", "OverallCond", "GrLivArea", "GarageCars"] 
categ_features = ["Neighborhood", "Foundation"]

X = df[numeric_features + categ_features].copy()
y = df["SalePrice"]

X[["LotArea", "GrLivArea"]] = X[["LotArea", "GrLivArea"]].astype(float)

scaler = MinMaxScaler()
X[["LotArea", "GrLivArea"]] = scaler.fit_transform(X[["LotArea", "GrLivArea"]])

X = pd.get_dummies(X, columns=categ_features, drop_first=True) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar Filters (interactive EDA)

st.sidebar.header("EDA Controls")

## Numeric columns selection 

selected_num = st.sidebar.selectbox("Select a numeric column", numeric_features)
num_min = float(df[selected_num].min())
num_max = float(df[selected_num].max())
num_range = st.sidebar.slider(f"Filter {selected_num}", num_min, num_max, (num_min, num_max))
df_filtered = df[(df[selected_num] >= num_range[0]) & (df[selected_num] <= num_range[1])]

## Categorical columns selection 

selected_cat = st.sidebar.selectbox("Select a categorical column", categ_features)
selected_val = st.sidebar.selectbox(f"Choose value for {selected_cat}", df[selected_cat].unique())
df_filtered = df_filtered[df_filtered[selected_cat] == selected_val]

# EDA Section 

st.subheader("Exploratory Data Analysis")

#1. Histogram of the selected numeric column 
st.write(f"Histogram of **{selected_num}**")
fig, ax = plt.subplots()
sns.histplot(df_filtered[selected_num], bins=30, kde=True, ax=ax, color='skyblue')
st.pyplot(fig)

#2. Scatterplot between two numeric columns 
st.write("Scatterplot between two numeric columns")
col_x = st.selectbox("X-axis", numeric_features, index=0)
col_y = st.selectbox("Y-axis", numeric_features, index=1)
fig, ax = plt.subplots()
sns.scatterplot(x=df_filtered[col_x], y=df_filtered[col_y], ax=ax)
st.pyplot(fig)

#3. Bar chart for the selected categorical column 
st.write(f"Bar chart of **{selected_cat}**")
fig, ax = plt.subplots()
df_filtered[selected_cat].value_counts().plot(kind='bar', ax=ax, color='lightgreen')
plt.xticks(rotation=45)
st.pyplot(fig)

# Model Comparison 

st.subheader("Model Comparison")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R²": r2_score(y_test, preds),
    }

results_df = pd.DataFrame(results).T
st.dataframe(results_df.style.highlight_max(axis=0))

# Prediction Demo 

st.subheader("Prediction Demo")

lotarea = st.number_input("Lot Area", min_value=int(df["LotArea"].min()), max_value=int(df["LotArea"].max()), value=int(df["LotArea"].median()))
overallqual = st.slider("Overall Quality (1-10)", 1, 10, int(df["OverallQual"].median()))
overallcond = st.slider("Overall Condition (1-10)", 1, 10, int(df["OverallCond"].median()))
grlivarea = st.number_input("Living Area (sqft)", min_value=int(df["GrLivArea"].min()), max_value=int(df["GrLivArea"].max()), value=int(df["GrLivArea"].median()))
garagecars = st.slider("Garage Cars", 0, 4, int(df["GarageCars"].median()))
neighborhood = st.selectbox("Neighborhood", df["Neighborhood"].unique())
foundation = st.selectbox("Foundation", df["Foundation"].unique())

input_data = pd.DataFrame(
    {
        "LotArea": [lotarea],
        "OverallQual": [overallqual],
        "OverallCond": [overallcond],
        "GrLivArea": [grlivarea],
        "GarageCars": [garagecars],
        "Neighborhood": [neighborhood],
        "Foundation": [foundation],
    }
)

# Preprocess input 

input_data[["LotArea", "GrLivArea"]] = scaler.transform(input_data[["LotArea", "GrLivArea"]])
input_data = pd.get_dummies(input_data, columns=categ_features, drop_first=True)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Predict with Random Forest
best_model = RandomForestRegressor(random_state=42, n_estimators=100)
best_model.fit(X_train, y_train)
pred_price = best_model.predict(input_data)[0]

st.success(f"Predicted House Price: ${pred_price:,.2f}")

## Classification not applied because target variable "SalePrice" is numeric, not categorical