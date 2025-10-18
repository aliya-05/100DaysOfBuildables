import streamlit as st
import pandas as pd
import joblib 

# Load trained model 
model = joblib.load("tuned_random_forest.pkl")

st.title("Titanic Passenger Class Predictor")
st.write("Enter passenger details to predict their travel class (1st, 2nd or 3rd):")

# Input features 
sex = st.selectbox("Sex", ["male", "female"])
fare = st.slider("Fare", 0, 500, 50)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
age_group = st.selectbox("Age Group", ["Children", "Teenage", "Adult", "Elder"])
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Convert inputs to match model columns
input_dict = {
    "SibSp": sibsp,
    "Parch": parch,
    "Sex_female": 1 if sex == "female" else 0,
    "Sex_male": 1 if sex == "male" else 0,
    "Age_Children": 1 if age_group == "Children" else 0,
    "Age_Teenage": 1 if age_group == "Teenage" else 0,
    "Age_Adult": 1 if age_group == "Adult" else 0,
    "Age_Elder": 1 if age_group == "Elder" else 0,
    "Embarked_C": 1 if embarked == "C" else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0,
    "Fare_Low_fare": 1 if fare < 50 else 0,
    "Fare_median_fare": 1 if 50 <= fare < 100 else 0,
    "Fare_Average_fare": 1 if 100 <= fare < 200 else 0,
    "Fare_high_fare": 1 if fare >= 200 else 0
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Class"):
    prediction = model.predict(input_df)
    st.success(f" Predicted Passenger Class: {prediction[0]}")
