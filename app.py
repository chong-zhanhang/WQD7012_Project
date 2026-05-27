import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD ARTIFACTS
# ==========================

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Numeric columns used during scaler fit
numeric_cols = list(scaler.feature_names_in_)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    layout="wide"
)

st.title("💓 Heart Attack Risk Prediction")
st.write("Enter patient details to estimate heart attack risk.")

# ==========================
# INPUT FORM
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    cholesterol_level = st.number_input("Total Cholesterol", 100, 400, 200)
    obesity = st.selectbox("Obesity", [0, 1])
    waist_circumference = st.number_input("Waist Circumference", 50, 200, 90)

with col2:
    family_history = st.selectbox("Family History", [0, 1])
    sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
    blood_pressure_systolic = st.number_input("Systolic BP", 80, 250, 120)
    blood_pressure_diastolic = st.number_input("Diastolic BP", 40, 150, 80)
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar", 50, 300, 100)
    cholesterol_hdl = st.number_input("HDL Cholesterol", 10, 150, 50)

with col3:
    cholesterol_ldl = st.number_input("LDL Cholesterol", 10, 300, 120)
    triglycerides = st.number_input("Triglycerides", 20, 500, 150)
    previous_heart_disease = st.selectbox("Previous Heart Disease", [0, 1])
    medication_usage = st.selectbox("Medication Usage", [0, 1])
    participated_in_free_screening = st.selectbox("Participated in Free Screening", [0, 1])

# Categorical inputs
gender = st.selectbox("Gender", ["Female", "Male"])
region = st.selectbox("Region", ["Rural", "Urban"])
income_level = st.selectbox("Income Level", ["High", "Middle", "Low"])
smoking_status = st.selectbox("Smoking Status", ["Current", "Past", "Never"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["High", "Moderate", "None"])
physical_activity = st.selectbox("Physical Activity", ["High", "Moderate", "Low"])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
air_pollution_exposure = st.selectbox("Air Pollution Exposure", ["High", "Moderate", "Low"])
stress_level = st.selectbox("Stress Level", ["High", "Moderate", "Low"])
EKG_results = st.selectbox("EKG Results", ["Abnormal", "Normal"])

# ==========================
# BUILD RAW INPUT
# ==========================

input_df = pd.DataFrame([{
    "age": age,
    "hypertension": hypertension,
    "diabetes": diabetes,
    "cholesterol_level": cholesterol_level,
    "obesity": obesity,
    "waist_circumference": waist_circumference,
    "family_history": family_history,
    "sleep_hours": sleep_hours,
    "blood_pressure_systolic": blood_pressure_systolic,
    "blood_pressure_diastolic": blood_pressure_diastolic,
    "fasting_blood_sugar": fasting_blood_sugar,
    "cholesterol_hdl": cholesterol_hdl,
    "cholesterol_ldl": cholesterol_ldl,
    "triglycerides": triglycerides,
    "previous_heart_disease": previous_heart_disease,
    "medication_usage": medication_usage,
    "participated_in_free_screening": participated_in_free_screening,
    "gender": gender,
    "region": region,
    "income_level": income_level,
    "smoking_status": smoking_status,
    "alcohol_consumption": alcohol_consumption,
    "physical_activity": physical_activity,
    "dietary_habits": dietary_habits,
    "air_pollution_exposure": air_pollution_exposure,
    "stress_level": stress_level,
    "EKG_results": EKG_results
}])

# ==========================
# PREPROCESS
# ==========================

# One-hot encode
input_encoded = pd.get_dummies(input_df)

# Match training columns
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# Scale ONLY numeric columns
input_encoded[numeric_cols] = scaler.transform(
    input_encoded[numeric_cols]
)

# ==========================
# PREDICT
# ==========================

if st.button("Predict Risk"):

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="Heart Attack Risk Probability",
        value=f"{probability:.2%}"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack")
        st.write("Recommendation: Immediate clinical follow-up advised.")
    else:
        st.success("✅ Low Risk of Heart Attack")
        st.write("Recommendation: Maintain healthy lifestyle and regular monitoring.")