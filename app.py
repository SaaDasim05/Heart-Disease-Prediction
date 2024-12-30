import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load a sample dataset (replace with actual heart disease dataset)
# This is simulated data for demonstration purposes
st.set_page_config(page_title="AI HEART DISEASE PREDICTION",page_icon="https://img.freepik.com/free-vector/novel-coronavirus-ncov-abstract-concept-vector-illustration-novel-coronavirus-disease-outbreak-ncov-infection-prevention-control-preventive-measures-covid-19-statistics-abstract-metaphor_335657-1595.jpg?t=st=1735579269~exp=1735582869~hmac=61ee56fcaed6fdadf3b9a50953b20681f83776170cbcddeb94b57eaf50b293c1&w=740")
data = {
    'Age': np.random.randint(29, 78, 100),
    'Gender': np.random.choice([0, 1], 100),
    'Cp': np.random.choice([0, 1, 2, 3], 100),
    'Trestbps': np.random.randint(94, 201, 100),
    'Chol': np.random.randint(126, 565, 100),
    'Fbs': np.random.choice([0, 1], 100),
    'Restecg': np.random.choice([0, 1, 2], 100),
    'Thalach': np.random.randint(71, 203, 100),
    'Exang': np.random.choice([0, 1], 100),
    'Oldpeak': np.random.uniform(0, 6.3, 100),
    'Slope': np.random.choice([0, 1, 2], 100),
    'Ca': np.random.choice([0, 1, 2, 3], 100),
    'Thal': np.random.choice([0, 1, 2], 100),
    'Target': np.random.randint(0, 2, 100),
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train-test split
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Prediction")

st.write("This app predicts the likelihood of heart disease based on user-provided features.")

# Feature descriptions
with st.expander("Feature Descriptions"):
    st.write("""
    1. **Age**: Age in years (29 to 77)
    2. **Gender**: Gender (1 = male, 0 = female)
    3. **Cp**: Chest pain type:
        - 0: typical angina
        - 1: atypical angina
        - 2: non-anginal pain
        - 3: asymptomatic
    4. **Trestbps**: Resting blood pressure in mm Hg (94 to 200)
    5. **Chol**: Serum cholesterol in mg/dL (126 to 564)
    6. **Fbs**: Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
    7. **Restecg**: Resting electrocardiographic results:
        - 0: Normal
        - 1: ST-T wave abnormality
        - 2: Left ventricular hypertrophy
    8. **Thalach**: Maximum heart rate achieved (71 to 202)
    9. **Exang**: Exercise-induced angina (1 = yes, 0 = no)
    10. **Oldpeak**: Stress test depression induced by exercise (0 to 6.2)
    11. **Slope**: Slope of the peak exercise ST segment:
        - 0: Upsloping
        - 1: Flat
        - 2: Downsloping
    12. **Ca**: Number of major vessels colored by fluoroscopy (0 to 3)
    13. **Thal**: Thallium heart rate:
        - 0: Normal
        - 1: Fixed defect
        - 2: Reversible defect
    """)

# User inputs
st.sidebar.header("Enter the Features:")
user_input = {
    'Age': st.sidebar.slider("Age", 29, 77, 50),
    'Gender': st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
    'Cp': st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]),
    'Trestbps': st.sidebar.slider("Resting Blood Pressure", 94, 200, 120),
    'Chol': st.sidebar.slider("Serum Cholesterol", 126, 564, 200),
    'Fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True"),
    'Restecg': st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2],
                                    format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x]),
    'Thalach': st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150),
    'Exang': st.sidebar.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    'Oldpeak': st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0),
    'Slope': st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                  format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]),
    'Ca': st.sidebar.slider("Number of Major Vessels", 0, 3, 1),
    'Thal': st.sidebar.selectbox("Thallium Heart Rate", options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x]),
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Make prediction
prediction = model.predict(user_df)[0]
prediction_percentage = prediction * 100

st.subheader("Prediction")
st.write(f"The predicted likelihood of heart disease is: **{prediction_percentage:.2f}%**")

st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
st.write(f"Mean Squared Error on Test Data: {mean_squared_error(y_test, y_pred):.2f}")
