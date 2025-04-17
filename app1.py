
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load your trained XGBoost model
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")

st.title("AI Automation Impact Prediction")

# Inputs
year = st.selectbox("Select Year", [2021, 2022, 2023])
country = st.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT', 
                                        'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'])
avg_pre_ai = st.slider("Average Pre-AI Impact", 0, 100, 50)
avg_post_ai = st.slider("Average Post-AI Impact", 0, 100, 50)
avg_automation_impact = st.slider("Average Automation Impact", 0, 100, 50)
avg_ai_role_jobs = st.slider("Average AI Role Jobs", 0, 100, 50)
avg_reskilling_programs = st.slider("Average Reskilling Programs", 0, 100, 50)
avg_economic_impact = st.slider("Average Economic Impact", 0, 100, 50)
avg_sector_growth = st.slider("Average Sector Growth", 0, 100, 50)

# Create DataFrame from inputs
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year],
    '_id.EducationLevel': [education_level],
    'Avg_PreAI': [avg_pre_ai],
    'Avg_PostAI': [avg_post_ai],
    'Avg_Automation_Impact': [avg_automation_impact],
    'Avg_AI_Role_Jobs': [avg_ai_role_jobs],
    'Avg_ReskillingPrograms': [avg_reskilling_programs],
    'Avg_EconomicImpact': [avg_economic_impact],
    'Avg_SectorGrowth': [avg_sector_growth]
})

# One-hot encode categorical variables
input_encoded = pd.get_dummies(input_df)

# Load training data to get full feature list (one-hot encoded)
training_df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")
training_encoded = pd.get_dummies(training_df.drop(columns=['Avg_Automation_Impact']))  # Target column dropped

# Match input with training features
missing_cols = set(training_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0

# Reorder columns to match training data
input_encoded = input_encoded[training_encoded.columns]

# Predict
prediction = model.predict(input_encoded)

# Show result
st.success(f"Predicted Automation Impact: {prediction[0]:.2f}")

# Feature Importance Visualization
feature_importance = model.feature_importances_
feature_names = training_encoded.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.subheader("Feature Importance")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
st.pyplot()

# Confidence Interval (using bootstrap method)
predictions = [model.predict(input_encoded) for _ in range(1000)]
predictions = np.array(predictions).flatten()
lower_bound = np.percentile(predictions, 2.5)
upper_bound = np.percentile(predictions, 97.5)

st.write(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

# User Feedback
st.subheader("How accurate do you think this prediction is?")
accuracy_rating = st.slider("Rate prediction accuracy", 1, 5, 3)
st.write(f"Thank you for your feedback! You rated the prediction accuracy as: {accuracy_rating}/5")

# Writing feedback to file (ensure this is properly formatted)
with open("feedback.txt", "a") as f:
    f.write(f"Prediction: {prediction[0]:.2f}, Accuracy Rating: {accuracy_rating}
")

# Batch Prediction (File Upload)
st.subheader("Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_encoded = pd.get_dummies(batch_df)
    missing_cols = set(training_encoded.columns) - set(batch_encoded.columns)
    for col in missing_cols:
        batch_encoded[col] = 0
    batch_encoded = batch_encoded[training_encoded.columns]

    batch_predictions = model.predict(batch_encoded)
    batch_df['Predicted_Automation_Impact'] = batch_predictions
    st.write(batch_df)

    st.download_button(
        label="Download Predictions",
        data=batch_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
