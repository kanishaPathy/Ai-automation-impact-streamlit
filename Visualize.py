import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

# Load model and encoders
model = joblib.load("models/xgboost_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Load datasets
df = pd.read_csv("merged_data_updated.csv")
# df_encoded = pd.read_csv('datasets/merged_combined_data_encoded.csv')

# App title
st.title("AI & Automation Impact Prediction Dashboard")

# Sidebar filters
country = st.sidebar.selectbox("Select Country", sorted(df["Country"].unique()))
sector = st.sidebar.selectbox("Select Sector", sorted(df["Sector"].unique()))
education = st.sidebar.selectbox("Select Education Level", sorted(df["EducationLevel"].unique()))
year_range = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2015, 2024))

# Filtered data for visualizations
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# Line chart: PreAI vs PostAI impact
st.subheader("Unemployment Impact Before vs After AI")
fig1, ax1 = plt.subplots()
sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Bar chart: AI vs Automation Impact
st.subheader("AI vs Automation Impact")
fig2, ax2 = plt.subplots()
bar_width = 0.35
years = filtered_df["Year"].astype(str)
x = range(len(years))
ax2.bar(x, filtered_df["Avg_Automation_Impact"], width=bar_width, label="Automation")
ax2.bar([i + bar_width for i in x], filtered_df["Avg_AI_Role_Jobs"], width=bar_width, label="AI Role Jobs")
ax2.set_xticks([i + bar_width / 2 for i in x])
ax2.set_xticklabels(years, rotation=45)
ax2.set_ylabel("Impact")
ax2.legend()
st.pyplot(fig2)

# Prediction button
if st.button("Predict Future Impact"):
    # Prepare prediction input (first year in the selected range)
    input_df = pd.DataFrame({
        'Country': [country],
        'Sector': [sector],
        'Year': [year_range[0]],
        'EducationLevel': [education]
    })

    # Add the rest of the features using mean from original dataset
    additional_features = [
        'Avg_PreAI', 'Avg_PostAI', 'Avg_Automation_Impact', 'Avg_AI_Role_Jobs',
        'Avg_ReskillingPrograms', 'Avg_EconomicImpact', 'Skill_Level', 'Skills_Gap',
        'Reskilling_Demand', 'Upskilling_Programs', 'Automation_Impact_Level',
        'Revenue', 'Growth_Rate', 'AI_Adoption_Rate', 'Automation_Level',
        'Sector_Impact_Score', 'Tech_Investment', 'Sector_Growth_Decline',
        'Male_Percentage', 'Female_Percentage'
    ]
    
    for col in additional_features:
        input_df[col] = df[col].mean()

    # Apply label encoding to categorical variables
    for col in ['Country', 'Sector', 'EducationLevel']:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict using model
    prediction = model.predict(input_df)[0]

    # Display prediction
    st.success(f"Predicted Impact Score for {year_range[0]}: {prediction:.2f}")
