import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import plotly.express as px

# Load model and encoders
model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load new dataset
df = pd.read_csv("FINAL__Compressed_Dataset.csv.gz")

# ... Keep all import and initial setup unchanged ...

# App title
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")

# Create 4 side-by-side columns
col1, col2, col3, col4 = st.columns(4)

# Inline controls instead of sidebar
year_range = col1.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2015, 2024)
)
country = col2.selectbox("Select Country", sorted(df["Country"].unique()))
sector = col3.selectbox("Select Sector", sorted(df["Sector"].unique()))
education = col4.selectbox("Select Education Level", sorted(df["EducationLevel"].unique()))

# Filtered data for visualizations
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# --- Unemployment Impact Before vs After AI ---
st.subheader("Unemployment Impact Before vs After AI")
col1, _ = st.columns([2, 1])  # Narrow second column to keep plot small
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

# --- AI vs Automation Impact ---
st.subheader("AI vs Automation Impact")
col2, _ = st.columns([2, 1])
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bar_width = 0.35
    years = filtered_df["Year"].astype(str)
    x = range(len(years))
    ax2.bar(x, filtered_df["Avg_Automation_Impact"], width=bar_width, label="Automation")
    ax2.bar([i + bar_width for i in x], filtered_df["Avg_AI_Role_Jobs"], width=bar_width, label="AI Role Jobs")
    ax2.set_xticks([i + bar_width / 2 for i in x])
    ax2.set_xticklabels(years, rotation=45)
    ax2.legend()
    st.pyplot(fig2)

# --- Prediction Button ---
if st.button("Predict Future Impact"):
    input_df = pd.DataFrame({
        'Country': [country],
        'Sector': [sector],
        'Year': [year_range[0]],
        'EducationLevel': [education]
    })
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
    for col in ['Country', 'Sector', 'EducationLevel']:
        input_df[col] = label_encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Impact Score for {year_range[0]}: {prediction:.2f}")
    
#Reskilling
st.subheader("Reskilling & Upskilling Programs Trend")
col3, _ = st.columns([2, 1])
with col3:
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Reskilling_Demand", label="Reskilling Demand", marker="o", ax=ax3)
    sns.lineplot(data=filtered_df, x="Year", y="Upskilling_Programs", label="Upskilling Programs", marker="o", ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

# --- Gender ---
st.subheader("Gender Distribution in Employment (%)")
col4, _ = st.columns([2, 1])
with col4:
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    bar_width = 0.4
    x = range(len(filtered_df["Year"]))
    ax4.bar(x, filtered_df["Male_Percentage"], width=bar_width, label="Male")
    ax4.bar([i + bar_width for i in x], filtered_df["Female_Percentage"], width=bar_width, label="Female")
    ax4.set_xticks([i + bar_width / 2 for i in x])
    ax4.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    ax4.legend()
    st.pyplot(fig4)

# --- Tech Investment vs AI Adoption ---
st.subheader("Tech Investment vs AI Adoption Rate")
col5, _ = st.columns([2, 1])
with col5:
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Tech_Investment", label="Tech Investment", marker="o", ax=ax5)
    sns.lineplot(data=filtered_df, x="Year", y="AI_Adoption_Rate", label="AI Adoption Rate", marker="o", ax=ax5)
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

# --- Sector Growth ---
st.subheader("Sector Growth/Decline Over Time")
col6, _ = st.columns([2, 1])
with col6:
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=filtered_df, x="Year", y="Sector_Growth_Decline", palette="coolwarm", ax=ax6)
    ax6.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    st.pyplot(fig6)

# --- Automation Level ---
st.subheader("Automation Level by Year")
col7, _ = st.columns([2, 1])
with col7:
    fig7, ax7 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Automation_Level", marker="o", ax=ax7)
    ax7.tick_params(axis='x', rotation=45)
    st.pyplot(fig7)

# --- Plotly Chart: Unemployment vs Skills Gap ---
st.subheader("Unemployment Impact vs Skills Gap")
col8, _ = st.columns([2, 1])
with col8:
    fig8 = px.line(
        filtered_df,
        x="Year",
        y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
        labels={"value": "Impact/Gap"},
        title="AI's Impact on Unemployment and Skills Gap"
    )
    st.plotly_chart(fig8, use_container_width=True)

# --- Plotly Chart: AI Adoption vs Sector Growth ---
st.subheader("AI Adoption vs Sector Growth")
col9, _ = st.columns([2, 1])
with col9:
    fig9 = px.bar(
        filtered_df,
        x="Year",
        y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
        barmode="group",
        title="AI Adoption Rate vs Sector Growth Decline"
    )
    st.plotly_chart(fig9, use_container_width=True)
