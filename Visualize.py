import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import plotly.express as px


# Load model and encoders
model = joblib.load("xgb_model_final.pkl")
label_encoders = joblib.load("label_encoders1.pkl")

# Load new dataset
df = pd.read_csv("FINAL__Compressed_Dataset.csv.gz")

# ... Keep all import and initial setup unchanged ...
# Set wide layout and app title
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# Load model and data
model = joblib.load("xgb_model_final.pkl")
label_encoders = joblib.load("label_encoders1.pkl")
df = pd.read_csv("FINAL__Compressed_Dataset.csv.gz")

# ---------- User Input Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2015, 2024))
country = col2.selectbox("Select Country", sorted(df["Country"].unique()))
sector = col3.selectbox("Select Sector", sorted(df["Sector"].unique()))
education = col4.selectbox("Select Education Level", sorted(df["EducationLevel"].unique()))

# Filter data for visualizations
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# --- Prediction Section ---
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
input_encoded = input_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]
st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# --- Visualization Sections ---
def center_plot(fig):
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.pyplot(fig)

def center_plotly(fig):
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.plotly_chart(fig, use_container_width=True)

# --- Unemployment Before vs After AI ---
st.subheader("Unemployment Impact Before vs After AI")
fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
ax1.set_title("Unemployment Before vs After AI", fontsize=12)
ax1.tick_params(axis='x', rotation=45)
center_plot(fig1)

# --- AI vs Automation Impact ---
st.subheader("AI vs Automation Impact")
fig2, ax2 = plt.subplots(figsize=(6, 3))
bar_width = 0.35
years = filtered_df["Year"].astype(str)
x = range(len(years))
ax2.bar(x, filtered_df["Avg_Automation_Impact"], width=bar_width, label="Automation")
ax2.bar([i + bar_width for i in x], filtered_df["Avg_AI_Role_Jobs"], width=bar_width, label="AI Role Jobs")
ax2.set_xticks([i + bar_width / 2 for i in x])
ax2.set_xticklabels(years, rotation=45)
ax2.legend()
center_plot(fig2)

# --- Reskilling & Upskilling ---
st.subheader("Reskilling & Upskilling Programs Trend")
fig3, ax3 = plt.subplots(figsize=(6, 3))
sns.lineplot(data=filtered_df, x="Year", y="Reskilling_Demand", label="Reskilling Demand", marker="o", ax=ax3)
sns.lineplot(data=filtered_df, x="Year", y="Upskilling_Programs", label="Upskilling Programs", marker="o", ax=ax3)
ax3.tick_params(axis='x', rotation=45)
center_plot(fig3)

# --- Gender Distribution ---
st.subheader("Gender Distribution in Employment (%)")
fig4, ax4 = plt.subplots(figsize=(6, 3))
x = range(len(filtered_df["Year"]))
ax4.bar(x, filtered_df["Male_Percentage"], width=0.4, label="Male")
ax4.bar([i + 0.4 for i in x], filtered_df["Female_Percentage"], width=0.4, label="Female")
ax4.set_xticks([i + 0.2 for i in x])
ax4.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
ax4.legend()
center_plot(fig4)

# --- Tech Investment vs AI Adoption ---
st.subheader("Tech Investment vs AI Adoption Rate")
fig5, ax5 = plt.subplots(figsize=(6, 3))
sns.lineplot(data=filtered_df, x="Year", y="Tech_Investment", label="Tech Investment", marker="o", ax=ax5)
sns.lineplot(data=filtered_df, x="Year", y="AI_Adoption_Rate", label="AI Adoption Rate", marker="o", ax=ax5)
ax5.tick_params(axis='x', rotation=45)
center_plot(fig5)

# --- Sector Growth/Decline ---
st.subheader("Sector Growth/Decline Over Time")
fig6, ax6 = plt.subplots(figsize=(6, 3))
filtered_df['Year'] = filtered_df['Year'].astype(str)
sns.barplot(data=filtered_df, x="Year", y="Sector_Growth_Decline", palette="coolwarm", ax=ax6)
ax6.set_xticklabels(filtered_df["Year"].unique(), rotation=45, ha="right")
fig6.tight_layout()
center_plot(fig6)

# --- Automation Level ---
st.subheader("Automation Level by Year")
fig7, ax7 = plt.subplots(figsize=(6, 3))
sns.lineplot(data=filtered_df, x="Year", y="Automation_Level", marker="o", ax=ax7)
ax7.tick_params(axis='x', rotation=45)
center_plot(fig7)

# --- Unemployment vs Skills Gap ---
st.subheader("Unemployment Impact vs Skills Gap")
fig8 = px.line(filtered_df, x="Year", y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
               labels={"value": "Impact/Gap"}, title="AI's Impact on Unemployment and Skills Gap")
center_plotly(fig8)

# --- AI Adoption vs Sector Growth ---
st.subheader("AI Adoption vs Sector Growth")
fig9 = px.bar(filtered_df, x="Year", y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
              barmode="group", title="AI Adoption Rate vs Sector Growth Decline")
center_plotly(fig9)

# --- Country vs Sector Comparison ---
st.header("📊 Country vs Selected Sectors Comparison")
selected_country = st.selectbox("Select Country", sorted(df["Country"].unique()), key="country_sector_view")
available_sectors = df["Sector"].unique()
selected_sectors = st.multiselect("Select Sectors", sorted(available_sectors), default=list(available_sectors[:2]), key="sector_multi")

comparison_df = df[(df["Country"] == selected_country) & (df["Sector"].isin(selected_sectors))]

if comparison_df.empty:
    st.warning("No data available for the selected filters.")
else:
    st.subheader(f"AI Adoption Rate over Years in {selected_country}")
    fig_sector_ai, ax_ai = plt.subplots(figsize=(6, 2.5))
    sns.lineplot(data=comparison_df, x="Year", y="AI_Adoption_Rate", hue="Sector", marker="o", ax=ax_ai)
    ax_ai.set_ylabel("AI Adoption Rate")
    ax_ai.set_xticks(sorted(comparison_df["Year"].unique()))
    ax_ai.tick_params(axis='x', rotation=45)
    fig_sector_ai.tight_layout()
    center_plot(fig_sector_ai)

    st.subheader(f"Automation Level over Years in {selected_country}")
    fig_sector_auto, ax_auto = plt.subplots(figsize=(6, 2.5))
    sns.lineplot(data=comparison_df, x="Year", y="Automation_Level", hue="Sector", marker="o", ax=ax_auto)
    ax_auto.set_ylabel("Automation Level")
    ax_auto.set_xticks(sorted(comparison_df["Year"].unique()))
    ax_auto.tick_params(axis='x', rotation=45)
    fig_sector_auto.tight_layout()
    center_plot(fig_sector_auto)
