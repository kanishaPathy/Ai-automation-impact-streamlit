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

# At the top of your file
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ... (user input code remains unchanged)

# ---------- Matplotlib/Seaborn Plots ----------
# 1. Pre-AI vs Post-AI Unemployment
st.subheader("Unemployment Impact Before vs After AI")
fig1, ax1 = plt.subplots(figsize=(6, 2.5))  # Reduced size
sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# 2. AI vs Automation Impact
st.subheader("AI vs Automation Impact")
fig2, ax2 = plt.subplots(figsize=(6, 2.5))
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

# 3. Reskilling and Upskilling
st.subheader("Reskilling & Upskilling Programs Trend")
fig3, ax3 = plt.subplots(figsize=(6, 2.5))
sns.lineplot(data=filtered_df, x="Year", y="Reskilling_Demand", label="Reskilling Demand", marker="o", ax=ax3)
sns.lineplot(data=filtered_df, x="Year", y="Upskilling_Programs", label="Upskilling Programs", marker="o", ax=ax3)
ax3.set_ylabel("Programs / Demand Level")
ax3.set_xticks(filtered_df["Year"].unique())
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# 4. Gender Distribution
st.subheader("Gender Distribution in Employment (%)")
fig4, ax4 = plt.subplots(figsize=(6, 2.5))
bar_width = 0.4
x = range(len(filtered_df["Year"]))
ax4.bar(x, filtered_df["Male_Percentage"], width=bar_width, label="Male")
ax4.bar([i + bar_width for i in x], filtered_df["Female_Percentage"], width=bar_width, label="Female")
ax4.set_xticks([i + bar_width / 2 for i in x])
ax4.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
ax4.set_ylabel("Percentage")
ax4.legend()
st.pyplot(fig4)

# 5. Tech Investment vs AI Adoption
st.subheader("Tech Investment vs AI Adoption Rate")
fig5, ax5 = plt.subplots(figsize=(6, 2.5))
sns.lineplot(data=filtered_df, x="Year", y="Tech_Investment", label="Tech Investment", marker="o", ax=ax5)
sns.lineplot(data=filtered_df, x="Year", y="AI_Adoption_Rate", label="AI Adoption Rate", marker="o", ax=ax5)
ax5.set_ylabel("Values")
ax5.set_xticks(filtered_df["Year"].unique())
ax5.tick_params(axis='x', rotation=45)
st.pyplot(fig5)

# 6. Sector Growth / Decline
st.subheader("Sector Growth/Decline Over Time")
fig6, ax6 = plt.subplots(figsize=(6, 2.5))
sns.barplot(data=filtered_df, x="Year", y="Sector_Growth_Decline", palette="coolwarm", ax=ax6)
ax6.set_ylabel("Growth/Decline Index")
ax6.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
st.pyplot(fig6)

# 7. Automation Level
st.subheader("Automation Level by Year")
fig7, ax7 = plt.subplots(figsize=(6, 2.5))
sns.lineplot(data=filtered_df, x="Year", y="Automation_Level", marker="o", ax=ax7)
ax7.set_ylabel("Automation Level")
ax7.set_xticks(filtered_df["Year"].unique())
ax7.tick_params(axis='x', rotation=45)
st.pyplot(fig7)

# ---------- Plotly Graphs (Resized) ----------
# 8. Unemployment vs Skills Gap (Plotly)
st.subheader("Unemployment Impact vs Skills Gap")
fig8 = px.line(
    filtered_df,
    x="Year",
    y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
    labels={"value": "Impact/Gap"},
    title="AI's Impact on Unemployment and Skills Gap",
    height=400
)
st.plotly_chart(fig8, use_container_width=True)

# 9. AI Adoption vs Sector Growth (Plotly)
st.subheader("AI Adoption vs Sector Growth")
fig9 = px.bar(
    filtered_df,
    x="Year",
    y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
    barmode="group",
    title="AI Adoption Rate vs Sector Growth Decline",
    height=400
)
st.plotly_chart(fig9, use_container_width=True)

# Preview
st.write("Filtered Data:", filtered_df.head())
st.write("Number of rows in filtered data:", filtered_df.shape[0])
