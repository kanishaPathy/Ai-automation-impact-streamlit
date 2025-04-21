import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the merged dataset
df = pd.read_csv("merged_automation_dataset.csv")

# Streamlit app layout
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("AI Automation Impact on Employment and Skills")

# Sidebar filters
st.sidebar.header("Filter Options")
year_range = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2010, 2024))
country = st.sidebar.selectbox("Select Country", df["Country"].unique())
sector = st.sidebar.selectbox("Select Sector", df["Sector"].unique())
education = st.sidebar.selectbox("Select Education Level", df["EducationLevel"].unique())

# Filtered data
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]

if filtered_df.empty:
    st.warning("⚠️ No data found for the selected combination. Please try different inputs.")
else:
    # Line chart: PreAI vs PostAI impact
    st.subheader("Unemployment Impact Before vs After AI")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
    ax1.set_xticks(filtered_df["Year"].unique())
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Bar chart: AI_Role_Jobs vs Automation_Impact
    st.subheader("AI Role Jobs vs Automation Impact")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(filtered_df["Year"], filtered_df["AI_Role_Jobs"], label="AI Role Jobs", alpha=0.7)
    ax2.set_ylabel("AI Role Jobs", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2_2 = ax2.twinx()
    ax2_2.plot(filtered_df["Year"], filtered_df["Automation_Impact"], label="Automation Impact", color="red", marker="o")
    ax2_2.set_ylabel("Automation Impact", color="red")
    ax2_2.tick_params(axis='y', labelcolor="red")
    ax2.set_xticks(filtered_df["Year"].unique())
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Skills Gap Analysis
    st.subheader("Skills Gap and Reskilling Demand")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(filtered_df["Year"], filtered_df["Skills_Gap"], label="Skills Gap", marker="o")
    ax3.plot(filtered_df["Year"], filtered_df["Reskilling_Demand"], label="Reskilling Demand", marker="s")
    ax3.set_xticks(filtered_df["Year"].unique())
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    st.pyplot(fig3)

    # Salary Trend
    st.subheader("Average Salary Over Time")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Salary_USD", marker="o", ax=ax4)
    ax4.set_xticks(filtered_df["Year"].unique())
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

    # Remote Opportunities Trend
    st.subheader("Remote Opportunities Over Time")
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Remote_Opportunities", marker="o", ax=ax5)
    ax5.set_xticks(filtered_df["Year"].unique())
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

    # Automation Impact by Skill Level
    st.subheader("Automation Impact by Skill Level")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=filtered_df, x="Skill_Level", y="Automation_Impact", ax=ax6)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
    st.pyplot(fig6)

    # Job Creation vs Reskilling Programs
    st.subheader("Job Creation vs Reskilling Programs")
    fig7, ax7 = plt.subplots(figsize=(6, 3))
    ax7.plot(filtered_df["Year"], filtered_df["Job_Creation"], label="Job Creation", marker="o")
    ax7.plot(filtered_df["Year"], filtered_df["ReskillingPrograms"], label="Reskilling Programs", marker="s")
    ax7.set_xticks(filtered_df["Year"].unique())
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend()
    st.pyplot(fig7)

    # Plotly chart: AI Adoption Rate by Sector
    st.subheader("AI Adoption Rate by Sector")
    fig8 = px.bar(filtered_df, x="Sector", y="AI_Adoption_Rate", color="Year", barmode="group")
    st.plotly_chart(fig8, use_container_width=True)

    # Economic Impact Analysis (Plotly)
    st.subheader("Economic Impact Over Time")
    fig9 = px.line(filtered_df, x="Year", y="EconomicImpact", markers=True)
    st.plotly_chart(fig9, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for AI Automation Analysis")
