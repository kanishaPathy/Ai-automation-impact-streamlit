import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Comparison")
col1, col2 = st.columns(2)
year_range = col1.slider("Select Year Range", int(df1['_id.Year'].min()), int(df1['_id.Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df1['_id.Country'].unique()))

# Merge the two datasets on common columns (e.g., Year, Country, Sector)
merged_df = pd.merge(df1, df2, how='inner', on=['_id.Country', '_id.Sector', '_id.Year'])

# ---------- Comparison Visualizations ----------

# Sector-wise Comparison of Automation Impact (Avg_Automation_Impact vs Automation_Impact_Level)
st.markdown("---")
st.header("🏭 Sector-wise Automation Impact Comparison")

sector_selected = st.selectbox("Select Sector", merged_df['_id.Sector'].unique())
sector_df = merged_df[merged_df['_id.Sector'] == sector_selected]

fig1 = px.bar(sector_df, x='_id.Year', y=['Avg_Automation_Impact', 'Automation_Impact_Level'],
              title=f"Automation Impact Comparison in {sector_selected} (PreAI vs PostAI)")
st.plotly_chart(fig1, use_container_width=True)

# Skills Gap vs Automation Impact
st.markdown("---")
st.header("💼 Skills Gap vs Automation Impact")
fig2 = px.scatter(merged_df, x='Skills_Gap', y='Avg_Automation_Impact', color='_id.Sector',
                  title="Skills Gap vs Automation Impact Across Sectors")
st.plotly_chart(fig2, use_container_width=True)

# Gender Distribution vs Automation Impact
st.markdown("---")
st.header("👩‍💻 Gender Distribution vs Automation Impact")

if 'Gender_Distribution' in df2.columns:
    gender_df = merged_df.groupby(['_id.Sector', 'Gender_Distribution']).agg({
        'Avg_Automation_Impact': 'mean'}).reset_index()

    fig3 = px.bar(gender_df, x='_id.Sector', y='Avg_Automation_Impact', color='Gender_Distribution',
                  title="Gender Distribution and Automation Impact")
    st.plotly_chart(fig3, use_container_width=True)

# Reskilling Demand vs Automation Impact
st.markdown("---")
st.header("📚 Reskilling Demand vs Automation Impact")

fig4 = px.scatter(merged_df, x='Reskilling_Demand', y='Avg_Automation_Impact', color='_id.Sector',
                  title="Reskilling Demand vs Automation Impact Across Sectors")
st.plotly_chart(fig4, use_container_width=True)

# Skill Level vs Automation Impact
st.markdown("---")
st.header("🧑‍💻 Skill Level vs Automation Impact")

fig5 = px.box(merged_df, x='Skill_Level', y='Avg_Automation_Impact', color='_id.Sector',
              title="Skill Level vs Automation Impact Across Sectors")
st.plotly_chart(fig5, use_container_width=True)

# ---------- Export Prediction Option ----------
st.markdown("---")
if st.button("💾 Save Comparison to CSV"):
    merged_df.to_csv("merged_comparison.csv", index=False)
    st.success("📁 Data saved to **merged_comparison.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by [Your Name] | Powered by Streamlit + Plotly")
