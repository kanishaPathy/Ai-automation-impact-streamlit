import pandas as pd
import streamlit as st

# Sample data (replace with your actual dataset loading)
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Set default values based on the data
latest_year = sorted(df['_id.Year'].unique())[-1]  # Get the latest year
default_country = df[df['_id.Year'] == latest_year]['_id.Country'].mode()[0]  # Most common country in the latest year
default_sector = df[df['_id.Country'] == default_country]['_id.Sector'].mode()[0]  # Most common sector for the selected country

# Layout for user selections
st.sidebar.header("Select Parameters")

# Parameters for Year, Country, Sector, and Education Level
year = st.sidebar.selectbox("Year", sorted(df['_id.Year'].unique()), index=len(sorted(df['_id.Year'].unique()))-1)  # Default to latest year
country = st.sidebar.selectbox("Country", sorted(df['_id.Country'].unique()), index=sorted(df['_id.Country'].unique()).index(default_country))
sector = st.sidebar.selectbox("Sector", sorted(df['_id.Sector'].unique()), index=sorted(df['_id.Sector'].unique()).index(default_sector))
education_level = st.sidebar.selectbox("Education Level", sorted(df['Education_Level'].unique()))  # Assuming this column exists in your dataset

# Extract dynamic values for sliders based on current trends from the dataset
avg_pre_ai = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector) & (df['_id.Year'] == year)]['Avg_PreAI'].mean()
avg_post_ai = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector) & (df['_id.Year'] == year)]['Avg_PostAI'].mean()
avg_automation_impact = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector)]['Avg_Automation_Impact'].mean()
avg_ai_roles = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector)]['Avg_AI_Role_Jobs'].mean()
avg_reskill = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector)]['Avg_ReskillingPrograms'].mean()
avg_economic_impact = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector)]['Avg_EconomicImpact'].mean()
avg_sector_growth = df[(df['_id.Country'] == country) & (df['_id.Sector'] == sector)]['Avg_SectorGrowth'].mean()

# Display sliders with calculated values as default values
st.sidebar.header("Adjust Parameters Based on Trends")

pre_ai = st.sidebar.slider("Pre-AI Impact (%)", 0, 100, int(avg_pre_ai))  # Set based on current data
post_ai = st.sidebar.slider("Post-AI Impact (%)", 0, 100, int(avg_post_ai))  # Set based on current data
automation_impact = st.sidebar.slider("Automation Impact (%)", 0, 100, int(avg_automation_impact))
ai_roles = st.sidebar.slider("AI Role Jobs (%)", 0, 100, int(avg_ai_roles))
reskill = st.sidebar.slider("Reskilling Programs (%)", 0, 100, int(avg_reskill))
econ_impact = st.sidebar.slider("Economic Impact (%)", 0, 100, int(avg_economic_impact))
sector_growth = st.sidebar.slider("Sector Growth (%)", 0, 100, int(avg_sector_growth))

# Now you can display results based on these inputs
st.header("Analysis Results")

# You can use the sliders to perform predictions or generate insights
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year],
    'Avg_PreAI': [pre_ai],
    'Avg_PostAI': [post_ai],
    'Avg_Automation_Impact': [automation_impact],
    'Avg_AI_Role_Jobs': [ai_roles],
    'Avg_ReskillingPrograms': [reskill],
    'Avg_EconomicImpact': [econ_impact],
    'Avg_SectorGrowth': [sector_growth]
})

# Prediction or analysis based on input_df
prediction = model.predict(input_df)  # Assuming you have your model loaded

# Show prediction results
st.write(f"Prediction for the year {year}, country {country}, sector {sector}:")
st.write(f"Predicted Impact: {prediction}")
