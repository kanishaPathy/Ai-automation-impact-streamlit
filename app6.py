import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load("xgboost_model.pkl")

# Load your data (this would ideally be from your database or CSV file)
# You can also load from a CSV, but in this case, we assume you already have a dataframe loaded
df = pd.read_csv("your_dataset.csv")  # Replace with your dataset

# Function to make prediction
def make_prediction(year, avg_pre_ai, avg_post_ai, avg_ai_role_jobs, avg_reskilling_programs, avg_economic_impact, avg_sector_growth,
                     country, sector, education_level):
    input_df = pd.DataFrame({
        "_id.Year": [year],
        "Avg_PreAI": [avg_pre_ai],
        "Avg_PostAI": [avg_post_ai],
        "Avg_AI_Role_Jobs": [avg_ai_role_jobs],
        "Avg_ReskillingPrograms": [avg_reskilling_programs],
        "Avg_EconomicImpact": [avg_economic_impact],
        "Avg_SectorGrowth": [avg_sector_growth],
        "_id.Country_Canada": [1 if country == 'Canada' else 0],
        "_id.Country_Germany": [1 if country == 'Germany' else 0],
        "_id.Country_India": [1 if country == 'India' else 0],
        "_id.Country_Ireland": [1 if country == 'Ireland' else 0],
        "_id.Country_USA": [1 if country == 'USA' else 0],
        "_id.Sector_Agriculture": [1 if sector == 'Agriculture' else 0],
        "_id.Sector_Education": [1 if sector == 'Education' else 0],
        "_id.Sector_Finance": [1 if sector == 'Finance' else 0],
        "_id.Sector_Healthcare": [1 if sector == 'Healthcare' else 0],
        "_id.Sector_IT": [1 if sector == 'IT' else 0],
        "_id.Sector_Manufacturing": [1 if sector == 'Manufacturing' else 0],
        "_id.Sector_Media & Entertainment": [1 if sector == 'Media & Entertainment' else 0],
        "_id.Sector_Retail": [1 if sector == 'Retail' else 0],
        "_id.Sector_Transportation": [1 if sector == 'Transportation' else 0],
        "_id.EducationLevel_Bachelor": [1 if education_level == 'Bachelor' else 0],
        "_id.EducationLevel_High School": [1 if education_level == 'High School' else 0],
        "_id.EducationLevel_Master": [1 if education_level == 'Master' else 0],
        "_id.EducationLevel_PhD": [1 if education_level == 'PhD' else 0],
        "_id.EducationLevel_Unknown": [1 if education_level == 'Unknown' else 0]
    })

    input_df = input_df[model.get_booster().feature_names]
    prediction = model.predict(input_df)
    return prediction

# Streamlit app interface
st.title("AI & Automation Impact Prediction")

# Create user inputs
year = st.number_input("Year", min_value=2010, max_value=2024, value=2022)
avg_pre_ai = st.number_input("Avg Pre-AI")
avg_post_ai = st.number_input("Avg Post-AI")
avg_ai_role_jobs = st.number_input("Avg AI Role Jobs")
avg_reskilling_programs = st.number_input("Avg Reskilling Programs")
avg_economic_impact = st.number_input("Avg Economic Impact")
avg_sector_growth = st.number_input("Avg Sector Growth")

# Select box for categorical inputs
country = st.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT', 'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.selectbox("Select Education Level", ['Bachelor', 'High School', 'Master', 'PhD', 'Unknown'])

# Display inputs
st.write(f"Selected Inputs - Year: {year}, Avg Pre-AI: {avg_pre_ai}, Avg Post-AI: {avg_post_ai}, Avg AI Role Jobs: {avg_ai_role_jobs}, "
         f"Avg Reskilling Programs: {avg_reskilling_programs}, Avg Economic Impact: {avg_economic_impact}, Avg Sector Growth: {avg_sector_growth}, "
         f"Country: {country}, Sector: {sector}, Education Level: {education_level}")

# Prediction Button
if st.button("Make Prediction"):
    prediction = make_prediction(year, avg_pre_ai, avg_post_ai, avg_ai_role_jobs, avg_reskilling_programs, avg_economic_impact, avg_sector_growth, country, sector, education_level)
    st.write(f"Prediction: {prediction[0]}")

# Display Data Visualizations
st.subheader("Visualizations")

# Line chart for trends over the years (Example)
df_grouped = df.groupby('Year').mean()  # You can adjust this depending on your dataset
st.line_chart(df_grouped['Avg_SectorGrowth'])  # Replace with the column you're interested in

# Bar plot for sector growth by country (Example)
sector_growth_by_country = df.groupby(['Country', 'Sector'])['Avg_SectorGrowth'].mean().unstack()
sector_growth_by_country.plot(kind='bar', figsize=(10, 6))
st.pyplot()

# Seaborn heatmap for correlations between features
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# Display DataFrame of the first few rows
st.subheader("Dataset Preview")
st.write(df.head())

