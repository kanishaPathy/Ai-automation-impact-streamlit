import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load your merged data
@st.cache_data
def load_data():
    df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
    df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
    df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")
    merged_1_2 = pd.merge(df1, df2, left_on=['_id.Country', '_id.Sector', '_id.Year', '_id.EducationLevel'],
                          right_on=['Country', 'Sector', 'Year', 'Education_Level'], how='inner')
    merged_all = pd.merge(merged_1_2, df3, on=['Country', 'Sector', 'Year'], how='inner')
    return merged_all

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())
selected_sector = st.sidebar.selectbox("Select Sector", df["Sector"].unique())
selected_year_range = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2010, 2024))

# Filtered Data
filtered_df = df[(df["Country"] == selected_country) & 
                 (df["Sector"] == selected_sector) & 
                 (df["Year"] >= selected_year_range[0]) & 
                 (df["Year"] <= selected_year_range[1])]

# Visualization 1: Unemployment vs Skills Gap
st.subheader("Unemployment Impact vs Skills Gap")
fig1 = px.line(filtered_df, x="Year", 
               y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
               labels={"value": "Impact/Gap"}, 
               title="AI's Impact on Unemployment and Skills Gap")
st.plotly_chart(fig1)

# Visualization 2: AI Adoption vs Sector Growth
st.subheader("AI Adoption vs Sector Growth")
fig2 = px.bar(filtered_df, x="Year", y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
              barmode="group", title="AI Adoption Rate vs Sector Growth Decline")
st.plotly_chart(fig2)
# Check the shape of filtered_df before transformation
st.write("Filtered Data:", filtered_df.head())  # Display first few rows
st.write("Number of rows in filtered data:", filtered_df.shape[0])

# Check the shape of filtered_df before transformation
st.write("Filtered Data Preview:", filtered_df.head())  # Display first few rows
st.write("Number of rows in filtered data:", filtered_df.shape[0])

# Visualization 3: Automation Impact Level and Reskilling Demand

st.subheader("Automation Impact and Reskilling Demand Over Time")

# Convert Automation_Impact_Level from categorical to numeric (LOW: 0, MEDIUM: 1, HIGH: 2)
impact_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
filtered_df["Automation_Impact_Level"] = filtered_df["Automation_Impact_Level"].map(impact_mapping)

# Ensure other columns are numeric, using coercion to handle any non-numeric values
filtered_df["Reskilling_Demand"] = pd.to_numeric(filtered_df["Reskilling_Demand"], errors="coerce")
filtered_df["Avg_ReskillingPrograms"] = pd.to_numeric(filtered_df["Avg_ReskillingPrograms"], errors="coerce")

# Drop rows with any NaN values in relevant columns
valid_df = filtered_df.dropna(subset=["Year", "Automation_Impact_Level", "Reskilling_Demand", "Avg_ReskillingPrograms"])

# Ensure there is valid data to plot
if not valid_df.empty:
    # Create the line plot
    fig3 = px.line(valid_df, x="Year", 
                   y=["Automation_Impact_Level", "Reskilling_Demand", "Avg_ReskillingPrograms"],
                   title="Automation Impact vs Reskilling Efforts")
    st.plotly_chart(fig3)
else:
    st.warning("No valid data available to plot.")



# Load prediction model
try:
    model = joblib.load("xgboost_model.pkl")

    if st.button("Predict AI Impact (Post-AI Unemployment / Skills Gap)"):
        if not filtered_df.empty:
            # Select relevant features for prediction — modify this list based on your model
            features = filtered_df[['Avg_PreAI', 'Avg_Automation_Impact', 'Avg_ReskillingPrograms']].dropna()
            if not features.empty:
                predictions = model.predict(features)
                filtered_df.loc[features.index, 'Predicted_PostAI'] = predictions

                st.subheader("🔮 Predicted Post-AI Unemployment")
                st.dataframe(filtered_df[['Country', 'Sector', 'Year', 'Avg_PreAI', 'Predicted_PostAI']].dropna())

                if 'Avg_PostAI' in filtered_df.columns:
                    fig_pred = px.line(filtered_df.dropna(subset=['Predicted_PostAI']), x='Year',
                                       y=['Avg_PostAI', 'Predicted_PostAI'], color='Country',
                                       title='Actual vs Predicted Post-AI Unemployment')
                    st.plotly_chart(fig_pred)
            else:
                st.warning("Not enough data available for prediction.")

except FileNotFoundError:
    st.warning("Model file not found. Please upload 'model.pkl' to use prediction features.")
