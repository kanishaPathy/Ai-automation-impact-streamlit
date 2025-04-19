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

# # Visualization 3: Automation Impact Level and Reskilling Demand
# st.subheader("Automation Impact and Reskilling Demand Over Time")
# fig3 = px.line(filtered_df, x="Year", 
#                y=["Automation_Impact_Level", "Reskilling_Demand", "Avg_ReskillingPrograms"],
#                title="Automation Impact vs Reskilling Efforts")
# st.plotly_chart(fig3)
# Visualization 3: Automation Impact Level and Reskilling Demand
st.subheader("Automation Impact and Reskilling Demand Over Time")

# Define the relevant columns
reskilling_cols = ["Automation_Impact_Level", "Reskilling_Demand", "Avg_ReskillingPrograms"]

# Check if all required columns exist in the filtered DataFrame
missing_cols = [col for col in reskilling_cols if col not in filtered_df.columns]
if missing_cols:
    st.warning(f"The following columns are missing and required for this plot: {', '.join(missing_cols)}")
else:
    # Drop rows with missing values in those columns
    valid_df = filtered_df.dropna(subset=reskilling_cols)

    if not valid_df.empty:
        fig3 = px.line(valid_df, x="Year", 
                       y=reskilling_cols,
                       title="Automation Impact vs Reskilling Efforts")
        st.plotly_chart(fig3)
    else:
        st.warning("No valid data available to plot Automation Impact and Reskilling Demand.")


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
