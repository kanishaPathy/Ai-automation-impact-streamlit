import pandas as pd
import plotly.express as px

# 🔹 Step 1: Load the cleaned datasets
df1 = pd.read_csv("Unemployment_cleaned_df1.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned_df2.csv")
df3 = pd.read_csv("Sector_cleaned_df3.csv")

# 🔹 Step 2: Merge the datasets
# Merge df1 and df2 on 'Country', 'Sector', 'Year', and 'EducationLevel'
merged = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")

# Merge the above result with df3 on 'Country', 'Sector', and 'Year'
merged = merged.merge(df3, on=["Country", "Sector", "Year"], how="left")

# 🔹 Step 3: Visualize data using Plotly
fig = px.bar(
    merged,
    x="Country",
    y=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"],
    barmode="group",
    title="AI & Automation Impact by Country",
    labels={"Country": "Country", "Avg_Automation_Impact": "Avg Automation Impact", "Automation_Impact_Level": "Automation Impact Level", "Predicted_Impact": "Predicted Impact"}
)

# Show the plot
fig.show()

print("✅ Data merged and visualized successfully.")
