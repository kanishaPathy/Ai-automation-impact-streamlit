import pandas as pd
import plotly.express as px

df1 = pd.read_csv("Unemployment_cleaned_df1.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned_df2.csv")
df3 = pd.read_csv("Sector_cleaned_df3.csv")

# 🔹 Step 2: Merge the datasets
# Merge df1 and df2 on 'Country', 'Sector', 'Year', and 'EducationLevel'
merged = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")

# Merge the above result with df3 on 'Country', 'Sector', and 'Year'
merged = merged.merge(df3, on=["Country", "Sector", "Year"], how="left")

# 🔹 Step 3: Check merged data
print(f"Merged data shape: {merged.shape}")
print(f"Columns in merged data: {merged.columns}")

# 🔹 Step 4: Ensure there are no NaN or missing values in the columns you're plotting
merged = merged.dropna(subset=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"])

# 🔹 Step 5: Verify that 'y' columns exist and have the same length
print(f"Avg_Automation_Impact column length: {len(merged['Avg_Automation_Impact'])}")
print(f"Automation_Impact_Level column length: {len(merged['Automation_Impact_Level'])}")
print(f"Predicted_Impact column length: {len(merged['Predicted_Impact'])}")

# 🔹 Step 6: Visualize data using Plotly (adjusting if necessary)
fig = px.bar(
    merged,
    x="Country",
    y=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"],
    barmode="group",
    title="AI & Automation Impact by Country",
    labels={
        "Country": "Country", 
        "Avg_Automation_Impact": "Avg Automation Impact", 
        "Automation_Impact_Level": "Automation Impact Level", 
        "Predicted_Impact": "Predicted Impact"
    }
)

# Show the plot
fig.show()

print("✅ Data merged and visualized successfully.")
