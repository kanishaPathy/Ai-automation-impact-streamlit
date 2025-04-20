import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
@st.cache
def load_data():
   df1 = pd.read_csv("Unemployment_cleaned_df1.csv")
   df2 = pd.read_csv("reskilling_dataset_cleaned_df2.csv")
   df3 = pd.read_csv("Sector_cleaned_df3.csv")
    return df1, df2, df3

df1, df2, df3 = load_data()

# Rename columns to match
df1 = df1.rename(columns={"_id.Country": "Country", "_id.Sector": "Sector", "_id.Year": "Year", "_id.EducationLevel": "EducationLevel"})

# Merge the datasets
merged = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")
merged = merged.merge(df3, on=["Country", "Sector", "Year"], how="left")

# Check if there are any missing values in critical columns
missing_values = merged[['Avg_Automation_Impact', 'Avg_PreAI', 'Avg_PostAI', 'Avg_AI_Role_Jobs', 'Avg_ReskillingPrograms', 'Avg_EconomicImpact', 'Avg_SectorGrowth']].isnull().sum()
st.write(f"Missing values in important columns:\n{missing_values}")

# Drop rows with missing target value (Avg_Automation_Impact)
merged = merged.dropna(subset=["Avg_Automation_Impact"])

# Ensure the target and features are aligned
X = merged[['Year', 'Sector', 'EducationLevel', 'Avg_PreAI', 'Avg_PostAI', 'Avg_AI_Role_Jobs', 'Avg_ReskillingPrograms', 'Avg_EconomicImpact', 'Avg_SectorGrowth']]
y = merged['Avg_Automation_Impact']

# Handle categorical features by one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Ensure that X and y have the same length after dropping NaN values
st.write(f"Shape of features (X): {X.shape}")
st.write(f"Shape of target (y): {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Add predictions to the merged dataset
merged['Predicted_Impact'] = model.predict(pd.get_dummies(merged[['Year', 'Sector', 'EducationLevel', 'Avg_PreAI', 'Avg_PostAI', 'Avg_AI_Role_Jobs', 'Avg_ReskillingPrograms', 'Avg_EconomicImpact', 'Avg_SectorGrowth']], drop_first=True))

# Visualization section
st.title("AI and Automation Impact Visualization")

# Choose the type of plot to display
plot_option = st.selectbox("Select Visualization Type", ["Actual vs Predicted", "Automation Impact Over Time"])

if plot_option == "Actual vs Predicted":
    # Plot actual vs predicted impact
    fig = px.scatter(
        merged,
        x="Avg_Automation_Impact",
        y="Predicted_Impact",
        color="Sector",
        title="Actual vs Predicted Automation Impact",
        labels={"Avg_Automation_Impact": "Actual Automation Impact", "Predicted_Impact": "Predicted Automation Impact"}
    )
    st.plotly_chart(fig)

elif plot_option == "Automation Impact Over Time":
    # Plot automation impact over time (use a suitable aggregation like mean or sum)
    fig2 = px.line(
        merged,
        x="Year",
        y="Avg_Automation_Impact",
        color="Sector",
        title="Automation Impact Over Time",
        labels={"Avg_Automation_Impact": "Automation Impact"}
    )
    st.plotly_chart(fig2)

# Optional: Let the user download the data with predictions
csv = merged.to_csv(index=False)
st.download_button("Download Data with Predictions", csv, "predictions.csv", "text/csv")
