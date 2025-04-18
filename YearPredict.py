import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load training data
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Create label encoders
le_country = LabelEncoder()
le_sector = LabelEncoder()
le_edu = LabelEncoder()

# Fit encoders
df['Country_enc'] = le_country.fit_transform(df['Country'])
df['Sector_enc'] = le_sector.fit_transform(df['Sector'])
df['Education_enc'] = le_edu.fit_transform(df['EducationLevel'])

# Features used in model
features = ['Country_enc', 'Sector_enc', 'Education_enc', 'Year']

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# Streamlit UI
st.title("📈 Predict Automation Impact")

st.markdown("### Provide input below to predict automation impact level")

# Dropdowns populated from actual data
country = st.selectbox("Select Country", sorted(df['Country'].unique()))
sector = st.selectbox("Select Sector", sorted(df['Sector'].unique()))
education = st.selectbox("Select Education Level", sorted(df['EducationLevel'].unique()))
year = st.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()), 2024)

if st.button("Predict Impact"):
    try:
        # Encode user inputs
        input_data = {
            "Country_enc": le_country.transform([country])[0],
            "Sector_enc": le_sector.transform([sector])[0],
            "Education_enc": le_edu.transform([education])[0],
            "Year": year
        }

        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]

        st.success(f"📊 Predicted Automation Impact Level: **{prediction:.2f}**")

    except Exception as e:
        st.error("❌ Error during prediction.")
        st.exception(e)
