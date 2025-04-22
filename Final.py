import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

# Load model and encoders
model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders1.pkl")

# Load new dataset
df = pd.read_csv("FINAL__Compressed_Dataset.csv.gz")

# ... Keep all import and initial setup unchanged ...

# App title
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")

# Create 4 side-by-side columns
col1, col2, col3, col4 = st.columns(4)

# Inline controls instead of sidebar
year_range = col1.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2015, 2024)
)
country = col2.selectbox("Select Country", sorted(df["Country"].unique()))
sector = col3.selectbox("Select Sector", sorted(df["Sector"].unique()))
education = col4.selectbox("Select Education Level", sorted(df["EducationLevel"].unique()))

# # Filtered data for visualizations
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]
# Prepare input data for prediction (using first year in range)
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

# Encoding and preprocessing (same as in your code)
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

# Align the input data with the model's feature names
input_encoded = input_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)

# Make prediction with the aligned data
with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# --- Unemployment Impact Before vs After AI ---
st.subheader("Unemployment Impact Before vs After AI")

# Create three columns and put the plot in the center one
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    fig1, ax1 = plt.subplots(figsize=(6, 3))  # Moderate size
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
    ax1.set_title("Unemployment Impact Before vs After AI", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

# --- AI vs Automation Impact ---
st.subheader("AI vs Automation Impact")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bar_width = 0.35
    years = filtered_df["Year"].astype(str)
    x = range(len(years))
    ax2.bar(x, filtered_df["Avg_Automation_Impact"], width=bar_width, label="Automation")
    ax2.bar([i + bar_width for i in x], filtered_df["Avg_AI_Role_Jobs"], width=bar_width, label="AI Role Jobs")
    ax2.set_xticks([i + bar_width / 2 for i in x])
    ax2.set_xticklabels(years, rotation=45)
    ax2.legend()
    st.pyplot(fig2)

# # --- Prediction Button ---
# if st.button("Predict Future Impact"):
#     input_df = pd.DataFrame({
#         'Country': [country],
#         'Sector': [sector],
#         'Year': [year_range[0]],
#         'EducationLevel': [education]
#     })

#     additional_features = [
#         'Avg_PreAI', 'Avg_PostAI', 'Avg_Automation_Impact', 'Avg_AI_Role_Jobs',
#         'Avg_ReskillingPrograms', 'Avg_EconomicImpact', 'Skill_Level', 'Skills_Gap',
#         'Reskilling_Demand', 'Upskilling_Programs', 'Automation_Impact_Level',
#         'Revenue', 'Growth_Rate', 'AI_Adoption_Rate', 'Automation_Level',
#         'Sector_Impact_Score', 'Tech_Investment', 'Sector_Growth_Decline',
#         'Male_Percentage', 'Female_Percentage'
#     ]

#     # Fill in means/modes for numeric and categorical features
#     for col in additional_features:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             input_df[col] = df[col].mean()  # Fill with mean for numeric columns
#         else:
#             input_df[col] = df[col].mode()[0]  # Fill with mode for categorical columns

#     # Initialize OneHotEncoder
#     ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Updated argument

#     # Define categorical columns
#     categorical_cols = ['Country', 'Sector', 'EducationLevel', 'Skill_Level', 'Automation_Impact_Level', 'AI_Adoption_Rate', 'Automation_Level', 'Sector_Growth_Decline']

#     # Apply OneHotEncoder for categorical columns
#     for col in categorical_cols:
#         if col in input_df.columns:
#             ohe.fit(df[[col]])  # Fit on the entire original data to capture all categories
#             encoded = ohe.transform(input_df[[col]])  # Transform the input data
            
#             # Convert the output to a DataFrame for easier concatenation
#             encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))
            
#             # Concatenate the new encoded columns with the input_df
#             input_df = pd.concat([input_df, encoded_df], axis=1)
#             input_df = input_df.drop(columns=[col])  # Drop the original column

#     # Debugging Step: Check if the encoded columns exist in input_df
#     st.write(f"Columns after encoding: {input_df.columns.tolist()}")

#     # Optional: reorder columns if model requires specific order
#     if hasattr(model, 'feature_names_in_'):
#         missing_columns = [col for col in model.feature_names_in_ if col not in input_df.columns]
#         if missing_columns:
#             st.error(f"Missing columns in input_df: {missing_columns}")
#             st.stop()
#         input_df = input_df[model.feature_names_in_]

#     # Predict
#     prediction = model.predict(input_df)[0]
#     st.success(f"Predicted Impact Score for {year_range[0]}: {prediction:.2f}")



#Reskilling
st.subheader("Reskilling & Upskilling Programs Trend")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Reskilling_Demand", label="Reskilling Demand", marker="o", ax=ax3)
    sns.lineplot(data=filtered_df, x="Year", y="Upskilling_Programs", label="Upskilling Programs", marker="o", ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

# --- Gender ---
st.subheader("Gender Distribution in Employment (%)")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    bar_width = 0.4
    x = range(len(filtered_df["Year"]))
    ax4.bar(x, filtered_df["Male_Percentage"], width=bar_width, label="Male")
    ax4.bar([i + bar_width for i in x], filtered_df["Female_Percentage"], width=bar_width, label="Female")
    ax4.set_xticks([i + bar_width / 2 for i in x])
    ax4.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    ax4.legend()
    st.pyplot(fig4)

# --- Tech Investment vs AI Adoption ---
st.subheader("Tech Investment vs AI Adoption Rate")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Tech_Investment", label="Tech Investment", marker="o", ax=ax5)
    sns.lineplot(data=filtered_df, x="Year", y="AI_Adoption_Rate", label="AI Adoption Rate", marker="o", ax=ax5)
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

# --- Sector Growth ---
st.subheader("Sector Growth/Decline Over Time")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=filtered_df, x="Year", y="Sector_Growth_Decline", palette="coolwarm", ax=ax6)
    ax6.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    st.pyplot(fig6)

# --- Automation Level ---
st.subheader("Automation Level by Year")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig7, ax7 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Automation_Level", marker="o", ax=ax7)
    ax7.tick_params(axis='x', rotation=45)
    st.pyplot(fig7)

# --- Plotly Chart: Unemployment vs Skills Gap ---
st.subheader("Unemployment Impact vs Skills Gap")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig8 = px.line(
        filtered_df,
        x="Year",
        y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
        labels={"value": "Impact/Gap"},
        title="AI's Impact on Unemployment and Skills Gap"
    )
    st.plotly_chart(fig8, use_container_width=True)

# --- Plotly Chart: AI Adoption vs Sector Growth ---
st.subheader("AI Adoption vs Sector Growth")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig9 = px.bar(
        filtered_df,
        x="Year",
        y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
        barmode="group",
        title="AI Adoption Rate vs Sector Growth Decline"
    )
    st.plotly_chart(fig9, use_container_width=True)

# --- Country vs Selected Sectors ---
st.header("📊 Country vs Selected Sectors Comparison")
selected_country = st.selectbox("Select Country", sorted(df["Country"].unique()), key="country_sector_view")
available_sectors = df["Sector"].unique()
selected_sectors = st.multiselect("Select Sectors", sorted(available_sectors), default=list(available_sectors[:2]), key="sector_multi")
comparison_df = df[(df["Country"] == selected_country) & (df["Sector"].isin(selected_sectors))]

if comparison_df.empty:
    st.warning("No data available for the selected filters.")
else:
    st.subheader(f"AI Adoption Rate over Years in {selected_country}")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        fig_sector_ai, ax_ai = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=comparison_df, x="Year", y="AI_Adoption_Rate", hue="Sector", marker="o", ax=ax_ai)
        ax_ai.set_ylabel("AI Adoption Rate")
        ax_ai.set_xticks(sorted(comparison_df["Year"].unique()))
        ax_ai.tick_params(axis='x', rotation=45)
        fig_sector_ai.tight_layout()
        st.pyplot(fig_sector_ai)

    st.subheader(f"Automation Level over Years in {selected_country}")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        fig_sector_auto, ax_auto = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=comparison_df, x="Year", y="Automation_Level", hue="Sector", marker="o", ax=ax_auto)
        ax_auto.set_ylabel("Automation Level")
        ax_auto.set_xticks(sorted(comparison_df["Year"].unique()))
        ax_auto.tick_params(axis='x', rotation=45)
        fig_sector_auto.tight_layout()
        st.pyplot(fig_sector_auto)

    st.subheader(f"Average Sector Impact Score in {selected_country}")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        avg_impact_df = comparison_df.groupby("Sector")["Sector_Impact_Score"].mean().reset_index()
        fig_impact, ax_impact = plt.subplots(figsize=(6, 2.5))
        sns.barplot(data=avg_impact_df, x="Sector", y="Sector_Impact_Score", palette="viridis", ax=ax_impact)
        ax_impact.set_ylabel("Avg Impact Score")
        ax_impact.set_title("Sector-Wise Avg Impact")
        fig_impact.tight_layout()
        st.pyplot(fig_impact)

# --- Country Comparison ---
st.header("🌍 Country Comparison from 2010 to 2022")
country1 = st.selectbox("Select First Country", sorted(df["Country"].unique()), key="country1")
country2 = st.selectbox("Select Second Country", sorted(df["Country"].unique()), index=1, key="country2")
country_df = df[df["Country"].isin([country1, country2]) & df["Year"].between(2010, 2022)]

if country_df.empty:
    st.warning("No data available for selected countries and years.")
else:
    melted_df = pd.melt(country_df, id_vars=["Year", "Country"], value_vars=["Avg_PreAI", "Avg_PostAI"], var_name="Type", value_name="Unemployment")
    melted_df["Type"] = melted_df["Type"].replace({"Avg_PreAI": "Pre-AI", "Avg_PostAI": "Post-AI"})

    st.subheader("Unemployment Impact (Pre-AI vs Post-AI)")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        fig_cmp, ax_cmp = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=melted_df, x="Year", y="Unemployment", hue="Type", style="Country", markers=True, dashes=False, ax=ax_cmp)
        ax_cmp.set_title("Country-wise Unemployment Trend (Pre-AI vs Post-AI)")
        ax_cmp.set_ylabel("Unemployment Rate")
        ax_cmp.tick_params(axis='x', rotation=45)
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp)

    st.subheader("AI Adoption Rate Comparison")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        fig_ai, ax_ai = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=country_df, x="Year", y="AI_Adoption_Rate", hue="Country", marker="o", ax=ax_ai)
        ax_ai.set_ylabel("AI Adoption Rate")
        ax_ai.set_title("AI Adoption Rate (2010-2022)")
        ax_ai.tick_params(axis='x', rotation=45)
        fig_ai.tight_layout()
        st.pyplot(fig_ai)

# --- Sector-wise ---
st.header("🏭 Sector-wise Unemployment Comparison")
selected_sector = st.selectbox("Select Sector", sorted(df["Sector"].unique()), key="sector_comp")
sector_year_range = st.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2010, 2022))
sector_df = df[(df["Sector"] == selected_sector) & (df["Year"].between(sector_year_range[0], sector_year_range[1]))]

if sector_df.empty:
    st.warning("No data found for selected sector and years.")
else:
    st.subheader(f"Unemployment in {selected_sector} from {sector_year_range[0]} to {sector_year_range[1]}")
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        fig_sector, ax_sector = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=sector_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax_sector)
        sns.lineplot(data=sector_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax_sector)
        ax_sector.set_ylabel("Unemployment Rate")
        ax_sector.set_title(f"{selected_sector} Sector Unemployment Trend")
        ax_sector.tick_params(axis='x', rotation=45)
        fig_sector.tight_layout()
        st.pyplot(fig_sector)

# --- Unemployment vs Skills Gap ---
st.subheader("Unemployment Impact vs Skills Gap")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig1 = px.line(
        filtered_df,
        x="Year",
        y=["Avg_PreAI", "Avg_PostAI", "Skills_Gap"],
        labels={"value": "Impact/Gap"},
        title="AI's Impact on Unemployment and Skills Gap"
    )
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)

# --- AI Adoption vs Sector Growth ---
st.subheader("AI Adoption vs Sector Growth")
left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    fig2 = px.bar(
        filtered_df,
        x="Year",
        y=["AI_Adoption_Rate", "Sector_Growth_Decline"],
        barmode="group",
        title="AI Adoption Rate vs Sector Growth Decline"
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)
