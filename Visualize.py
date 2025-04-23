import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import plotly.express as px


# Load model and encoders
model = joblib.load("xgb_model_final.pkl")
label_encoders = joblib.load("label_encoders1.pkl")

# Load new dataset
df = pd.read_csv("FINAL__Compressed_Dataset.csv.gz")

# Title and setup
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# User Inputs
st.markdown("### 🎯 Select Parameters for Prediction")

col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2015, 2024))
country = col2.selectbox("Select Country", sorted(df["Country"].unique()))
sector = col3.selectbox("Select Sector", sorted(df["Sector"].unique()))
education = col4.selectbox("Select Education Level", sorted(df["EducationLevel"].unique()))

# Filter data
filtered_df = df[
    (df["Country"] == country) &
    (df["Sector"] == sector) &
    (df["EducationLevel"] == education) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# Prediction block
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
input_encoded = input_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]
st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# --- TABS SECTION ---
tab1, tab2, tab3, tab4 = st.tabs(["📉 Unemployment & Automation", "📊 Skills & Gender", "🚀 Tech & Sector Growth", "🌐 Country Comparison"])

# --- TAB 1 ---
with tab1:
    st.subheader("Unemployment Impact Before vs After AI")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PreAI", label="Pre-AI", marker="o", ax=ax1)
    sns.lineplot(data=filtered_df, x="Year", y="Avg_PostAI", label="Post-AI", marker="o", ax=ax1)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.subheader("AI vs Automation Impact Over Time")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    x = range(len(filtered_df["Year"]))
    bar_width = 0.35
    ax2.bar(x, filtered_df["Avg_Automation_Impact"], width=bar_width, label="Automation")
    ax2.bar([i + bar_width for i in x], filtered_df["Avg_AI_Role_Jobs"], width=bar_width, label="AI Role Jobs")
    ax2.set_xticks([i + bar_width / 2 for i in x])
    ax2.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    ax2.legend()
    st.pyplot(fig2)

# --- TAB 2 ---
with tab2:
    st.subheader("Reskilling & Upskilling Programs Trend")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Reskilling_Demand", label="Reskilling Demand", marker="o", ax=ax3)
    sns.lineplot(data=filtered_df, x="Year", y="Upskilling_Programs", label="Upskilling Programs", marker="o", ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    st.subheader("Gender Distribution in Employment (%)")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    bar_width = 0.4
    x = range(len(filtered_df["Year"]))
    ax4.bar(x, filtered_df["Male_Percentage"], width=bar_width, label="Male")
    ax4.bar([i + bar_width for i in x], filtered_df["Female_Percentage"], width=bar_width, label="Female")
    ax4.set_xticks([i + bar_width / 2 for i in x])
    ax4.set_xticklabels(filtered_df["Year"].astype(str), rotation=45)
    ax4.legend()
    st.pyplot(fig4)

# --- TAB 3 ---
with tab3:
    st.subheader("Tech Investment vs AI Adoption Rate")
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=filtered_df, x="Year", y="Tech_Investment", label="Tech Investment", marker="o", ax=ax5)
    sns.lineplot(data=filtered_df, x="Year", y="AI_Adoption_Rate", label="AI Adoption Rate", marker="o", ax=ax5)
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

    st.subheader("Sector Growth/Decline Over Time")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    filtered_df["Year"] = filtered_df["Year"].astype(str)
    sns.barplot(data=filtered_df, x="Year", y="Sector_Growth_Decline", palette="coolwarm", ax=ax6)
    ax6.set_xticklabels(filtered_df["Year"].unique(), rotation=45, ha="right")
    fig6.tight_layout()
    st.pyplot(fig6)
    
    # Avg Automation Impact by Sector
    bar_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Sector:N", sort='-y'),
        y="Avg_Automation_Impact:Q",
        color="Sector:N",
        tooltip=["Sector", "Avg_Automation_Impact"]
    ).properties(title="Automation Impact by Sector").interactive()
    st.altair_chart(bar_chart, use_container_width=True)
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


# --- TAB 4 ---
with tab4:
    st.subheader("📊 Country vs Selected Sectors Comparison")
    selected_country = st.selectbox("Select Country", sorted(df["Country"].unique()), key="country_tab")
    selected_sectors = st.multiselect("Select Sectors", sorted(df["Sector"].unique()), default=list(df["Sector"].unique()[:2]), key="sector_tab")
    comparison_df = df[(df["Country"] == selected_country) & (df["Sector"].isin(selected_sectors))]

    if comparison_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader(f"AI Adoption Rate over Years in {selected_country}")
        fig_ai, ax_ai = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=comparison_df, x="Year", y="AI_Adoption_Rate", hue="Sector", marker="o", ax=ax_ai)
        ax_ai.set_xticks(sorted(comparison_df["Year"].unique()))
        ax_ai.tick_params(axis='x', rotation=45)
        st.pyplot(fig_ai)

        st.subheader(f"Automation Level over Years in {selected_country}")
        fig_auto, ax_auto = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=comparison_df, x="Year", y="Automation_Level", hue="Sector", marker="o", ax=ax_auto)
        ax_auto.set_xticks(sorted(comparison_df["Year"].unique()))
        ax_auto.tick_params(axis='x', rotation=45)
        st.pyplot(fig_auto)
        
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

# --- TAB 5 ---
with tab5:
    # --- Country Comparison ---
    st.subheader("🌍 Country Comparison from 2010 to 2022")
    selected_country1 = st.selectbox("Select First Country", sorted(df["Country"].unique()), key="country1")
    selected_country2 = st.selectbox("Select Second Country", sorted(df["Country"].unique()), index=1, key="country2")
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
