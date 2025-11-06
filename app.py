
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("/Users/saikiranchevula/Desktop/farm_biogas-prediction/Castboost.joblib")

st.set_page_config(page_title="Farm Biogas Prediction", page_icon="🌱", layout="centered")
st.title("🌾 Farm Biogas Prediction App")
st.markdown("Predict **Biogas Generation (cu-ft/day)** based on project & farm data.")

# -------------------------------
# Input Fields
# -------------------------------
st.header("Enter Farm Project Details")

# Numeric Inputs
year_operational = st.number_input("Year Operational", 2000, 2025, 2022)
dairy = st.number_input("Dairy", 0, 50000, 10000)
electricity_generated = st.number_input("Electricity Generated (kWh/yr)", 0.0, 10000000.0, 500000.0)
total_emission_reductions = st.number_input("Total Emission Reductions (MTCO2e/yr)", 0.0, 100000.0, 10000.0)
operational_years = st.number_input("Operational Years", 0.0, 50.0, 5.0)
biogas_per_animal = st.number_input("Biogas per Animal (cu-ft/day)", 0.0, 500.0, 50.0)
emission_reduction_per_year = st.number_input("Emission Reduction per Year", 0.0, 100000.0, 5000.0)
electricity_to_biogas_ratio = st.number_input("Electricity to Biogas Ratio", 0.0, 50.0, 5.0)
total_waste = st.number_input("Total Waste (kg/day)", 0.0, 10000000.0, 100000.0)
waste_efficiency = st.number_input("Waste Efficiency", 0.0, 10.0, 1.0)
electricity_efficiency = st.number_input("Electricity Efficiency", 0.0, 10.0, 1.0)

# -------------------------------
# Categorical Inputs (match your dummies)
# -------------------------------
project_type = st.selectbox("Project Type", [
    "Farm Scale", "Multiple Farm/Facility", "Research"
])

digester_type = st.selectbox("Digester Type", [
    "Complete Mix", "Complete Mix Mini Digester", "Covered Lagoon",
    "Fixed Film/Attached Media", "Horizontal Plug Flow", "Induced Blanket Reactor",
    "Mixed Plug Flow", "Modular Plug Flow", "No-Info",
    "Plug Flow - Unspecified", "Unknown or Unspecified", "Vertical Plug Flow"
])

status = st.selectbox("Status", ["Operational", "Shut down"])

animal_type = st.selectbox("Animal/Farm Type(s)", [
    "Dairy", "Dairy; Poultry; Swine", "Dairy; Swine"
])

co_digestion = st.selectbox("Co-Digestion", ["Yes", "No"])

biogas_end_use = st.selectbox("Biogas End Use(s)", [
    "Boiler/Furnace fuel; CNG", "CNG", "Cogeneration",
    "Cogeneration; Boiler/Furnace fuel", "Cogeneration; CNG",
    "Cogeneration; Pipeline Gas", "Cogeneration; Refrigeration",
    "Electricity", "Electricity; Boiler/Furnace fuel", "Electricity; CNG",
    "Electricity; Cogeneration", "Electricity; Cogeneration; Boiler/Furnace fuel",
    "Electricity; Pipeline Gas", "Flared Full-time", "No-Info",
    "Pipeline Gas", "Pipeline to Electricity"
])

lcfs_pathway = st.selectbox("LCFS Pathway?", ["Yes", "No"])
receiving_utility = st.selectbox("Receiving Utility", ["Yes", "No"])
usda_funding = st.selectbox("Awarded USDA Funding?", ["Yes", "No"])

# -------------------------------
# Build Input Data
# -------------------------------
numeric_data = {
    'Year Operational': year_operational,
    'Dairy': dairy,
    'Electricity Generated (kWh/yr)': electricity_generated,
    'Total Emission Reductions (MTCO2e/yr)': total_emission_reductions,
    'Operational Years': operational_years,
    'Biogas_per_Animal (cu-ft/day)': biogas_per_animal,
    'Emission_Reduction_per_Year': emission_reduction_per_year,
    'Electricity_to_Biogas_Ratio': electricity_to_biogas_ratio,
    'Total_Waste_kg/day': total_waste,
    'Waste_Efficiency': waste_efficiency,
    'Electricity_Efficiency': electricity_efficiency
}

categorical_data = {
    'Project Type': project_type,
    'Digester Type': digester_type,
    'Status': status,
    'Animal/Farm Type(s)': animal_type,
    'Co-Digestion': co_digestion,
    'Biogas End Use(s)': biogas_end_use,
    'LCFS Pathway?': lcfs_pathway,
    'Receiving Utility': receiving_utility,
    'Awarded USDA Funding?': usda_funding
}

# Combine both
input_df = pd.DataFrame([{**numeric_data, **categorical_data}])

# -------------------------------
# One-hot encode to match model training
# -------------------------------
input_encoded = pd.get_dummies(input_df)

# Add missing columns (fill with 0)
expected_order = [
    'Year Operational', 'Dairy', 'Electricity Generated (kWh/yr)', 'Total Emission Reductions (MTCO2e/yr)',
    'Operational Years', 'Biogas_per_Animal (cu-ft/day)', 'Emission_Reduction_per_Year',
    'Electricity_to_Biogas_Ratio', 'Total_Waste_kg/day', 'Waste_Efficiency', 'Electricity_Efficiency',
    'Project Type_Farm Scale', 'Project Type_Multiple Farm/Facility', 'Project Type_Research',
    'Digester Type_Complete Mix', 'Digester Type_Complete Mix Mini Digester', 'Digester Type_Covered Lagoon',
    'Digester Type_Fixed Film/Attached Media', 'Digester Type_Horizontal Plug Flow', 'Digester Type_Induced Blanket Reactor',
    'Digester Type_Mixed Plug Flow', 'Digester Type_Modular Plug Flow', 'Digester Type_No-Info',
    'Digester Type_Plug Flow - Unspecified', 'Digester Type_Unknown or Unspecified', 'Digester Type_Vertical Plug Flow',
    'Status_Operational', 'Status_Shut down', 'Animal/Farm Type(s)_Dairy', 'Animal/Farm Type(s)_Dairy; Poultry; Swine',
    'Animal/Farm Type(s)_Dairy; Swine', 'Co-Digestion_Yes', 'Biogas End Use(s)_Boiler/Furnace fuel; CNG',
    'Biogas End Use(s)_CNG', 'Biogas End Use(s)_Cogeneration', 'Biogas End Use(s)_Cogeneration; Boiler/Furnace fuel',
    'Biogas End Use(s)_Cogeneration; CNG', 'Biogas End Use(s)_Cogeneration; Pipeline Gas', 'Biogas End Use(s)_Cogeneration; Refrigeration',
    'Biogas End Use(s)_Electricity', 'Biogas End Use(s)_Electricity; Boiler/Furnace fuel', 'Biogas End Use(s)_Electricity; CNG',
    'Biogas End Use(s)_Electricity; Cogeneration', 'Biogas End Use(s)_Electricity; Cogeneration; Boiler/Furnace fuel',
    'Biogas End Use(s)_Electricity; Pipeline Gas', 'Biogas End Use(s)_Flared Full-time', 'Biogas End Use(s)_No-Info',
    'Biogas End Use(s)_Pipeline Gas', 'Biogas End Use(s)_Pipeline to Electricity', 'LCFS Pathway?_Yes',
    'Receiving Utility_Yes', 'Awarded USDA Funding?_Yes'
]

for col in expected_order:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns exactly
input_encoded = input_encoded[expected_order]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Biogas Generation"):
    try:
        pred = model.predict(input_encoded)[0]
        st.success(f"### 🌿 Estimated Biogas Generation: **{pred:,.2f} cu-ft/day**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

st.markdown("---")
st.caption("Developed with 💚 using Streamlit + CatBoost | Farm Biogas ML 2025")
