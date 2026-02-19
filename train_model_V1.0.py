import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Region mapping (expand as needed from your original data)
region_map = {
    'Afghanistan': 'Asia', 'Albania': 'Europe', 'Algeria': 'Africa', 'Angola': 'Africa',
    'Argentina': 'Americas', 'Armenia': 'Asia', 'Aruba': 'Americas', 'Australia': 'Oceania',
    'Austria': 'Europe', 'Azerbaijan': 'Asia', 'Bahamas': 'Americas', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Barbados': 'Americas', 'Belarus': 'Europe', 'Belgium': 'Europe',
    'Belize': 'Americas', 'United Arab Emirates': 'Asia', 'United States': 'Americas',
    'China': 'Asia', 'India': 'Asia', 'Saudi Arabia': 'Asia', 'Germany': 'Europe'
    # Add more: df_old.groupby('Country')['Region'].first().to_dict()
}

# Load & process new World Bank data
df_new = pd.read_csv("2c068aa1-30dc-45b2-bd84-6730351ffa25_Data.csv")
co2_cols = [col for col in df_new.columns if col.startswith(tuple(str(y) for y in range(1990,2025)))]

df_new = df_new[df_new['Series Name'] == 'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)'].copy()
df_melt = pd.melt(df_new, id_vars=['Country Name'], value_vars=co2_cols, 
                  var_name='Year_str', value_name='Mt_CO2')
df_melt['Year'] = df_melt['Year_str'].str.extract('(\\d{4})').astype(int)
df_melt['Country'] = df_melt['Country Name']
df_melt['Region'] = df_melt['Country'].map(region_map)  # Add region!
df_melt['Kilotons of Co2'] = pd.to_numeric(df_melt['Mt_CO2'], errors='coerce') * 1000
df_melt = df_melt.dropna(subset=['Kilotons of Co2', 'Region'])  # Drop missing CO2 or region

print(f"Processed data: {df_melt.shape[0]} rows across {df_melt['Country'].nunique()} countries")

# Encode categoricals (now includes Region)
le_country = LabelEncoder()
le_region = LabelEncoder()
df_melt['Country_encoded'] = le_country.fit_transform(df_melt['Country'])
df_melt['Region_encoded'] = le_region.fit_transform(df_melt['Region'])

feature_cols = ['Country_encoded', 'Region_encoded', 'Year']
X = df_melt[feature_cols]
y = df_melt['Kilotons of Co2']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train improved model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(Xtrain, ytrain)

# Metrics
y_pred = model.predict(Xtest)
print(f"Test R²: {r2_score(ytest, y_pred):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.0f}")

# Save model + encoders (full original compatibility)
with open("co2_model_V1.0.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "le_country": le_country,
        "le_region": le_region
    }, f)
print("Saved: co2_model_V1.0.pkl (with le_region)")

# UAE future predictions (2025-2030)
uae_idx = le_country.transform(["United Arab Emirates"])[0]
uae_region_idx = le_region.transform(["Asia"])[0]
future_X = np.array([[uae_idx, uae_region_idx, yr] for yr in range(2025, 2031)])
preds = model.predict(future_X)
print("UAE kt CO2 predictions (with Region feature):")
for yr, pred in zip(range(2025, 2031), preds):
    print(f"  {yr}: {pred:,.0f}")
