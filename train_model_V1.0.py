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

# Custom predictor class that combines model with trend projection
class TrendAwarePredictor:
    def __init__(self, model, le_country, le_region, trend_slopes, last_hist_year):
        self.model = model
        self.le_country = le_country
        self.le_region = le_region
        self.trend_slopes = trend_slopes  # Dict: country -> slope
        self.last_hist_year = last_hist_year
    
    def predict(self, country_encoded, region_encoded, year):
        """Predict with trend projection for future years"""
        # Get base prediction from model
        X = pd.DataFrame([{
            'Country_encoded': country_encoded,
            'Region_encoded': region_encoded,
            'Year': year
        }])
        base_pred = self.model.predict(X)[0]
        
        # Get country name from encoded value
        country_name = self.le_country.inverse_transform([country_encoded])[0]
        
        # Apply trend adjustment for future years
        slope = self.trend_slopes.get(country_name, 0)
        if year > self.last_hist_year:
            years_ahead = year - self.last_hist_year
            trend_adjustment = slope * years_ahead
            return base_pred + trend_adjustment
        elif year < self.last_hist_year:
            # For historical years, just use base prediction
            return base_pred
        else:
            return base_pred
    
    def predict_batch(self, country_encoded_list, region_encoded_list, year_list):
        """Predict for multiple inputs"""
        predictions = []
        for c, r, y in zip(country_encoded_list, region_encoded_list, year_list):
            predictions.append(self.predict(c, r, y))
        return np.array(predictions)


# Load & process new World Bank data
df_new = pd.read_csv("2c068aa1-30dc-45b2-bd84-6730351ffa25_Data.csv")
co2_cols = [col for col in df_new.columns if col.startswith(tuple(str(y) for y in range(1990,2025)))]

df_new = df_new[df_new['Series Name'] == 'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)'].copy()
df_melt = pd.melt(df_new, id_vars=['Country Name'], value_vars=co2_cols, 
                  var_name='Year_str', value_name='Mt_CO2')
df_melt['Year'] = df_melt['Year_str'].str.extract('(\d{4})').astype(int)
df_melt['Country'] = df_melt['Country Name']
df_melt['Region'] = df_melt['Country'].map(region_map)  # Add region!
df_melt['Kilotons of Co2'] = pd.to_numeric(df_melt['Mt_CO2'], errors='coerce') * 1000
df_melt = df_melt.dropna(subset=['Kilotons of Co2', 'Region'])  # Drop missing CO2 or region

print(f"Processed data: {df_melt.shape[0]} rows across {df_melt['Country'].nunique()} countries")

# Calculate trend slopes for each country
trend_slopes = {}
for country in df_melt['Country'].unique():
    country_data = df_melt[df_melt['Country'] == country].sort_values('Year')
    if len(country_data) >= 2:
        # Use last 5 data points for trend
        recent = country_data.tail(5)
        years = recent['Year'].values
        emissions = recent['Kilotons of Co2'].values
        year_diff = years[-1] - years[0]
        if year_diff > 0:
            slope = (emissions[-1] - emissions[0]) / year_diff
            trend_slopes[country] = slope
        else:
            trend_slopes[country] = 0
    else:
        trend_slopes[country] = 0

last_hist_year = df_melt['Year'].max()
print(f"Calculated trends for {len(trend_slopes)} countries (last historical year: {last_hist_year})")

# Encode categoricals (now includes Region)
le_country = LabelEncoder()
le_region = LabelEncoder()
df_melt['Country_encoded'] = le_country.fit_transform(df_melt['Country'])
df_melt['Region_encoded'] = le_region.fit_transform(df_melt['Region'])

feature_cols = ['Country_encoded', 'Region_encoded', 'Year']
X = df_melt[feature_cols]
y = df_melt['Kilotons of Co2']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train base model
base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
base_model.fit(Xtrain, ytrain)

# Metrics
y_pred = base_model.predict(Xtest)
print(f"Test R²: {r2_score(ytest, y_pred):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(ytest, y_pred)):.0f}")

# Create trend-aware predictor
predictor = TrendAwarePredictor(base_model, le_country, le_region, trend_slopes, last_hist_year)

# Save model + encoders + trend data
with open("co2_model_V1.0.pkl", "wb") as f:
    pickle.dump({
        "model": base_model,
        "le_country": le_country,
        "le_region": le_region,
        "trend_slopes": trend_slopes,
        "last_hist_year": last_hist_year
    }, f)
print("Saved: co2_model_V1.0.pkl (with trend projection)")

# Test predictions for India at different years
print("\nIndia predictions (with trend projection):")
india_idx = le_country.transform(["India"])[0]
india_region_idx = le_region.transform(["Asia"])[0]
for yr in [2025, 2030, 2035, 2040]:
    pred = predictor.predict(india_idx, india_region_idx, yr)
    print(f"  {yr}: {pred:,.0f} kilotons")

# UAE future predictions (2025-2030)
print("\nUAE predictions (with trend projection):")
uae_idx = le_country.transform(["United Arab Emirates"])[0]
uae_region_idx = le_region.transform(["Asia"])[0]
for yr in range(2025, 2031):
    pred = predictor.predict(uae_idx, uae_region_idx, yr)
    print(f"  {yr}: {pred:,.0f} kilotons")
