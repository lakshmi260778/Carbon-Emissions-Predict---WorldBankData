import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🌍", layout="wide")

# Trend-aware predictor class (must match train_model_V1.0.py)
class TrendAwarePredictor:
    def __init__(self, model, le_country, le_region, trend_slopes, last_hist_year):
        self.model = model
        self.le_country = le_country
        self.le_region = le_region
        self.trend_slopes = trend_slopes
        self.last_hist_year = last_hist_year
    
    def predict(self, country_encoded, region_encoded, year):
        """Predict with trend projection for future years"""
        X = pd.DataFrame([{
            'Country_encoded': country_encoded,
            'Region_encoded': region_encoded,
            'Year': year
        }])
        base_pred = self.model.predict(X)[0]
        
        country_name = self.le_country.inverse_transform([country_encoded])[0]
        slope = self.trend_slopes.get(country_name, 0)
        
        if year > self.last_hist_year:
            years_ahead = year - self.last_hist_year
            trend_adjustment = slope * years_ahead
            return base_pred + trend_adjustment
        else:
            return base_pred

# 1. Load trained model and encoders
@st.cache_resource
def load_model():
    with open("co2_model_V1.0.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
base_model = model_data["model"]
le_country = model_data["le_country"]
le_region = model_data["le_region"]
trend_slopes = model_data.get("trend_slopes", {})
last_hist_year = model_data.get("last_hist_year", 2024)

# Create trend-aware predictor
model = TrendAwarePredictor(base_model, le_country, le_region, trend_slopes, last_hist_year)

# Region mapping (must match train_model_V1.0.py)
region_map = {
    'Afghanistan': 'Asia', 'Albania': 'Europe', 'Algeria': 'Africa', 'Angola': 'Africa',
    'Argentina': 'Americas', 'Armenia': 'Asia', 'Aruba': 'Americas', 'Australia': 'Oceania',
    'Austria': 'Europe', 'Azerbaijan': 'Asia', 'Bahamas': 'Americas', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Barbados': 'Americas', 'Belarus': 'Europe', 'Belgium': 'Europe',
    'Belize': 'Americas', 'United Arab Emirates': 'Asia', 'United States': 'Americas',
    'China': 'Asia', 'India': 'Asia', 'Saudi Arabia': 'Asia', 'Germany': 'Europe',
    'Brazil': 'Americas', 'Canada': 'Americas', 'France': 'Europe', 'Italy': 'Europe',
    'Japan': 'Asia', 'Mexico': 'Americas', 'Russia': 'Europe', 'South Africa': 'Africa',
    'United Kingdom': 'Europe', 'Indonesia': 'Asia', 'Turkey': 'Asia', 'Iran': 'Asia',
    'Thailand': 'Asia', 'Egypt': 'Africa', 'Nigeria': 'Africa', 'Pakistan': 'Asia',
    'Vietnam': 'Asia', 'Philippines': 'Asia', 'Malaysia': 'Asia', 'Singapore': 'Asia',
    'South Korea': 'Asia', 'Spain': 'Europe', 'Poland': 'Europe', 'Netherlands': 'Europe',
    'Sweden': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe', 'Finland': 'Europe',
    'Switzerland': 'Europe', 'New Zealand': 'Oceania', 'Chile': 'Americas', 'Peru': 'Americas',
    'Colombia': 'Americas', 'Venezuela': 'Americas', 'Ecuador': 'Americas', 'Bolivia': 'Americas',
    'Paraguay': 'Americas', 'Uruguay': 'Americas', 'Argentina': 'Americas', 'Morocco': 'Africa',
    'Tunisia': 'Africa', 'Libya': 'Africa', 'Sudan': 'Africa', 'Ethiopia': 'Africa',
    'Kenya': 'Africa', 'Tanzania': 'Africa', 'Uganda': 'Africa', 'Ghana': 'Africa',
    'Ivory Coast': 'Africa', 'Senegal': 'Africa', 'Mali': 'Africa', 'Burkina Faso': 'Africa',
    'Niger': 'Africa', 'Chad': 'Africa', 'Cameroon': 'Africa', 'Gabon': 'Africa',
    'Congo': 'Africa', 'Democratic Republic of the Congo': 'Africa', 'Angola': 'Africa',
    'Zambia': 'Africa', 'Zimbabwe': 'Africa', 'Botswana': 'Africa', 'Namibia': 'Africa',
    'Mozambique': 'Africa', 'Madagascar': 'Africa', 'Mauritius': 'Africa', 'Seychelles': 'Africa',
    'Israel': 'Asia', 'Jordan': 'Asia', 'Lebanon': 'Asia', 'Syria': 'Asia', 'Iraq': 'Asia',
    'Kuwait': 'Asia', 'Qatar': 'Asia', 'Oman': 'Asia', 'Yemen': 'Asia', 'Afghanistan': 'Asia',
    'Uzbekistan': 'Asia', 'Kazakhstan': 'Asia', 'Turkmenistan': 'Asia', 'Kyrgyzstan': 'Asia',
    'Tajikistan': 'Asia', 'Mongolia': 'Asia', 'Nepal': 'Asia', 'Bhutan': 'Asia', 'Sri Lanka': 'Asia',
    'Maldives': 'Asia', 'Myanmar': 'Asia', 'Cambodia': 'Asia', 'Laos': 'Asia', 'Brunei': 'Asia',
    'Timor-Leste': 'Asia', 'Papua New Guinea': 'Oceania', 'Fiji': 'Oceania', 'Samoa': 'Oceania',
    'Tonga': 'Oceania', 'Vanuatu': 'Oceania', 'Solomon Islands': 'Oceania', 'Kiribati': 'Oceania',
    'Tuvalu': 'Oceania', 'Nauru': 'Oceania', 'Palau': 'Oceania', 'Marshall Islands': 'Oceania',
    'Micronesia': 'Oceania', 'Cuba': 'Americas', 'Haiti': 'Americas', 'Dominican Republic': 'Americas',
    'Jamaica': 'Americas', 'Trinidad and Tobago': 'Americas', 'Barbados': 'Americas',
    'Saint Lucia': 'Americas', 'Grenada': 'Americas', 'Saint Vincent and the Grenadines': 'Americas',
    'Antigua and Barbuda': 'Americas', 'Dominica': 'Americas', 'Saint Kitts and Nevis': 'Americas',
    'Bahamas': 'Americas', 'Belize': 'Americas', 'Costa Rica': 'Americas', 'Panama': 'Americas',
    'Guatemala': 'Americas', 'Honduras': 'Americas', 'El Salvador': 'Americas', 'Nicaragua': 'Americas',
    'Guyana': 'Americas', 'Suriname': 'Americas', 'French Guiana': 'Americas'
}

# (Optional) load data once to get valid choices for country and region
@st.cache_data
def load_data():
    try:
        df_new = pd.read_csv("2c068aa1-30dc-45b2-bd84-6730351ffa25_Data.csv")
        
        # Filter for CO2 emissions data
        df_new = df_new[df_new['Series Name'] == 'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)'].copy()
        
        # Get year columns
        co2_cols = [col for col in df_new.columns if col.startswith(tuple(str(y) for y in range(1990, 2025)))]
        
        # Melt from wide to long format
        df_melt = pd.melt(df_new, id_vars=['Country Name'], value_vars=co2_cols,
                          var_name='Year_str', value_name='Mt_CO2')
        df_melt['Year'] = df_melt['Year_str'].str.extract('(\\d{4})').astype(int)
        df_melt['Country'] = df_melt['Country Name']
        df_melt['Region'] = df_melt['Country'].map(region_map)
        df_melt['Kilotons of Co2'] = pd.to_numeric(df_melt['Mt_CO2'], errors='coerce') * 1000
        df_melt = df_melt.dropna(subset=['Kilotons of Co2', 'Region'])
        
        return df_melt
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Create country-region mapping for validation
    country_region_map = df.groupby("Country")["Region"].first().to_dict()
    regions = sorted(df["Region"].unique().tolist())
else:
    # fallback if CSV not present in production
    country_region_map = {}
    regions = []

# Sidebar for inputs and predict button
#st.sidebar.title("CO₂ Emissions Predictor")
#st.sidebar.markdown("<small>Predict kilotons of CO₂ based on country, region, and year.</small>", unsafe_allow_html=True)
#st.sidebar.markdown("---")

# Get list of countries that the model knows
model_countries = set(le_country.classes_)

# Input widgets in sidebar
st.sidebar.subheader("Input Parameters")

if regions:
    region = st.sidebar.selectbox("Region", regions)
    # Filter countries based on selected region AND model training data
    countries_in_region = sorted([c for c, r in country_region_map.items() if r == region and c in model_countries])
    if countries_in_region:
        country = st.sidebar.selectbox("Country", countries_in_region)
    else:
        st.sidebar.warning(f"No trained model data available for countries in {region}")
        country = None
else:
    region = st.sidebar.text_input("Region (type name)")
    country = st.sidebar.text_input("Country (type name)")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, value=2030, step=1)

st.sidebar.markdown("---")


# Main page content
st.title("CO₂ Emissions Predictor")
st.subheader("Global Warming Potential (GWP)")
st.markdown("""
<small>
<b>About:</b> This application helps you explore, predict, and visualize carbon dioxide (CO₂) emissions from the agriculture, energy, waste, and industrial sectors across countries over time.
<b>Units:</b> CO₂ emissions are expressed in carbon dioxide equivalent (CO₂e)<br>
<b>Data Source:</b> World Development Indicators<br><br> 
</small>
""", unsafe_allow_html=True)
st.info("Choose a region, country and year in the sidebar and click 'Predict CO₂ Emissions' to see the result.")
 

prediction_result = None
if country and st.button("🚀 Predict CO₂ Emissions", type="primary"):
    try:
        # Encode the user inputs using the same encoders from training
        country_encoded = le_country.transform([country])[0]
        region_encoded = le_region.transform([region])[0]

        # Use trend-aware prediction (model already includes trend projection)
        pred = model.predict(country_encoded, region_encoded, year)
        
        prediction_result = pred
        st.success(f"**Predicted: {pred:,.0f} kilotons**")
        st.markdown("<medium>Be prudent. Act wise. Protect our resources.</medium>", unsafe_allow_html=True)  
    except Exception as e:
        st.error(f"Error: {e}")

# Visualization section in sidebar
#st.sidebar.markdown("---")
st.sidebar.subheader("Visualizations")

show_top_emitters = st.sidebar.button("📊 View vs Top Emitters", use_container_width=True)
show_regional_avg = st.sidebar.button("📈 View vs Regional Average", use_container_width=True)
show_latest_comparison = st.sidebar.button("📉 View Latest Year Comparison", use_container_width=True)

# 4. Visualizations Section - Only show when button is clicked
if df is not None and country:
    # Prepare data for visualization
    # Year column already exists from load_data()
    
    # Helper function to predict future values using trend-aware model
    def predict_future(c_name, r_name, years):
        # Check if country is in the model's training data
        if c_name not in model_countries:
            return None  # Country not in training data
        
        c_encoded = le_country.transform([c_name])[0]
        r_encoded = le_region.transform([r_name])[0]
        
        predictions = []
        for y in years:
            # Use the trend-aware predictor
            pred = model.predict(c_encoded, r_encoded, y)
            predictions.append(pred)
        return predictions
    
    # Get year range for predictions
    last_historical_year = df["Year"].max()
    future_years = list(range(last_historical_year + 1, year + 1)) if year > last_historical_year else []
    
    # Plot 1: Selected Country vs Top Emitters Over Time
    if show_top_emitters:
        with st.expander(f"📊 {country} vs Top Emitting Countries", expanded=True):
            # Get top 5 countries by total emissions for comparison (only those in model)
            top_countries = df.groupby("Country")["Kilotons of Co2"].sum().nlargest(10).index.tolist()
            top_countries = [c for c in top_countries if c in model_countries][:5]
            
            # Ensure selected country is included (if it's in the model)
            if country not in top_countries and country in model_countries:
                comparison_countries = top_countries[:4] + [country]
            else:
                comparison_countries = top_countries
            
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            for c in comparison_countries:
                # Historical data
                country_data = df[df["Country"] == c].sort_values("Year")
                c_region = country_region_map[c]
                
                if c == country:
                    # Plot historical data for selected country
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", linewidth=3, label=f"{c} (Historical)", color="red")
                    
                    # Plot future predictions for selected country
                    if future_years:
                        future_preds = predict_future(c, c_region, future_years)
                        if future_preds:
                            ax1.plot(future_years, future_preds, 
                                    marker="x", linewidth=3, linestyle="--", 
                                    label=f"{c} (Predicted)", color="darkred")
                else:
                    # Plot historical data for other countries
                    ax1.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                            marker="o", alpha=0.7, label=f"{c} (Historical)")
                    
                    # Plot future predictions for other countries
                    if future_years:
                        future_preds = predict_future(c, c_region, future_years)
                        if future_preds:
                            ax1.plot(future_years, future_preds, 
                                    marker="x", alpha=0.7, linestyle="--", label=f"{c} (Predicted)")
            
            # Add vertical line to separate historical and predicted
            if future_years:
                ax1.axvline(x=last_historical_year + 0.5, color="gray", linestyle=":", alpha=0.7, label="Prediction Start")
            
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Kilotons of CO2")
            ax1.set_title(f"CO2 Emissions Over Time: {country} vs Top Emitters (with Predictions till {year})")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
    
    # Plot 2: Selected Country vs Regional Average
    if show_regional_avg:
        with st.expander(f"📈 {country} vs {region} Regional Average", expanded=True):
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Selected country - Historical data
            country_data = df[df["Country"] == country].sort_values("Year")
            ax2.plot(country_data["Year"], country_data["Kilotons of Co2"], 
                    marker="o", linewidth=3, label=f"{country} (Historical)", color="red")
            
            # Selected country - Future predictions
            if future_years and country in model_countries:
                future_preds = predict_future(country, region, future_years)
                if future_preds:
                    ax2.plot(future_years, future_preds, 
                            marker="x", linewidth=3, linestyle="--", 
                            label=f"{country} (Predicted)", color="darkred")
            
            # Regional average - Historical
            region_data = df[df["Region"] == region].groupby("Year")["Kilotons of Co2"].mean().reset_index()
            ax2.plot(region_data["Year"], region_data["Kilotons of Co2"], 
                    marker="s", linewidth=2, label=f"{region} Average (Historical)", color="blue", linestyle="-")
            
            # Regional average - Future predictions (predict for all countries in region and average)
            if future_years:
                region_countries = [c for c, r in country_region_map.items() if r == region and c in model_countries]
                region_future_preds = []
                for y in future_years:
                    year_preds = []
                    for c in region_countries:
                        c_region = country_region_map[c]
                        pred_list = predict_future(c, c_region, [y])
                        if pred_list:
                            year_preds.append(pred_list[0])
                    if year_preds:
                        region_future_preds.append(sum(year_preds) / len(year_preds))
                
                ax2.plot(future_years, region_future_preds, 
                        marker="x", linewidth=2, linestyle="--", 
                        label=f"{region} Average (Predicted)", color="darkblue")
            
            # Add vertical line to separate historical and predicted
            if future_years:
                ax2.axvline(x=last_historical_year + 0.5, color="gray", linestyle=":", alpha=0.7, label="Prediction Start")
            
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Kilotons of CO2")
            ax2.set_title(f"{country} vs {region} Regional Average Over Time (with Predictions till {year})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
    
    # Plot 3: Bar Chart Comparison - Selected Year
    if show_latest_comparison:
        with st.expander(f"📉 Top 10 Countries Comparison ({year})", expanded=True):
            # Get historical data for latest year
            latest_historical_year = df["Year"].max()
            
            if year <= latest_historical_year:
                # Use historical data
                year_data = df[df["Year"] == year].groupby("Country")["Kilotons of Co2"].sum().nlargest(10)
            else:
                # Use predictions for countries in model only
                year_preds = {}
                for c in model_countries:
                    c_region = country_region_map.get(c)
                    if c_region:
                        pred_list = predict_future(c, c_region, [year])
                        if pred_list:
                            year_preds[c] = pred_list[0]
                year_data = pd.Series(year_preds).nlargest(10)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            colors = ["red" if c == country else "steelblue" for c in year_data.index]
            sns.barplot(x=year_data.values, y=year_data.index, palette=colors, ax=ax3)
            ax3.set_xlabel("Kilotons of CO2")
            title_type = "Historical" if year <= latest_historical_year else "Predicted"
            ax3.set_title(f"Top 10 Countries by CO2 Emissions ({year}) - {title_type}")
            st.pyplot(fig3)
