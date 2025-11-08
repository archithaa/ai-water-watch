
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
NUM_ROWS = 300
STATES = {
    "Arizona": "AZ",
    "California": "CA",
    "Nevada": "NV",
    "Texas": "TX",
    "Utah": "UT"
}
STATE_POPULATIONS = {
    "Arizona": 7.4,
    "California": 39.2,
    "Nevada": 3.2,
    "Texas": 30.0,
    "Utah": 3.4
}
# Base water use per million people in MLD (Million Liters per Day)
WATER_USE_PER_MILLION_PEOPLE = 1000

# Temperature and precipitation ranges (monthly variations will be added)
STATE_METRICS = {
    "Arizona": {"temp_range": (15, 35), "precip_range": (5, 30)},
    "California": {"temp_range": (10, 25), "precip_range": (10, 100)},
    "Nevada": {"temp_range": (5, 25), "precip_range": (10, 40)},
    "Texas": {"temp_range": (15, 30), "precip_range": (40, 120)},
    "Utah": {"temp_range": (5, 25), "precip_range": (20, 60)}
}

def generate_data():
    """Generates synthetic water usage data."""
    data = []
    state_names = list(STATES.keys())
    
    # Generate dates for the last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    months = pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m').tolist()

    for i in range(1, NUM_ROWS + 1):
        state = random.choice(state_names)
        month = random.choice(months)
        
        # Basic info
        row = {
            "id": i,
            "month": month,
            "state": state,
            "state_abbr": STATES[state],
            "population_million": STATE_POPULATIONS[state]
        }
        
        # Data centers
        row["data_centers"] = random.randint(20, 400)
        
        # Weather data with monthly seasonality
        month_index = int(month.split('-')[1])
        
        temp_range = STATE_METRICS[state]["temp_range"]
        # Simple seasonality: warmer in summer months (6-8)
        temp_seasonality = (month_index - 6) if month_index in [6,7,8] else 0
        row["avg_temp_celsius"] = round(random.uniform(temp_range[0], temp_range[1]) + temp_seasonality * 1.5, 2)

        precip_range = STATE_METRICS[state]["precip_range"]
        # Simple seasonality: less rain in summer
        precip_seasonality = (month_index - 6) if month_index in [6,7,8] else 0
        row["precipitation_mm"] = round(max(0, random.uniform(precip_range[0], precip_range[1]) - precip_seasonality * 5), 2)

        # Drought Index (correlate with temp and precip)
        drought = (row["avg_temp_celsius"] / 30) - (row["precipitation_mm"] / 100) + np.random.normal(0, 0.1)
        row["drought_index"] = round(max(0, min(1, drought)), 3)
        
        # Total Water Use (MLD)
        population_water_use = row["population_million"] * WATER_USE_PER_MILLION_PEOPLE
        # Assume data center water use is proportional to number of data centers and temperature
        datacenter_water_use = row["data_centers"] * (1 + row["avg_temp_celsius"] / 30) * np.random.uniform(0.5, 1.5)
        total_water_use = population_water_use + datacenter_water_use
        # Add some noise
        total_water_use *= np.random.uniform(0.95, 1.05)
        row["total_water_use_mld"] = round(total_water_use, 2)

        # Cooling efficiency
        row["cooling_efficiency"] = round(random.uniform(0.5, 0.9) - (row["avg_temp_celsius"] - 20) * 0.01, 3)

        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("water_5states_300rows.csv", index=False)
    print("Successfully generated water_5states_300rows.csv")

