import pandas as pd
import numpy as np

try:
    df = pd.read_csv('data/integrated_dengue_weather.csv')
    print(f'Total rows: {len(df)}')
    print(f'Duplicate rows: {df.duplicated().sum()}')
    print(f'Duplicate Year/Week/District: {df.duplicated(subset=["Year", "Week", "District"]).sum()}')
    
    print("\nDistinct Districts in integrated data:", df['District'].unique())
    print("\nNawalparasi Sample:")
    print(df[df['District'] == 'Nawalparasi'].head())
    
    # Check for repeating values in columns
    print("\nChecking for repeating weather values across districts in 2024-W1:")
    week_sample = df[(df['Year'] == 2024) & (df['Week'] == 1)]
    repeated_temp = week_sample['temp_lag0'].value_counts()
    print("Temperature lag0 occurrences in 2024-W1:")
    print(repeated_temp[repeated_temp > 1])
    
except Exception as e:
    print(f"Error: {e}")
