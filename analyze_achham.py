import pandas as pd
import numpy as np

df = pd.read_csv('data/integrated_dengue_weather.csv')
achham = df[df['District'] == 'Achham'].sort_values(['Year', 'Week'])

print("--- Achham (Jomsom Station) - 2019 Samples ---")
pd.set_option('display.max_columns', None)
cols = ['Year', 'Week', 'temp_lag0', 'humidity_lag0', 'precipitation_lag0']
print(achham[cols].head(10))

print("\n--- Checking for Identical Rows across Weeks for Achham ---")
# Check how many consecutive weeks have identical precipitation
diffs = achham['precipitation_lag0'].diff() == 0
print(f"Number of weeks where precipitation is exactly the same as the previous week: {diffs.sum()} / {len(achham)}")

print("\n--- Why is this happening? ---")
print("Achham uses the 'Jomsom' weather station. If the Jomsom station has missing data for a week,")
print("the script uses 'ffill()' (forward fill) which copies the value from the last available week.")

print("\n--- Comparing Achham with another Jomsom district (Doti) ---")
doti = df[df['District'] == 'Doti'].sort_values(['Year', 'Week'])
compare = pd.merge(
    achham[cols].rename(columns={c: f'achham_{c}' for c in cols[2:]}),
    doti[cols].rename(columns={c: f'doti_{c}' for c in cols[2:]}),
    on=['Year', 'Week']
)
print(compare.head(5))

# Check if Achham and Doti have identical values (they should, due to station sharing)
identical_weather = (compare['achham_temp_lag0'] == compare['doti_temp_lag0']).all()
print(f"\nDo Achham and Doti share identical weather data? {identical_weather}")
