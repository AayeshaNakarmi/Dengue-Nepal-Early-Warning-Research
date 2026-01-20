"""
Diagnostic script to check actual station names in your weather files
"""

import pandas as pd

print("="*70)
print("STATION NAMES DIAGNOSTIC")
print("="*70)

# Load temperature data
print("\n1. TEMPERATURE STATIONS (from part1.xlsx - MAX sheet):")
print("-"*70)
temp_max = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part1.xlsx', 
                          sheet_name='Manual Daily Maximum Air Tempe')

# Get all column names except the first (date) column
temp_stations = [col for col in temp_max.columns[1:]]
print(f"Found {len(temp_stations)} temperature stations:")
for i, station in enumerate(temp_stations, 1):
    print(f"  {i}. '{station}'")

# Load humidity data
print("\n2. HUMIDITY STATIONS (from rh_part1.xlsx):")
print("-"*70)
humidity = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_rh_part1.xlsx', 
                          sheet_name='Manual Relative Humidity')

humidity_stations = [col for col in humidity.columns[1:]]
print(f"Found {len(humidity_stations)} humidity stations:")
for i, station in enumerate(humidity_stations, 1):
    print(f"  {i}. '{station}'")

# Load precipitation data
print("\n3. PRECIPITATION STATIONS (from part1.xlsx - PRECIPITATION sheet):")
print("-"*70)
precip = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part1.xlsx', 
                        sheet_name='24h accumulated Precipitation ')

precip_stations = [col for col in precip.columns[1:]]
print(f"Found {len(precip_stations)} precipitation stations:")
for i, station in enumerate(precip_stations, 1):
    print(f"  {i}. '{station}'")

# Load mapping to compare
print("\n4. STATIONS IN MAPPING FILE:")
print("-"*70)
mapping = pd.read_csv(r'E:\Dengue-Research-Project\data\station_district_mapping.csv')
print(f"Found {len(mapping)} districts in mapping")

# Check which stations from mapping are NOT in the actual data
print("\n5. MAPPING ISSUES:")
print("-"*70)

missing_temp = []
missing_humidity = []
missing_precip = []

for _, row in mapping.iterrows():
    district = row['district_code']
    temp_station = row['temp_station']
    humidity_station = row['humidity_station']
    precip_station = row['precipitation_station']
    
    if temp_station not in temp_stations:
        missing_temp.append(f"  District {district}: '{temp_station}' NOT FOUND")
    
    if humidity_station not in humidity_stations:
        missing_humidity.append(f"  District {district}: '{humidity_station}' NOT FOUND")
    
    if precip_station not in precip_stations:
        missing_precip.append(f"  District {district}: '{precip_station}' NOT FOUND")

print(f"\nMissing TEMPERATURE stations ({len(missing_temp)}):")
for msg in missing_temp[:10]:  # Show first 10
    print(msg)
if len(missing_temp) > 10:
    print(f"  ... and {len(missing_temp) - 10} more")

print(f"\nMissing HUMIDITY stations ({len(missing_humidity)}):")
for msg in missing_humidity[:10]:
    print(msg)
if len(missing_humidity) > 10:
    print(f"  ... and {len(missing_humidity) - 10} more")

print(f"\nMissing PRECIPITATION stations ({len(missing_precip)}):")
for msg in missing_precip[:10]:
    print(msg)
if len(missing_precip) > 10:
    print(f"  ... and {len(missing_precip) - 10} more")

print("\n" + "="*70)
print("SAVE THIS OUTPUT AND SHARE IT!")
print("="*70)