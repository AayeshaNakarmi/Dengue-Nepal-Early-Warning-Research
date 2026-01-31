"""
Station Extraction Script - Run this on your local machine
Extracts stations from Islington weather files and creates mapping table
"""

import pandas as pd
import os
import sys

# Mapping of dengue district names to station file district names (aliases)
DENGUE_TO_STATION_MAPPING = {
    'ARGHAKHANCHI': 'Arghakhachi',
    'CHITAWAN': 'Chitwan',
    'DHANUSA': 'Dhanusha',
    'EAST': 'Nawalparasi',
    'KAPILBASTU': 'Kapilvastu',
    'KAVREPALANCHOK': 'Kavrepalanchowk',
    'SINDHUPALCHOK': 'Sindhupalchowk',
    'SOLUKHUMBU': 'Sholukhumbu',
    'WEST': 'Nawalparasi'
}

def extract_stations_from_excel(file_path, sheet_name):
    """Extract unique station column names from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # First column is dates, rest are stations
        stations = [col.strip() for col in df.columns[1:]]
        return stations
    except FileNotFoundError:
        print(f"ERROR: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return []

def parse_district_from_station_name(station_name):
    """
    Extract district name from station column name
    Common patterns:
    - "District Name - Info"
    - "District - Station"
    - Just "District Name"
    """
    # Split on common delimiters
    for delimiter in [' - ', '-', '(', ')']:
        if delimiter in station_name:
            base = station_name.split(delimiter)[0].strip()
            if base:
                return base
    return station_name.strip()

def main():
    print("="*90)
    print("ISLINGTON WEATHER DATA - STATION EXTRACTION")
    print("="*90)
    
    # Define file paths (adjust if needed)
    base_path = './data/weather_data/'
    humidity_part1_path = os.path.join(base_path, 'Islington_rh_part1.xlsx')
    humidity_part2_path = os.path.join(base_path, 'Islington_rh_part2.xlsx')
    
    # Check files exist
    files_to_check = [humidity_part1_path, humidity_part2_path]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        print("\nMISSING FILES:")
        for f in missing_files:
            print(f"  ✗ {f}")
        print("\nPlease ensure the weather data files are in ./data/weather_data/")
        return False
    
    print(f"\n✓ Found all required files")
    
    # Extract stations from both parts
    print(f"\nExtracting stations from Islington_rh_part1.xlsx...")
    stations_part1 = extract_stations_from_excel(humidity_part1_path, 'Manual Relative Humidity')
    print(f"  Found {len(stations_part1)} station columns")
    
    print(f"\nExtracting stations from Islington_rh_part2.xlsx...")
    stations_part2 = extract_stations_from_excel(humidity_part2_path, 'Manual Relative Humidity')
    print(f"  Found {len(stations_part2)} station columns")
    
    # Combine all stations
    all_stations = sorted(set(stations_part1 + stations_part2))
    
    print(f"\nTotal unique stations across both files: {len(all_stations)}\n")
    
    # Display all stations found
    print("-"*90)
    print("ALL STATIONS FOUND IN ISLINGTON FILES:")
    print("-"*90)
    for i, station in enumerate(all_stations, 1):
        district = parse_district_from_station_name(station)
        print(f"{i:3d}. {station:60s} | District: {district}")
    
    # Group stations by district (fuzzy matching)
    stations_by_district = {}
    for station in all_stations:
        district = parse_district_from_station_name(station)
        if district not in stations_by_district:
            stations_by_district[district] = []
        stations_by_district[district].append(station)
    
    # Create mapping table
    print("\n" + "="*90)
    print("DENGUE → STATION FILE MAPPING TABLE")
    print("="*90 + "\n")
    
    mapping_data = []
    found_count = 0
    missing_count = 0
    
    for dengue_district, file_district_alias in sorted(DENGUE_TO_STATION_MAPPING.items()):
        # Find best match in extracted districts
        matching_stations = stations_by_district.get(file_district_alias, [])
        
        if matching_stations:
            stations_str = '; '.join(matching_stations)
            found_count += 1
        else:
            stations_str = "⚠ NO STATIONS FOUND"
            missing_count += 1
            # Try to find similar names
            similar = [d for d in stations_by_district.keys() if file_district_alias.lower() in d.lower()]
            if similar:
                stations_str += f" (Similar: {', '.join(similar)})"
        
        mapping_data.append({
            'Dengue District': dengue_district,
            'File District': file_district_alias,
            'Station Names': stations_str,
            'Station Count': len(matching_stations)
        })
        
        # Print formatted row
        print(f"Dengue: {dengue_district:20s} → File: {file_district_alias:20s}")
        print(f"  Stations ({len(matching_stations)}): {stations_str}\n")
    
    # Create DataFrame
    mapping_df = pd.DataFrame(mapping_data)
    
    # Save to CSV
    output_csv = './dengue_station_mapping_table.csv'
    mapping_df.to_csv(output_csv, index=False)
    print("="*90)
    print(f"✓ Mapping table saved to: {output_csv}\n")
    
    # Display summary
    print("SUMMARY STATISTICS:")
    print(f"  Total mappings defined: {len(mapping_df)}")
    print(f"  Mappings with stations found: {found_count}")
    print(f"  Mappings with NO stations: {missing_count}")
    print(f"  Total unique stations discovered: {len(all_stations)}")
    
    # Show districts in data but not in mapping
    unmapped_districts = set(stations_by_district.keys()) - set(DENGUE_TO_STATION_MAPPING.values())
    if unmapped_districts:
        print(f"\nDistricts in data but NOT in mapping ({len(unmapped_districts)}):")
        for district in sorted(unmapped_districts):
            stations = stations_by_district[district]
            print(f"  - {district}: {', '.join(stations)}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
