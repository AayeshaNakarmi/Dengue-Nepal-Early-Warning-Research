"""
Extract and map stations from Islington weather files
Creates a clean mapping table showing Dengue District → Station File District | Station Names
"""

import pandas as pd
import os

# Mapping of dengue district names to station file district names
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
    """
    Extract station names from Excel file
    Returns list of station names (all columns except the first one which is dates)
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # First column is typically dates, rest are stations
        stations = [col.strip() for col in df.columns[1:]]
        return stations
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def extract_stations_by_district(humidity_part1_path, humidity_part2_path):
    """
    Extract stations from both humidity parts and organize by district
    """
    stations_by_district = {}
    
    # Load humidity data from both parts
    print("Loading humidity data from part1...")
    df_part1 = pd.read_excel(humidity_part1_path, sheet_name='Manual Relative Humidity')
    
    print("Loading humidity data from part2...")
    df_part2 = pd.read_excel(humidity_part2_path, sheet_name='Manual Relative Humidity')
    
    # Get all stations from both parts
    all_stations_part1 = [col.strip() for col in df_part1.columns[1:]]
    all_stations_part2 = [col.strip() for col in df_part2.columns[1:]]
    
    print("\nStations found in Part 1:")
    for station in all_stations_part1:
        print(f"  - {station}")
    
    print("\nStations found in Part 2:")
    for station in all_stations_part2:
        print(f"  - {station}")
    
    # Parse district from station names
    # Typical format: "District Name - Station Type" or just "District Name"
    for station in all_stations_part1:
        # Extract district name (usually the part before the dash or special characters)
        parts = station.split('-')
        district = parts[0].strip() if parts else station.strip()
        
        if district not in stations_by_district:
            stations_by_district[district] = []
        if station not in stations_by_district[district]:
            stations_by_district[district].append(station)
    
    for station in all_stations_part2:
        parts = station.split('-')
        district = parts[0].strip() if parts else station.strip()
        
        if district not in stations_by_district:
            stations_by_district[district] = []
        if station not in stations_by_district[district]:
            stations_by_district[district].append(station)
    
    return stations_by_district

def create_mapping_table(dengue_to_station, stations_by_district):
    """
    Create a clean mapping table
    """
    mapping_data = []
    
    for dengue_district, station_district in dengue_to_station.items():
        # Find matching stations
        matching_stations = stations_by_district.get(station_district, [])
        
        if matching_stations:
            stations_str = '; '.join(matching_stations)
        else:
            stations_str = "NO STATIONS FOUND"
        
        mapping_data.append({
            'Dengue District': dengue_district,
            'Station File District': station_district,
            'Station Names': stations_str,
            'Count': len(matching_stations)
        })
    
    return pd.DataFrame(mapping_data)

def main():
    # File paths
    humidity_part1 = './data/weather_data/Islington_rh_part1.xlsx'
    humidity_part2 = './data/weather_data/Islington_rh_part2.xlsx'
    
    print("="*80)
    print("STATION EXTRACTION AND MAPPING")
    print("="*80)
    
    # Check if files exist
    if not os.path.exists(humidity_part1):
        print(f"\nERROR: File not found: {humidity_part1}")
        print("Please ensure the weather data files are in ./data/weather_data/")
        return
    
    if not os.path.exists(humidity_part2):
        print(f"\nERROR: File not found: {humidity_part2}")
        return
    
    # Extract stations by district
    print("\nExtracting stations...")
    stations_by_district = extract_stations_by_district(humidity_part1, humidity_part2)
    
    # Create mapping table
    print("\nCreating mapping table...")
    mapping_df = create_mapping_table(DENGUE_TO_STATION_MAPPING, stations_by_district)
    
    # Display results
    print("\n" + "="*80)
    print("STATION MAPPING TABLE")
    print("="*80)
    print(mapping_df.to_string(index=False))
    
    # Save to CSV
    output_file = './dengue_to_station_mapping.csv'
    mapping_df.to_csv(output_file, index=False)
    print(f"\n✓ Mapping saved to: {output_file}")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total dengue districts mapped: {len(mapping_df)}")
    print(f"Districts with stations found: {(mapping_df['Count'] > 0).sum()}")
    print(f"Districts with NO stations found: {(mapping_df['Count'] == 0).sum()}")
    print(f"Total unique station entries: {mapping_df['Count'].sum()}")

if __name__ == "__main__":
    main()
