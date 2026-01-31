"""
Extract actual station names from Islington_rh_part1 and part2 CSV files.
Run this on your local machine in the Dengue project directory.
Shows the mapping before integration.
"""

import pandas as pd
import os
from pathlib import Path

# Mapping of dengue district names to station file district names
DISTRICT_NAME_MAPPING = {
    'ARGHAKHANCHI': 'Arghakhachi',
    'CHITAWAN': 'Chitwan',
    'DHANUSA': 'Dhanusha',
    'EAST': 'Nawalparasi',
    'KAPILBASTU': 'Kapilvastu',
    'KAVREPALANCHOK': 'Kavrepalanchowk',
    'SINDHUPALCHOK': 'Sindhupalchowk',
    'SOLUKHUMBU': 'Sholukhumbu',
    'WEST': 'Nawalparasi',
}

def extract_stations():
    """Extract stations from Islington CSV files."""
    
    weather_dir = Path('./data/weather_data')
    
    # Load both station files
    part1_file = weather_dir / 'Islington_rh_part1_Stations.csv'
    part2_file = weather_dir / 'Islington_rh_part2_Stations.csv'
    
    all_stations = {}
    
    # Load Part 1
    if part1_file.exists():
        print(f"Loading {part1_file.name}...")
        df1 = pd.read_csv(part1_file)
        print(f"  Columns: {list(df1.columns)}")
        print(f"  Shape: {df1.shape}")
        print(f"  First few rows:")
        print(df1.head(3))
        
        # Group by district
        for _, row in df1.iterrows():
            district = str(row.get('District', '')).strip().upper()
            name = str(row.get('Name', '')).strip()
            
            if district and name and district != 'NAN' and name != 'NAN':
                if district not in all_stations:
                    all_stations[district] = []
                if name not in all_stations[district]:
                    all_stations[district].append(name)
    else:
        print(f"NOT FOUND: {part1_file}")
    
    # Load Part 2
    if part2_file.exists():
        print(f"\nLoading {part2_file.name}...")
        df2 = pd.read_csv(part2_file)
        print(f"  Columns: {list(df2.columns)}")
        print(f"  Shape: {df2.shape}")
        print(f"  First few rows:")
        print(df2.head(3))
        
        # Group by district
        for _, row in df2.iterrows():
            district = str(row.get('District', '')).strip().upper()
            name = str(row.get('Name', '')).strip()
            
            if district and name and district != 'NAN' and name != 'NAN':
                if district not in all_stations:
                    all_stations[district] = []
                if name not in all_stations[district]:
                    all_stations[district].append(name)
    else:
        print(f"NOT FOUND: {part2_file}")
    
    # Print all unique districts found
    print(f"\n{'='*80}")
    print(f"ALL DISTRICTS FOUND IN STATION FILES (TOTAL: {len(all_stations)})")
    print(f"{'='*80}")
    for dist in sorted(all_stations.keys()):
        stations = all_stations[dist]
        print(f"{dist}: {', '.join(sorted(stations))}")
    
    # Create mapping table
    print(f"\n{'='*80}")
    print(f"MAPPING TABLE: Dengue District → Station File District | Stations")
    print(f"{'='*80}")
    
    mapping_data = []
    
    for dengue_dist, file_dist in sorted(DISTRICT_NAME_MAPPING.items()):
        # Check if file_dist exists in stations (case-insensitive)
        file_dist_key = None
        for key in all_stations.keys():
            if key == file_dist.upper():
                file_dist_key = key
                break
        
        if file_dist_key:
            stations = sorted(all_stations[file_dist_key])
            stations_str = ", ".join(stations)
            mapping_data.append({
                'Dengue_District': dengue_dist,
                'Station_File_District': file_dist,
                'Stations': stations_str,
                'Count': len(stations)
            })
            print(f"{dengue_dist:20} → {file_dist:20} | {stations_str}")
        else:
            print(f"{dengue_dist:20} → {file_dist:20} | NOT FOUND IN STATION FILES")
            mapping_data.append({
                'Dengue_District': dengue_dist,
                'Station_File_District': file_dist,
                'Stations': 'NOT FOUND',
                'Count': 0
            })
    
    # Save mapping table
    mapping_df = pd.DataFrame(mapping_data)
    output_file = './station_mapping_with_names.csv'
    mapping_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved mapping table to: {output_file}")
    
    # Display the CSV
    print(f"\n{'='*80}")
    print("MAPPING TABLE CSV OUTPUT:")
    print(f"{'='*80}")
    print(mapping_df.to_string(index=False))
    
    return mapping_df

if __name__ == "__main__":
    try:
        extract_stations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
