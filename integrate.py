"""
Integrate weather station data by mapping stations to their respective districts.
Reads station CSV files and creates a consolidated mapping of districts to stations.
"""

import pandas as pd
import os
from pathlib import Path


def integrate_stations_by_district(weather_data_dir=None, output_file=None):
    """
    Consolidate weather stations and map them to their respective districts.
    
    Parameters:
    -----------
    weather_data_dir : str, optional
        Path to the weather_data directory containing station CSV files.
        If None, uses the default weather_data path.
    output_file : str, optional
        Path to save the output CSV file.
        If None, saves as 'station_district_mapping.csv' in weather_data directory.
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with columns ['District', 'Corresponding Stations']
    """
    
    # Set default weather_data directory
    if weather_data_dir is None:
        weather_data_dir = os.path.join(os.path.dirname(__file__), "data", "weather_data")
    
    # Check if directory exists
    if not os.path.exists(weather_data_dir):
        raise FileNotFoundError(f"Weather data directory not found: {weather_data_dir}")
    
    # Find all station CSV files
    station_files = [f for f in os.listdir(weather_data_dir) 
                     if f.endswith('_Stations.csv')]
    
    if not station_files:
        print(f"No station CSV files found in {weather_data_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(station_files)} station file(s)")
    print(f"Files: {station_files}\n")
    
    # Dictionary to store district to stations mapping
    district_stations = {}
    
    # Process each station file
    for station_file in station_files:
        file_path = os.path.join(weather_data_dir, station_file)
        print(f"Processing: {station_file}")
        
        try:
            # Read station CSV file
            df = pd.read_csv(file_path)
            
            # Expected columns: Station Index, Name, District, Latitude, Longitude, Elevation
            if 'District' not in df.columns or 'Name' not in df.columns:
                print(f"  ✗ Missing required columns (District, Name)")
                continue
            
            # Iterate through each row
            for _, row in df.iterrows():
                district = str(row['District']).strip().upper()
                station_name = str(row['Name']).strip()
                
                # Skip empty values
                if district and district != 'NAN' and station_name and station_name != 'NAN':
                    # Add station to district mapping
                    if district not in district_stations:
                        district_stations[district] = []
                    
                    # Avoid duplicate station entries
                    if station_name not in district_stations[district]:
                        district_stations[district].append(station_name)
            
            print(f"  ✓ Processed {len(df)} rows")
        
        except Exception as e:
            print(f"  ✗ Error processing {station_file}: {e}")
    
    # Create DataFrame from district_stations mapping
    result_data = []
    for district in sorted(district_stations.keys()):
        stations = district_stations[district]
        # Concatenate multiple stations with comma separator
        stations_str = ", ".join(sorted(stations))
        result_data.append({
            'District': district,
            'Corresponding Stations': stations_str
        })
    
    result_df = pd.DataFrame(result_data)
    
    # Set default output file path
    if output_file is None:
        output_file = os.path.join(weather_data_dir, 'station_district_mapping2.csv')
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully created station-district mapping")
    print(f"✓ Total districts: {len(result_df)}")
    print(f"✓ Output saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("Sample of mapped districts:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df


def verify_mapping(mapping_df):
    """
    Verify and display the station-district mapping.
    
    Parameters:
    -----------
    mapping_df : pandas.DataFrame
        DataFrame with district to stations mapping
    """
    print(f"\nTotal unique districts: {len(mapping_df)}")
    print(f"\nDistricts with multiple stations:")
    multi_station_districts = mapping_df[mapping_df['Corresponding Stations'].str.contains(',')] 
    if len(multi_station_districts) > 0:
        print(multi_station_districts.to_string(index=False))
    else:
        print("None")
    
    print(f"\nDistricts with no stations:")
    no_station_districts = mapping_df[mapping_df['Corresponding Stations'].str.contains('No station')]
    if len(no_station_districts) > 0:
        print(no_station_districts.to_string(index=False))
    else:
        print("All districts have at least one station assigned")


def integrate_dengue_with_stations(dengue_file=None, mapping_file=None, output_file=None):
    """
    Integrate dengue data with station-district mapping.
    Merges dengue cases data with corresponding stations for each district.
    
    Parameters:
    -----------
    dengue_file : str, optional
        Path to the dengue_long.csv file.
        If None, uses the default path in data/dengue_data/ directory.
    mapping_file : str, optional
        Path to the station_district_mapping.csv file.
        If None, uses the default path in data/weather_data/ directory.
    output_file : str, optional
        Path to save the integrated output CSV file.
        If None, saves as 'integrated_dengue_stations.csv' in data/ directory.
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with columns [Year, Week, District, Cases, Corresponding Stations]
    """
    
    # Set default file paths
    if dengue_file is None:
        dengue_file = os.path.join(os.path.dirname(__file__), "data", "dengue_data", "dengue_long.csv")
    
    if mapping_file is None:
        mapping_file = os.path.join(os.path.dirname(__file__), "data", "weather_data", "station_district_mapping.csv")
    
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "data", "integrated_dengue_stations.csv")
    
    # Check if files exist
    if not os.path.exists(dengue_file):
        raise FileNotFoundError(f"Dengue data file not found: {dengue_file}")
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Station mapping file not found: {mapping_file}")
    
    print(f"Reading dengue data from: {dengue_file}")
    print(f"Reading station mapping from: {mapping_file}\n")
    
    # Read dengue data
    dengue_df = pd.read_csv(dengue_file)
    print(f"Dengue data shape: {dengue_df.shape}")
    print(f"Dengue columns: {list(dengue_df.columns)}\n")
    
    # Read station mapping
    mapping_df = pd.read_csv(mapping_file)
    print(f"Station mapping shape: {mapping_df.shape}")
    print(f"Station mapping columns: {list(mapping_df.columns)}\n")
    
    # Extract district name from dengue data
    # District column format: "101 TAPLEJUNG" -> extract "TAPLEJUNG"
    dengue_df['District_Clean'] = dengue_df['District'].str.extract(r'(\w+)$')[0].str.upper()
    
    print("Sample of cleaned district names:")
    print(dengue_df[['District', 'District_Clean']].drop_duplicates().head(10))
    print()
    
    # Merge dengue data with station mapping
    integrated_df = pd.merge(
        dengue_df[['Year', 'Week', 'District_Clean', 'Cases']],
        mapping_df,
        left_on='District_Clean',
        right_on='District',
        how='left'
    )
    
    # Drop the duplicate 'District' column from mapping_df and rename District_Clean
    integrated_df = integrated_df.drop('District', axis=1)
    integrated_df = integrated_df.rename(columns={'District_Clean': 'District'})
    
    # Reorder columns
    integrated_df = integrated_df[['Year', 'Week', 'District', 'Cases', 'Corresponding Stations']]
    
    # Replace NaN in Corresponding Stations with "No station found in dataset"
    integrated_df['Corresponding Stations'] = integrated_df['Corresponding Stations'].fillna('No station found in dataset')
    
    # Sort by Year, Week, District
    integrated_df = integrated_df.sort_values(['Year', 'Week', 'District']).reset_index(drop=True)
    
    # Save to CSV
    integrated_df.to_csv(output_file, index=False)
    
    print(f"{'='*80}")
    print(f"✓ Successfully integrated dengue data with stations")
    print(f"✓ Total rows: {len(integrated_df)}")
    print(f"✓ Output saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("Sample of integrated data:")
    print(integrated_df.head(15).to_string(index=False))
    
    print(f"\nDistricts with no stations:")
    no_station = integrated_df[integrated_df['Corresponding Stations'] == 'No station found in dataset']['District'].unique()
    # Filter out NaN values before sorting
    no_station = [d for d in no_station if pd.notna(d)]
    if len(no_station) > 0:
        print(", ".join(sorted(no_station)))
    else:
        print("All districts have corresponding stations")
    
    print(f"\nData summary by year:")
    summary = integrated_df.groupby('Year').agg({
        'Week': 'max',
        'Cases': ['sum', 'mean', 'max'],
        'District': 'nunique'
    }).round(2)
    print(summary)
    
    return integrated_df


if __name__ == "__main__":
    try:
        # Step 1: Integrate and create station-district mapping
        print("Step 1: Creating station-district mapping...\n")
        mapping_df = integrate_stations_by_district()
        
        # Verify the mapping
        verify_mapping(mapping_df)
        
        # Step 2: Integrate dengue data with stations
        print("\n\nStep 2: Integrating dengue data with stations...\n")
        integrated_df = integrate_dengue_with_stations()
        
    except Exception as e:
        print(f"Error: {e}")
