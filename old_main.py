"""
Integrate dengue_long.csv with station_district_mapping2.csv
Creates a consolidated dataset with Year, Week, District, Stations, and Cases columns.
"""

import pandas as pd
import os


def integrate_dengue_and_stations(dengue_file=None, mapping_file=None, output_file=None):
    """
    Integrate dengue data with station-district mapping.
    
    Parameters:
    -----------
    dengue_file : str, optional
        Path to the dengue_long.csv file.
    mapping_file : str, optional
        Path to the station_district_mapping2.csv file.
    output_file : str, optional
        Path to save the integrated output CSV file.
    
    Returns:
    --------
    pandas.DataFrame : Integrated dataframe with columns [Year, Week, District, Stations, Cases]
    """
    
    # Set default file paths
    base_dir = os.path.dirname(__file__)
    
    if dengue_file is None:
        dengue_file = os.path.join(base_dir, "data", "dengue_data", "dengue_long.csv")
    
    if mapping_file is None:
        mapping_file = os.path.join(base_dir, "station_district_mapping2.csv")
    
    if output_file is None:
        output_file = os.path.join(base_dir, "integrated_dengue_stations.csv")
    
    # Check if files exist
    if not os.path.exists(dengue_file):
        raise FileNotFoundError(f"Dengue file not found: {dengue_file}")
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    print("="*80)
    print("INTEGRATING DENGUE DATA WITH WEATHER STATIONS")
    print("="*80)
    print(f"\nReading files:")
    print(f"  Dengue data: {dengue_file}")
    print(f"  Station mapping: {mapping_file}\n")
    
    # Read CSV files
    dengue_df = pd.read_csv(dengue_file)
    mapping_df = pd.read_csv(mapping_file)
    
    print(f"Dengue data shape: {dengue_df.shape}")
    print(f"Dengue columns: {list(dengue_df.columns)}")
    print(f"Sample dengue data:")
    print(dengue_df.head(3))
    
    print(f"\nStation mapping shape: {mapping_df.shape}")
    print(f"Mapping columns: {list(mapping_df.columns)}")
    print(f"Sample mapping data:")
    print(mapping_df.head(3))
    
    # Extract district name from dengue data
    # Format: "101 TAPLEJUNG" -> extract "TAPLEJUNG"
    dengue_df['District_Clean'] = dengue_df['District'].str.extract(r'(\w+)$')[0].str.upper()
    
    print(f"\nCleaned district names from dengue data:")
    print(dengue_df[['District', 'District_Clean']].drop_duplicates().head(10))
    
    # Rename mapping columns to avoid conflict during merge
    mapping_df_temp = mapping_df.rename(columns={'District': 'District_Map'})
    
    # Merge on district
    integrated_df = pd.merge(
        dengue_df[['Year', 'Week', 'District_Clean', 'Cases']],
        mapping_df_temp,
        left_on='District_Clean',
        right_on='District_Map',
        how='left'
    )
    
    # Rename columns and reorder as requested: Year, Week, District, Stations, Cases
    integrated_df = integrated_df.rename(columns={
        'District_Clean': 'District',
        'Corresponding Stations': 'Stations'
    })
    
    # Select and reorder columns
    integrated_df = integrated_df[['Year', 'Week', 'District', 'Stations', 'Cases']]
    
    # Handle missing station mappings
    integrated_df['Stations'] = integrated_df['Stations'].fillna('No station found in dataset')
    
    # Sort by Year, Week, District
    integrated_df = integrated_df.sort_values(['Year', 'Week', 'District']).reset_index(drop=True)
    
    # Save to CSV
    integrated_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ INTEGRATION SUCCESSFUL")
    print(f"{'='*80}")
    print(f"Total rows: {len(integrated_df)}")
    print(f"Output saved to: {output_file}\n")
    
    # Print sample data
    print("Sample of integrated data:")
    print(integrated_df.head(10).to_string(index=False))
    
    # Print statistics
    print(f"\n{'='*80}")
    print("DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Year range: {integrated_df['Year'].min()} - {integrated_df['Year'].max()}")
    print(f"Week range: {integrated_df['Week'].min()} - {integrated_df['Week'].max()}")
    print(f"Unique districts: {integrated_df['District'].nunique()}")
    print(f"Total cases: {integrated_df['Cases'].sum():.0f}")
    print(f"Average cases per record: {integrated_df['Cases'].mean():.2f}")
    print(f"Max cases in a week: {integrated_df['Cases'].max():.0f}")
    
    # Districts without stations
    no_station = integrated_df[integrated_df['Stations'] == 'No station found in dataset']['District'].unique()
    # Filter out NaN values before sorting
    no_station = [d for d in no_station if pd.notna(d)]
    print(f"\nDistricts without stations: {len(no_station)}")
    if len(no_station) > 0:
        print(f"  {', '.join(sorted(no_station))}")
    
    print(f"\n{'='*80}\n")
    
    return integrated_df


if __name__ == "__main__":
    try:
        integrate_dengue_and_stations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
