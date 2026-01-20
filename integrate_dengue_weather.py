"""
Dengue-Weather Data Integration Script
Integrates weekly dengue case data with daily weather observations
Handles multiple Excel files with multiple sheets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
LAG_WEEKS = [0, 1, 2, 3, 4]  # Create lagged weather variables (current + 1-4 weeks prior)

def load_data():
    """Load all required data files"""
    print("Loading data files...")
    
    # Load dengue data
    dengue = pd.read_csv(r'E:\Dengue-Research-Project\data\dengue_data\dengue_long.csv')
    print(f"✓ Loaded dengue data: {dengue.shape}")
    
    # Load temperature data (PART1 and PART2 with MAX and MIN sheets)
    print("\nLoading temperature data...")
    temp_max_part1 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part1.xlsx', 
                                    sheet_name='Manual Daily Maximum Air Tempe')
    temp_min_part1 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part1.xlsx', 
                                    sheet_name='Manual Daily Minimum Air Tempe')
    temp_max_part2 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part2.xlsx', 
                                    sheet_name='Manual Daily Maximum Air Tempe')
    temp_min_part2 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part2.xlsx', 
                                    sheet_name='Manual Daily Minimum Air Tempe')
    print(f"✓ Loaded temperature MAX PART1: {temp_max_part1.shape}")
    print(f"✓ Loaded temperature MIN PART1: {temp_min_part1.shape}")
    print(f"✓ Loaded temperature MAX PART2: {temp_max_part2.shape}")
    print(f"✓ Loaded temperature MIN PART2: {temp_min_part2.shape}")
    
    # Load precipitation data
    print("\nLoading precipitation data...")
    precip_part1 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part1.xlsx', 
                                  sheet_name='24h accumulated Precipitation ')
    precip_part2 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_part2.xlsx', 
                                  sheet_name='24h accumulated Precipitation ')
    print(f"✓ Loaded precipitation PART1: {precip_part1.shape}")
    print(f"✓ Loaded precipitation PART2: {precip_part2.shape}")
    
    # Load humidity data (RH_PART1 and RH_PART2)
    print("\nLoading humidity data...")
    humidity_part1 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_rh_part1.xlsx', 
                                    sheet_name='Manual Relative Humidity')
    humidity_part2 = pd.read_excel(r'E:\Dengue-Research-Project\data\weather_data\Islington_rh_part2.xlsx', 
                                    sheet_name='Manual Relative Humidity')
    print(f"✓ Loaded humidity PART1: {humidity_part1.shape}")
    print(f"✓ Loaded humidity PART2: {humidity_part2.shape}")
    
    # Load station mapping
    mapping = pd.read_csv(r'E:\Dengue-Research-Project\data\station_district_mapping.csv')
    print(f"\n✓ Loaded station mapping: {len(mapping)} districts")
    
    return (dengue, 
            temp_max_part1, temp_min_part1, temp_max_part2, temp_min_part2,
            precip_part1, precip_part2,
            humidity_part1, humidity_part2,
            mapping)

def clean_column_names(df):
    """Standardize column names by removing extra whitespace only"""
    # Only strip whitespace - keep the units as they're part of station names
    df.columns = df.columns.str.strip()
    return df

def parse_weather_dates(df):
    """Parse dates from weather data"""
    # Handle different possible date formats
    date_col = df.columns[0]  # First column should be date/time
    
    # The dates are in dd/mm/yyyy HH:MM:SS format
    try:
        df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        parsed_count = df['date'].notna().sum()
        
        if parsed_count > 0:
            print(f"✓ Parsed {parsed_count} dates successfully")
        else:
            raise ValueError("No dates parsed")
            
    except:
        # Try alternative format without time
        try:
            df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
            parsed_count = df['date'].notna().sum()
            
            if parsed_count > 0:
                print(f"✓ Parsed {parsed_count} dates successfully (without time)")
            else:
                raise ValueError("No dates parsed")
        except:
            # Last resort: let pandas figure it out
            df['date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            parsed_count = df['date'].notna().sum()
            
            if parsed_count > 0:
                print(f"✓ Parsed {parsed_count} dates successfully (auto-detection)")
            else:
                print(f"ERROR: Could not parse dates in column '{date_col}'")
                print(f"First few values: {df[date_col].head()}")
                df['date'] = pd.NaT
                return df
    
    # Remove rows with invalid dates
    initial_rows = len(df)
    df = df[df['date'].notna()].copy()
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"  Note: Dropped {dropped_rows} rows with invalid dates")
    
    return df

def convert_to_numeric(df):
    """Convert all columns except 'date' to numeric values"""
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def combine_parts(part1, part2):
    """Combine PART1 and PART2 dataframes"""
    # Clean column names
    part1 = clean_column_names(part1)
    part2 = clean_column_names(part2)
    
    # Parse dates
    part1 = parse_weather_dates(part1)
    part2 = parse_weather_dates(part2)
    
    # Convert all weather values to numeric
    part1 = convert_to_numeric(part1)
    part2 = convert_to_numeric(part2)
    
    # Check if date column exists
    if 'date' not in part1.columns or 'date' not in part2.columns:
        print("ERROR: Date column not found after parsing")
        return pd.DataFrame()
    
    # Remove rows with NaT dates
    part1 = part1[part1['date'].notna()]
    part2 = part2[part2['date'].notna()]
    
    if len(part1) == 0 and len(part2) == 0:
        print("ERROR: No valid dates found in either part")
        return pd.DataFrame()
    
    # Combine by concatenating rows and removing duplicates
    combined = pd.concat([part1, part2], ignore_index=True)
    
    # Remove duplicate dates (keep first occurrence)
    if 'date' in combined.columns:
        combined = combined.drop_duplicates(subset=['date'], keep='first')
    
    # Sort by date
    combined = combined.sort_values('date').reset_index(drop=True)
    
    print(f"  Combined: {len(part1)} + {len(part2)} = {len(combined)} records (after removing duplicates)")
    
    return combined

def calculate_mean_temperature(temp_max, temp_min):
    """Calculate mean temperature from max and min"""
    # Get date column
    dates = temp_max['date']
    
    # Get all station columns (exclude date)
    stations = [col for col in temp_max.columns if col != 'date']
    
    # Calculate mean for each station
    temp_mean = pd.DataFrame({'date': dates})
    
    for station in stations:
        if station in temp_min.columns:
            temp_mean[station] = (temp_max[station] + temp_min[station]) / 2
        else:
            print(f"Warning: {station} not found in temp_min, using only temp_max")
            temp_mean[station] = temp_max[station]
    
    return temp_mean

def aggregate_weather_to_weekly(weather_df, agg_func='mean'):
    """Aggregate daily weather data to weekly summaries using Sunday-Saturday weeks"""
    # Set date as index
    weather_df = weather_df.set_index('date')
    
    # Create a proper week assignment based on Sunday-Saturday
    # Week 1 starts on the first Sunday of the year
    weather_df['day_of_week'] = weather_df.index.dayofweek  # Monday=0, Sunday=6
    
    # Find the first Sunday of each year
    years = weather_df.index.year.unique()
    week_assignments = []
    
    for date_idx in weather_df.index:
        year = date_idx.year
        
        # Find first Sunday of the year
        jan_1 = pd.Timestamp(year, 1, 1)
        days_until_sunday = (6 - jan_1.dayofweek) % 7
        first_sunday = jan_1 + pd.Timedelta(days=days_until_sunday)
        
        if date_idx < first_sunday:
            # Dates before first Sunday belong to last week of previous year
            week_num = 52  # or 53
            year_for_week = year - 1
        else:
            # Calculate week number from first Sunday
            days_since_first_sunday = (date_idx - first_sunday).days
            week_num = (days_since_first_sunday // 7) + 1
            year_for_week = year
        
        week_assignments.append((year_for_week, week_num))
    
    weather_df['year'] = [x[0] for x in week_assignments]
    weather_df['week'] = [x[1] for x in week_assignments]
    
    # Group by year and week
    if agg_func == 'sum':
        weekly = weather_df.groupby(['year', 'week']).sum()
    else:
        weekly = weather_df.groupby(['year', 'week']).mean()
    
    return weekly.reset_index()

def create_lagged_features(weather_weekly, mapping, variable_type='temperature'):
    """Create lagged weather features for each district"""
    results = []
    
    station_col_name = f'{variable_type}_station'
    if variable_type == 'temp':
        station_col_name = 'temp_station'
    elif variable_type == 'humidity':
        station_col_name = 'humidity_station'
    elif variable_type == 'precipitation':
        station_col_name = 'precipitation_station'
    
    # Convert mapping district_code to string
    mapping = mapping.copy()
    mapping['district_code'] = mapping['district_code'].astype(str).str.strip()
    
    for _, district_row in mapping.iterrows():
        district_code = district_row['district_code']
        station_name = district_row[station_col_name]
        
        # Check if station exists in weather data
        if station_name not in weather_weekly.columns:
            print(f"Warning: Station '{station_name}' not found for district {district_code} ({variable_type})")
            continue
        
        district_weather = weather_weekly[['year', 'week', station_name]].copy()
        district_weather['district_code'] = district_code
        district_weather.rename(columns={station_name: f'{variable_type}_lag0'}, inplace=True)
        
        # Create lagged features
        for lag in range(1, max(LAG_WEEKS) + 1):
            if lag in LAG_WEEKS:
                district_weather[f'{variable_type}_lag{lag}'] = district_weather[f'{variable_type}_lag0'].shift(lag)
        
        results.append(district_weather)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        print(f"ERROR: No valid data for {variable_type}")
        return pd.DataFrame()

def create_complete_dengue_frame(dengue, start_year=2019, end_year=2024):
    """Create a complete dengue dataframe with all weeks from start to end year"""
    print(f"\nCreating complete week coverage from {start_year} to {end_year}...")
    
    # Get all unique districts
    districts = dengue['District'].unique()
    
    # Create complete year-week combinations
    all_combinations = []
    for year in range(start_year, end_year + 1):
        # Most years have 52 weeks, some have 53
        # Use 52 for simplicity (Week 1-52)
        for week in range(1, 53):
            for district in districts:
                all_combinations.append({
                    'Year': year,
                    'Week': week,
                    'District': district
                })
    
    complete_frame = pd.DataFrame(all_combinations)
    
    # Merge with actual dengue data
    complete_frame = complete_frame.merge(
        dengue[['Year', 'Week', 'District', 'Cases']],
        on=['Year', 'Week', 'District'],
        how='left'
    )
    
    # Fill missing cases with 0
    complete_frame['Cases'] = complete_frame['Cases'].fillna(0)
    
    print(f"✓ Created {len(complete_frame)} records ({len(districts)} districts × {(end_year-start_year+1)*52} weeks)")
    print(f"✓ Filled {(complete_frame['Cases'] == 0).sum()} weeks with 0 cases")
    
    return complete_frame

def integrate_all_data(dengue, temp_weekly, humidity_weekly, precip_weekly):
    """Merge dengue data with all weather variables"""
    print("\nIntegrating datasets...")
    
    # Start with dengue data
    integrated = dengue.copy()
    
    # Fill missing Cases with 0 (weeks with no reported cases)
    integrated['Cases'] = integrated['Cases'].fillna(0)
    print(f"✓ Filled missing dengue cases with 0")
    
    # Extract district code from District column (e.g., "101 TAPLEJUNG" -> "101")
    integrated['District_Code'] = integrated['District'].str.split(' ').str[0].str.strip()
    print(f"✓ Extracted district codes from District column")
    
    # Convert district codes to string to ensure consistent data types
    integrated['District_Code'] = integrated['District_Code'].astype(str)
    
    # Convert district_code in weather data to string
    if not temp_weekly.empty:
        temp_weekly['district_code'] = temp_weekly['district_code'].astype(str).str.strip()
    if not humidity_weekly.empty:
        humidity_weekly['district_code'] = humidity_weekly['district_code'].astype(str).str.strip()
    if not precip_weekly.empty:
        precip_weekly['district_code'] = precip_weekly['district_code'].astype(str).str.strip()
    
    # Merge temperature
    if not temp_weekly.empty:
        integrated = integrated.merge(
            temp_weekly,
            left_on=['Year', 'Week', 'District_Code'],
            right_on=['year', 'week', 'district_code'],
            how='left'
        )
        integrated = integrated.drop(['year', 'week', 'district_code'], axis=1, errors='ignore')
        print("✓ Merged temperature data")
        # Check merge success
        temp_filled = integrated['temp_lag0'].notna().sum()
        print(f"  {temp_filled}/{len(integrated)} rows have temperature data ({temp_filled/len(integrated)*100:.1f}%)")
    
    # Merge humidity
    if not humidity_weekly.empty:
        integrated = integrated.merge(
            humidity_weekly,
            left_on=['Year', 'Week', 'District_Code'],
            right_on=['year', 'week', 'district_code'],
            how='left',
            suffixes=('', '_hum')
        )
        integrated = integrated.drop(['year', 'week', 'district_code'], axis=1, errors='ignore')
        print("✓ Merged humidity data")
        # Check merge success
        hum_filled = integrated['humidity_lag0'].notna().sum()
        print(f"  {hum_filled}/{len(integrated)} rows have humidity data ({hum_filled/len(integrated)*100:.1f}%)")
    
    # Merge precipitation
    if not precip_weekly.empty:
        integrated = integrated.merge(
            precip_weekly,
            left_on=['Year', 'Week', 'District_Code'],
            right_on=['year', 'week', 'district_code'],
            how='left',
            suffixes=('', '_precip')
        )
        integrated = integrated.drop(['year', 'week', 'district_code'], axis=1, errors='ignore')
        print("✓ Merged precipitation data")
        # Check merge success
        precip_filled = integrated['precipitation_lag0'].notna().sum()
        print(f"  {precip_filled}/{len(integrated)} rows have precipitation data ({precip_filled/len(integrated)*100:.1f}%)")
    
    # Forward fill weather data for missing weeks (use previous week's data)
    print("\n✓ Forward filling missing weather data...")
    weather_cols = [col for col in integrated.columns if any(x in col for x in ['temp_', 'humidity_', 'precipitation_'])]
    
    for district in integrated['District'].unique():
        district_mask = integrated['District'] == district
        for col in weather_cols:
            integrated.loc[district_mask, col] = integrated.loc[district_mask, col].fillna(method='ffill')
    
    filled_after = integrated[weather_cols].notna().sum().sum()
    total_cells = len(integrated) * len(weather_cols)
    print(f"  Weather data completeness: {filled_after}/{total_cells} cells ({filled_after/total_cells*100:.1f}%)")
    
    # Drop the temporary District_Code column (keep original District)
    integrated = integrated.drop(['District_Code'], axis=1, errors='ignore')
    
    return integrated

def calculate_data_completeness(df):
    """Calculate and report data completeness"""
    print("\n" + "="*70)
    print("DATA COMPLETENESS REPORT")
    print("="*70)
    
    total_rows = len(df)
    
    print(f"\nTotal records: {total_rows:,}")
    print(f"\nMissing data by column:")
    print("-" * 70)
    
    missing_stats = []
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / total_rows) * 100
        if missing > 0:
            missing_stats.append({
                'Column': col,
                'Missing': missing,
                'Percentage': f"{missing_pct:.2f}%"
            })
    
    if missing_stats:
        missing_df = pd.DataFrame(missing_stats)
        missing_df = missing_df.sort_values('Missing', ascending=False)
        print(missing_df.to_string(index=False))
    else:
        print("No missing data found! ✓")
    
    print("\n" + "="*70)

def main():
    """Main execution function"""
    print("="*70)
    print("DENGUE-WEATHER DATA INTEGRATION")
    print("="*70)
    
    # Load data
    (dengue, 
     temp_max_p1, temp_min_p1, temp_max_p2, temp_min_p2,
     precip_p1, precip_p2,
     humidity_p1, humidity_p2,
     mapping) = load_data()
    
    # Create complete dengue frame with all weeks from 2019 onwards
    print("\n" + "="*70)
    print("CREATING COMPLETE WEEK COVERAGE")
    print("="*70)
    
    # Determine the year range from the data
    min_year = dengue['Year'].min()
    max_year = dengue['Year'].max()
    
    dengue_complete = create_complete_dengue_frame(dengue, start_year=min_year, end_year=max_year)
    
    # Combine parts
    print("\n" + "="*70)
    print("COMBINING DATA PARTS")
    print("="*70)
    
    print("\nCombining temperature MAX...")
    temp_max = combine_parts(temp_max_p1, temp_max_p2)
    print(f"✓ Combined temperature MAX: {temp_max.shape}")
    
    print("\nCombining temperature MIN...")
    temp_min = combine_parts(temp_min_p1, temp_min_p2)
    print(f"✓ Combined temperature MIN: {temp_min.shape}")
    
    print("\nCalculating mean temperature...")
    temp_mean = calculate_mean_temperature(temp_max, temp_min)
    print(f"✓ Calculated mean temperature: {temp_mean.shape}")
    
    print("\nCombining precipitation...")
    precip = combine_parts(precip_p1, precip_p2)
    print(f"✓ Combined precipitation: {precip.shape}")
    
    print("\nCombining humidity...")
    humidity = combine_parts(humidity_p1, humidity_p2)
    print(f"✓ Combined humidity: {humidity.shape}")
    
    # Aggregate to weekly
    print("\n" + "="*70)
    print("AGGREGATING TO WEEKLY DATA")
    print("="*70)
    
    print("\nAggregating temperature (mean)...")
    temp_weekly_raw = aggregate_weather_to_weekly(temp_mean, 'mean')
    print(f"✓ Weekly temperature: {temp_weekly_raw.shape}")
    
    print("\nAggregating humidity (mean)...")
    humidity_weekly_raw = aggregate_weather_to_weekly(humidity, 'mean')
    print(f"✓ Weekly humidity: {humidity_weekly_raw.shape}")
    
    print("\nAggregating precipitation (sum)...")
    precip_weekly_raw = aggregate_weather_to_weekly(precip, 'sum')
    print(f"✓ Weekly precipitation: {precip_weekly_raw.shape}")
    
    # Create district-level lagged features
    print("\n" + "="*70)
    print("CREATING LAGGED FEATURES BY DISTRICT")
    print("="*70)
    
    print("\nCreating temperature features with lags...")
    temp_weekly = create_lagged_features(temp_weekly_raw, mapping, 'temp')
    print(f"✓ Temperature features: {temp_weekly.shape}")
    
    print("\nCreating humidity features with lags...")
    humidity_weekly = create_lagged_features(humidity_weekly_raw, mapping, 'humidity')
    print(f"✓ Humidity features: {humidity_weekly.shape}")
    
    print("\nCreating precipitation features with lags...")
    precip_weekly = create_lagged_features(precip_weekly_raw, mapping, 'precipitation')
    print(f"✓ Precipitation features: {precip_weekly.shape}")
    
    # Integrate all data
    print("\n" + "="*70)
    print("MERGING ALL DATA")
    print("="*70)
    final_data = integrate_all_data(dengue_complete, temp_weekly, humidity_weekly, precip_weekly)
    
    # Calculate completeness
    calculate_data_completeness(final_data)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_file = r'E:\Dengue-Research-Project\data\integrated_dengue_weather.csv'
    final_data.to_csv(output_file, index=False)
    print(f"\n✓ Integrated data saved to: {output_file}")
    
    # Create summary statistics
    print("\nCreating summary statistics...")
    
    # Get numeric columns for summary
    numeric_cols = final_data.select_dtypes(include=[np.number]).columns
    
    summary = final_data.groupby('District')[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    summary.to_csv(r'E:\Dengue-Research-Project\data\summary_statistics.csv')
    print("✓ Summary statistics saved to: summary_statistics.csv")
    
    # Display sample of final data
    print("\n" + "="*70)
    print("SAMPLE OF INTEGRATED DATA")
    print("="*70)
    print("\nFirst 5 rows:")
    print(final_data.head(5).to_string())
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE! ✓")
    print("="*70)
    print(f"\nFinal dataset:")
    print(f"  • Shape: {final_data.shape[0]:,} rows × {final_data.shape[1]} columns")
    print(f"  • Date range: {final_data['Year'].min()}-W{final_data['Week'].min()} to {final_data['Year'].max()}-W{final_data['Week'].max()}")
    print(f"  • Districts: {final_data['District'].nunique()}")
    print(f"  • Total dengue cases: {final_data['Cases'].sum():,.0f}")
    print(f"\nOutput files created:")
    print(f"  1. integrated_dengue_weather.csv")
    print(f"  2. summary_statistics.csv")
    print(f"  3. station_district_mapping.csv")
    
    return final_data

if __name__ == "__main__":
    final_data = main()