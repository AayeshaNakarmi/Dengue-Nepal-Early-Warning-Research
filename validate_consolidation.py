import pandas as pd
import sys

def validate():
    print("--- DENGUE CONSOLIDATION VALIDATION SCRIPT ---")
    
    try:
        # Load raw data
        dengue_raw = pd.read_csv('data/dengue_data/dengue_long.csv')
        dengue_raw['Cases'] = dengue_raw['Cases'].fillna(0)
        
        # Load integrated data
        integrated = pd.read_csv('data/integrated_dengue_weather.csv')
        
        # 1. Validate Nawalparasi
        nawal_raw_sum = dengue_raw[dengue_raw['District'].str.contains('NAWALPARASI', na=False)]['Cases'].sum()
        nawal_integ_sum = integrated[integrated['District'] == 'Nawalparasi']['Cases'].sum()
        
        print(f"\nNawalparasi Total Cases:")
        print(f"  Raw (Sum of East + West): {nawal_raw_sum}")
        print(f"  Integrated (Consolidated): {nawal_integ_sum}")
        
        if nawal_raw_sum == nawal_integ_sum:
            print("  ✓ Nawalparasi Validation PASSED!")
        else:
            print("  ✗ Nawalparasi Validation FAILED!")
            
        # 2. Validate a random district (e.g., Kathmandu)
        ktm_raw_sum = dengue_raw[dengue_raw['District'].str.contains('KATHMANDU', na=False)]['Cases'].sum()
        ktm_integ_sum = integrated[integrated['District'] == 'Kathmandu']['Cases'].sum()
        
        print(f"\nKathmandu Total Cases:")
        print(f"  Raw: {ktm_raw_sum}")
        print(f"  Integrated: {ktm_integ_sum}")
        
        if ktm_raw_sum == ktm_integ_sum:
            print("  ✓ Kathmandu Validation PASSED!")
        else:
            print("  ✗ Kathmandu Validation FAILED!")
            
        # 3. Check specific Week sample for Nawalparasi
        sample_week_raw = dengue_raw[dengue_raw['District'].str.contains('NAWALPARASI', na=False)].groupby(['Year', 'Week'])['Cases'].sum().iloc[0]
        sample_year = dengue_raw[dengue_raw['District'].str.contains('NAWALPARASI', na=False)].groupby(['Year', 'Week'])['Cases'].sum().index[0][0]
        sample_week = dengue_raw[dengue_raw['District'].str.contains('NAWALPARASI', na=False)].groupby(['Year', 'Week'])['Cases'].sum().index[0][1]
        
        sample_integ = integrated[(integrated['District'] == 'Nawalparasi') & (integrated['Year'] == sample_year) & (integrated['Week'] == sample_week)]['Cases'].sum()
        
        print(f"\nNawalparasi Sample Week ({sample_year}-W{sample_week}):")
        print(f"  Raw Sum: {sample_week_raw}")
        print(f"  Integrated Value: {sample_integ}")
        
        if sample_week_raw == sample_integ:
             print("  ✓ Weekly Sample Validation PASSED!")
        else:
             print("  ✗ Weekly Sample Validation FAILED!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure you are running the script from the project root directory.")

if __name__ == "__main__":
    validate()
