import geopandas as gpd

districts = gpd.read_file("npl_admin2.shp")
print(f"Total districts: {len(districts)}")
print(f"\nAll columns: {list(districts.columns)}")
print(f"\nFirst 5 rows:\n{districts.head()}")