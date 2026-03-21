# ERA5-Land Data Pipeline Guide

The `ERA5.py` script is a specialized tool for retrieving and processing high-resolution meteorological data for all 77 districts of Nepal.

## Core Features

### 1. Centralized Download Strategy
Instead of requesting data district-by-district (which would hit API rate limits), the pipeline:
1. Defines a Nepal-wide Bounding Box: `[30.5, 80.0, 26.3, 88.2]`.
2. Downloads month-long chunks of raw NetCDF data once.
3. Spatially clips the shared file for each of the 77 districts using administrative shapefiles.

### 2. Robust Resume System
The pipeline maintains state across three levels:
- **Level 1 (File)**: Skips download if a valid `.nc` file exists in `era5_downloads/`.
- **Level 2 (Progress Tracker)**: Uses `pipeline_progress.json` to track completed (district, year, month, chunk) combinations.
- **Level 3 (CSV Audit)**: Checks existing `daily_weather.csv` for specific dates before re-processing any chunk.

### 3. Physics & Meteorology Logic
- **Temperature**: Converts Kelvin to Celsius.
- **Heat Index**: Implements Steadman's Heat Index with fallbacks for low temp/humidity.
- **Humidity**: Derived using the Magnus formula from 2m Temperature and 2m Dewpoint.
- **Precipitation**: ERA5 `tp` is cumulative per forecast step; the pipeline applies `diff()` and `clip(min=0)` to extract true hourly/daily amounts.
- **Spatial Clipping**: Uses `rioxarray` for precise polygon-masking, with a fallback to bounding-box means if errors occur.

## Usage
```bash
# Full run (requires CDS_API_KEY in .env)
python ERA5.py

# Merge existing CSVs without calling the API
python ERA5.py --merge-only
```
