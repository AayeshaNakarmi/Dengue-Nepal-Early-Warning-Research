# Repository Updates & Milestones

This document tracks the major updates and technical milestones reached in the Nepal Dengue-Weather Research Project.

## Latest Milestones (March 2026)

### 1. Full Weather Data Coverage (2019–2025)
- **REACHED**: All 77 districts now have complete daily weather data from January 2019 through December 2025.
- **SOURCE**: ERA5-Land Reanalysis (Copernicus Climate Data Store).
- **VARIABLES**: 
    - Temperature (Mean, Max, Min)
    - Heat Index (Derived)
    - Relative Humidity (Derived)
    - Precipitation (De-accumulated)
    - Wind Speed (Derived)
    - Soil Moisture (Layer 1)

### 2. Automated ERA5-Land Pipeline
- **NEW**: The `ERA5.py` script provides a fully automated, resilient pipeline for data retrieval and processing.
- **EFFICIENCY**: Uses a centralized Nepal-wide download strategy to minimize API calls and redundant processing.
- **RELIABILITY**: Implements a 3-level resume system (File level, JSON Progress Tracker, CSV Date Auditing) to handle interruptions.

### 3. Integrated Long-term Dataset (2019–2024)
- **NEW**: Weekly integrated dataset (`integrated_dengue_weather.csv`) combining reported dengue cases with aggregated weather lags.
- **DISTRICTS**: Standardized district names and consolidated Nawalparasi (East/West) to match administrative boundaries.
- **LAYS**: 0 to 4-week lags generated for all weather features to support predictive modeling.

## Technical Validation
- All data consolidations are verified using `validate_consolidation.py`.
- Comprehensive data integrity audits are performed by `validate_integration_full.py`.
