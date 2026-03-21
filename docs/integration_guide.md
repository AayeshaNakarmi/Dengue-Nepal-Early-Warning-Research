# Dengue-Weather Data Integration Guide

This guide explains how raw health data and processed weather data are merged into a unified dataset for analysis.

## Workflow Overview

The integration is handled by `integrate_dengue_weather.py` and follows these steps:

### 1. District Standardization
Raw dengue data often contains administrative codes or inconsistent naming (e.g., "101 TAPLEJUNG"). The script maps these to a standard list:
- **Nawalparasi Merge**: "Nawalparasi East" and "Nawalparasi West" are summed and consolidated into a single "Nawalparasi" entry.
- **Clean Titles**: All names are converted to Title Case (e.g., "ARGHAKHANCHI" -> "Arghakhachi" per mapping).

### 2. Temporal Alignment
- **Dengue Weeks**: Reported by year and week number (ISO-like, but often Sunday-Saturday).
- **Weather Aggregation**: Daily ERA5 or Manual observations are grouped by the project's week definition (Sunday to Saturday) to ensure perfect alignment with case counts.

### 3. Lagged Feature Engineering
To account for the biological delay in mosquito breeding and viral incubation, the script generates lagged weather variables:
- `temp_lag0` to `temp_lag4` (current week to 4 weeks prior).
- Applied to Temperature, Humidity, and Precipitation.

### 4. Missing Data Strategy
- **Forward Fill**: Missing weather weeks for a district are forward-filled from the previous week.
- **Zero Filling**: Missing dengue weeks are filled with `0` cases, assuming no reports indicate no cases (following standard epidemiological cleaning).

## Validation
Run the following to ensure the integration is correct:
```bash
# Verify Nawalparasi merge and overall consistency
python validate_consolidation.py

# Full audit of case totals and spatial sharing
python validate_integration_full.py
```
