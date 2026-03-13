"""
========================================================
ERA5-Land Automated Pipeline for Nepal Dengue Research
========================================================
Covers   : All 77 districts, 2020-present
Output   : nepal_dengue_weather_daily.csv
Variables: Temperature, Humidity, Precipitation,
           Soil Moisture, Wind Speed
========================================================

SETUP:
  pip install cdsapi xarray pandas geopandas rasterstats
              netCDF4 numpy rioxarray tqdm python-dotenv

.env file (same folder as script):
  CDS_API_KEY=your-actual-token-here

RUN:
  python nepal_dengue_era5_pipeline.py
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import cdsapi
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ===================================================================
#  LOAD API KEY
# ===================================================================

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
CDS_API_KEY = os.getenv("CDS_API_KEY")

if not CDS_API_KEY:
    print(
        "\n[ERROR] CDS_API_KEY not found!\n"
        "Create a .env file with: CDS_API_KEY=your-token\n"
        "Get token at: https://cds.climate.copernicus.eu/profile\n"
    )
    sys.exit(1)

# ===================================================================
#  CONFIGURATION
# ===================================================================

CONFIG = {
    "start_year":     2020,
    "end_year":       2020,
    "bbox":           [30.5, 80.0, 26.3, 88.2],   # Nepal N,W,S,E
    "shapefile_path": "npl_admin2.shp",
    "district_col":   "adm2_name",
    "download_dir":   "era5_downloads",
    "output_csv":     "nepal_dengue_weather_daily.csv",
    "variables": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "volumetric_soil_water_layer_1",
    ],
    "pause_between_requests": 10,   # seconds between month requests
    "retry_wait_seconds":     120,  # initial wait on rate limit
    "max_retries":            5,
}

# ===================================================================
#  LOGGING
# ===================================================================

log = logging.getLogger("dengue_pipeline")
log.setLevel(logging.INFO)

fh = logging.FileHandler("pipeline.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

ch = logging.StreamHandler(sys.stdout)
ch.stream.reconfigure(errors="replace")
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

log.addHandler(fh)
log.addHandler(ch)


# ===================================================================
#  STEP 1: DOWNLOAD - ONE MONTH AT A TIME
# ===================================================================

def download_era5_month(client, year: int, month: int, download_dir: str) -> str:
    """
    Download ERA5-Land data for a single month (one request per month).
    Skips if file already exists. Retries on rate limit.
    Returns path to NetCDF file or None on failure.
    """
    os.makedirs(download_dir, exist_ok=True)
    fname       = f"era5_nepal_{year}_{month:02d}.nc"
    output_file = os.path.join(download_dir, fname)

    if os.path.exists(output_file):
        log.info(f"    {year}-{month:02d}: already exists, skipping.")
        return output_file

    request_params = {
        "product_type": "reanalysis",
        "variable":     CONFIG["variables"],
        "year":         str(year),
        "month":        f"{month:02d}",
        "day":          [f"{d:02d}" for d in range(1, 32)],
        "time":         [f"{h:02d}:00" for h in range(24)],
        "area":         CONFIG["bbox"],
        "data_format":  "netcdf",
    }

    retries     = 0
    max_retries = CONFIG["max_retries"]
    wait        = CONFIG["retry_wait_seconds"]

    while retries <= max_retries:
        try:
            log.info(f"    Downloading {year}-{month:02d} "
                     f"[attempt {retries + 1}/{max_retries + 1}]...")
            client.retrieve("reanalysis-era5-land", request_params, output_file)
            log.info(f"    {year}-{month:02d}: saved -> {fname}")
            return output_file

        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate limit", "too many", "429", "cost limit", "403"]):
                retries += 1
                if retries > max_retries:
                    log.error(f"    {year}-{month:02d}: failed after {max_retries} retries.")
                    return None
                log.warning(f"    Request limit hit. Waiting {wait}s "
                            f"(retry {retries}/{max_retries})...")
                time.sleep(wait)
                wait *= 2
            else:
                log.error(f"    {year}-{month:02d}: failed: {e}")
                return None

    return None


def get_months_to_download(start_year: int, end_year: int):
    """Return list of (year, month) tuples to download, up to last complete month."""
    current = datetime.now()
    months  = []
    for year in range(start_year, end_year + 1):
        last_month = 12 if year < current.year else current.month - 1
        for month in range(1, last_month + 1):
            months.append((year, month))
    return months


# ===================================================================
#  STEP 2: PHYSICS CALCULATIONS
# ===================================================================

def kelvin_to_celsius(k):
    return k - 273.15

def relative_humidity(t_k, td_k):
    t  = kelvin_to_celsius(t_k)
    td = kelvin_to_celsius(td_k)
    rh = 100.0 * (
        np.exp((17.625 * td) / (243.04 + td)) /
        np.exp((17.625 * t)  / (243.04 + t))
    )
    return np.clip(rh, 0, 100)

def wind_speed(u, v):
    return np.sqrt(u ** 2 + v ** 2)


# ===================================================================
#  STEP 3: SPATIAL AGGREGATION TO DISTRICTS
# ===================================================================

def get_affine_transform(da):
    try:
        import rioxarray  # noqa
        return da.rio.set_spatial_dims(
            x_dim="longitude", y_dim="latitude").rio.transform()
    except Exception:
        from affine import Affine
        lons  = da.longitude.values
        lats  = da.latitude.values
        res_x = abs(float(lons[1]) - float(lons[0]))
        res_y = abs(float(lats[1]) - float(lats[0]))
        return Affine(res_x, 0, float(lons.min()) - res_x / 2,
                      0, -res_y, float(lats.max()) + res_y / 2)

def zonal_mean_per_district(da, districts, district_col):
    from rasterstats import zonal_stats
    arr       = da.values.copy()
    transform = get_affine_transform(da)
    if da.latitude.values[0] < da.latitude.values[-1]:
        arr = np.flipud(arr)
    stats = zonal_stats(
        districts, arr, affine=transform,
        stats=["mean"], nodata=np.nan, all_touched=True,
    )
    return {
        districts.iloc[i][district_col]: (
            round(float(s["mean"]), 4) if s["mean"] is not None else np.nan
        )
        for i, s in enumerate(stats)
    }


# ===================================================================
#  STEP 4: PROCESS ONE MONTH NetCDF -> DAILY DISTRICT ROWS
# ===================================================================

def process_netcdf(nc_file: str, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    ds = xr.open_dataset(nc_file)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    district_col = CONFIG["district_col"]

    t2m  = ds["t2m"]
    d2m  = ds["d2m"]
    tp   = ds["tp"] * 1000
    u10  = ds["u10"]
    v10  = ds["v10"]
    swvl = ds["swvl1"]

    rh_da = xr.DataArray(relative_humidity(t2m.values, d2m.values),
                         coords=t2m.coords, dims=t2m.dims)
    ws_da = xr.DataArray(wind_speed(u10.values, v10.values),
                         coords=t2m.coords, dims=t2m.dims)
    tc_da = xr.DataArray(kelvin_to_celsius(t2m.values),
                         coords=t2m.coords, dims=t2m.dims)

    daily = xr.Dataset({
        "temp_mean_celsius":     tc_da.resample(time="1D").mean(),
        "temp_max_celsius":      tc_da.resample(time="1D").max(),
        "temp_min_celsius":      tc_da.resample(time="1D").min(),
        "precipitation_mm":      tp.resample(time="1D").sum(),
        "relative_humidity_pct": rh_da.resample(time="1D").mean(),
        "soil_moisture_m3m3":    swvl.resample(time="1D").mean(),
        "wind_speed_ms":         ws_da.resample(time="1D").mean(),
    })

    district_names = districts[district_col].tolist()
    records        = []

    for dt in daily.time.values:
        date_str  = pd.Timestamp(dt).strftime("%Y-%m-%d")
        day_ds    = daily.sel(time=dt)
        var_stats = {
            var: zonal_mean_per_district(day_ds[var], districts, district_col)
            for var in daily.data_vars
        }
        for dist in district_names:
            row = {"date": date_str, "district": dist}
            for var in daily.data_vars:
                row[var] = var_stats[var].get(dist, np.nan)
            records.append(row)

    ds.close()
    return pd.DataFrame(records)


# ===================================================================
#  STEP 5: VALIDATE OUTPUT
# ===================================================================

def validate_output(df: pd.DataFrame):
    log.info("--- Output Summary ---")
    log.info(f"  Total rows   : {len(df):,}")
    log.info(f"  Date range   : {df['date'].min()} to {df['date'].max()}")
    log.info(f"  Districts    : {df['district'].nunique()}")
    log.info(f"  Columns      : {list(df.columns)}")
    missing = df.isnull().sum()
    if missing[missing > 0].any():
        log.warning(f"  Missing vals :\n{missing[missing > 0]}")
    else:
        log.info("  Missing vals : None")
    log.info(f"  Sample:\n{df.head(3).to_string()}")


# ===================================================================
#  MAIN
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("Nepal ERA5-Land Dengue Weather Pipeline")
    log.info("=" * 60)

    # -- Load shapefile ------------------------------------------
    shapefile = CONFIG["shapefile_path"]
    if not os.path.exists(shapefile):
        log.error(f"Shapefile not found: {shapefile}")
        sys.exit(1)

    log.info(f"Loading shapefile: {shapefile}")
    districts = gpd.read_file(shapefile).to_crs("EPSG:4326")

    if CONFIG["district_col"] not in districts.columns:
        candidates = [c for c in districts.columns
                      if any(k in c.lower() for k in ["adm2", "dist", "name"])]
        if candidates:
            CONFIG["district_col"] = candidates[0]
            log.warning(f"  Auto-selected district column: {CONFIG['district_col']}")
        else:
            log.error(f"  District column not found. Columns: {list(districts.columns)}")
            sys.exit(1)

    # Filter to only Kathmandu district
    districts = districts[districts[CONFIG["district_col"]].str.lower() == "kathmandu"]
    
    if districts.empty:
        log.error("Kathmandu district not found in the shapefile.")
        sys.exit(1)
        
    # Update bbox to only download data for Kathmandu (+ small buffer)
    bounds = districts.total_bounds  # [minx(W), miny(S), maxx(E), maxy(N)]
    CONFIG["bbox"] = [
        round(bounds[3] + 0.1, 2),  # North
        round(bounds[0] - 0.1, 2),  # West
        round(bounds[1] - 0.1, 2),  # South
        round(bounds[2] + 0.1, 2)   # East
    ]

    log.info(f"  Loaded {len(districts)} district(s). Bbox updated to {CONFIG['bbox']}.")

    # -- Connect to CDS API --------------------------------------
    log.info("Connecting to CDS API...")
    try:
        client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=CDS_API_KEY,
        )
        log.info("  CDS API connected.")
    except Exception as e:
        log.error(f"CDS connection failed: {e}")
        sys.exit(1)

    # -- Get all (year, month) pairs to process ------------------
    months_list = get_months_to_download(CONFIG["start_year"], CONFIG["end_year"])[:1]
    total       = len(months_list)
    log.info(f"\nTotal months to download: {total} "
             f"({months_list[0][0]}-{months_list[0][1]:02d})")

    # -- Download one month at a time ----------------------------
    all_dfs     = []
    output_path = CONFIG["output_csv"]

    for idx, (year, month) in enumerate(months_list, 1):
        log.info(f"\n[{idx}/{total}] Processing {year}-{month:02d}...")

        # Download
        nc_file = download_era5_month(client, year, month, CONFIG["download_dir"])
        if not nc_file:
            log.warning(f"  Skipping {year}-{month:02d} due to download failure.")
            continue

        # Process month -> daily district rows
        df_month = process_netcdf(nc_file, districts)
        all_dfs.append(df_month)
        log.info(f"  {year}-{month:02d}: {len(df_month):,} rows extracted.")

        # Save running combined CSV after every month (crash-safe)
        df_running = (
            pd.concat(all_dfs, ignore_index=True)
            .sort_values(["date", "district"])
            .reset_index(drop=True)
        )
        df_running.to_csv(output_path, index=False)
        log.info(f"  Progress saved -> {output_path} ({len(df_running):,} total rows)")

        # Polite pause between requests
        if idx < total:
            time.sleep(CONFIG["pause_between_requests"])

    # -- Final combined output -----------------------------------
    if not all_dfs:
        log.error("No data processed. Exiting.")
        sys.exit(1)

    df_final = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["date", "district"])
        .reset_index(drop=True)
    )
    df_final.to_csv(output_path, index=False)
    log.info(f"\nFinal CSV saved: {os.path.abspath(output_path)}")

    validate_output(df_final)
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()