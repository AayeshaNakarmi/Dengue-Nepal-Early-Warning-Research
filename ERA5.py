"""
========================================================
ERA5-Land Automated Pipeline for Nepal Dengue Research
========================================================
Covers   : All 77 districts, 2019-2025
Output   : weather_data/<district>/<year>/daily_weather.csv
           nepal_dengue_weather_daily.csv  (master merged file)
Variables: Temperature (mean/max/min), Heat Index,
           Relative Humidity, Precipitation,
           Soil Moisture, Wind Speed

FIXES APPLIED:
  - Precipitation de-accumulation (diff + clip)
  - Robust resume system (3-level: file / progress tracker / CSV dates)
  - Corrupt NetCDF detection and re-download
  - expver handling fixed
  - rioxarray clip on derived arrays fixed (with fallback)
  - Double clipping removed
  - validate_chunk fixed
  - temp max/min added
  - Master CSV merge at end
  - Skip 10s pause if .nc file already existed (no API call made)
  - DataFrame built via date-merge instead of .values (prevents misalignment)
========================================================

PREREQUISITES:
  pip install cdsapi xarray pandas geopandas rasterstats
              netCDF4 numpy rioxarray tqdm python-dotenv affine

SETUP:
  1. Create CDS account: https://cds.climate.copernicus.eu/
  2. Accept ERA5-Land license on the CDS website
  3. Create a .env file:
       CDS_API_KEY=your-actual-token-here

RUN:
  python ERA5.py

MERGE ONLY (no API key required):
  python ERA5.py --merge-only
"""

import os
import sys
import time
import json
import logging
import warnings
import calendar
import zipfile
import shutil
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import cdsapi

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ===================================================================
#  LOAD API KEY (only required for downloads)
# ===================================================================

def load_cds_api_key() -> str | None:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    return os.getenv("CDS_API_KEY")

# ===================================================================
#  CONFIGURATION
# ===================================================================

CONFIG = {
    "start_year":          2023,
    "end_year":            2025,
    "nepal_bbox":          [30.5, 80.0, 26.3, 88.2],  # N, W, S, E
    "shapefile_path":      "npl_admin2.shp",
    "district_col":        "adm2_name",
    "shared_download_dir": "era5_downloads/nepal_shared",
    "base_output_dir":     "weather_data",
    "progress_file":       "pipeline_progress.json",
    "master_csv":          "nepal_dengue_weather_daily.csv",
    "variables": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "volumetric_soil_water_layer_1",
    ],
    "pause_between_chunks": 10,   # seconds between API chunk requests
    "retry_wait_seconds":   120,  # initial wait on rate limit
    "max_retries":          5,
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
#  RESUME SYSTEM  (3-level)
# ===================================================================

def load_progress() -> dict:
    """Load progress tracker from disk."""
    if os.path.exists(CONFIG["progress_file"]):
        with open(CONFIG["progress_file"], "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """Save progress tracker to disk."""
    with open(CONFIG["progress_file"], "w") as f:
        json.dump(progress, f, indent=2)


def progress_key(district: str, year: int, month: int, chunk_tag: str) -> str:
    return f"{district}__{year}__{month:02d}__{chunk_tag}"


def mark_done(progress: dict, district: str, year: int, month: int, chunk_tag: str):
    key = progress_key(district, year, month, chunk_tag)
    progress[key] = {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "district":     district,
        "year":         year,
        "month":        month,
        "chunk":        chunk_tag,
    }
    save_progress(progress)


def is_done(progress: dict, district: str, year: int, month: int, chunk_tag: str) -> bool:
    return progress_key(district, year, month, chunk_tag) in progress


def get_processed_dates(output_path: str) -> set:
    """Return set of dates already saved in the district CSV."""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["date"])
        return set(df["date"].astype(str).tolist())
    except Exception:
        return set()


def chunk_fully_in_csv(output_path: str, year: int, month: int, day_range: list) -> bool:
    """Return True only if ALL days in this chunk are already in the CSV."""
    processed = get_processed_dates(output_path)
    expected  = [f"{year}-{month:02d}-{d}" for d in day_range]
    missing   = [d for d in expected if d not in processed]
    return len(missing) == 0


# ===================================================================
#  LEVEL 1 — NetCDF FILE VALIDATION
# ===================================================================

def is_valid_netcdf(file_path: str) -> bool:
    """
    Check if a file exists, is large enough, and opens without error.
    Deletes corrupt files so they will be re-downloaded.
    """
    if not os.path.exists(file_path):
        return False

    # Suspiciously small file = probably an error response, not real data
    if os.path.getsize(file_path) < 1_000:
        log.warning(f"    File too small (likely corrupt): {os.path.basename(file_path)} — deleting.")
        os.remove(file_path)
        return False

    try:
        ds = xr.open_dataset(file_path, engine="netcdf4")
        ds.close()
        return True
    except Exception:
        log.warning(f"    Corrupt NetCDF detected — deleting: {os.path.basename(file_path)}")
        try:
            os.remove(file_path)
        except Exception:
            pass
        return False


# ===================================================================
#  STEP 1: DOWNLOAD
# ===================================================================

def ensure_netcdf(file_path: str, download_dir: str):
    """If the downloaded file is a ZIP, extract the .nc inside it."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "rb") as f:
        magic = f.read(4)
    if magic != b"PK\x03\x04":
        return  # not a zip

    log.info(f"    ZIP detected — extracting NetCDF...")
    temp_extracted = None
    with zipfile.ZipFile(file_path, "r") as z:
        nc_files = [n for n in z.namelist() if n.endswith(".nc")]
        if nc_files:
            temp_extracted = os.path.join(download_dir, nc_files[0])
            z.extract(nc_files[0], download_dir)

    if temp_extracted and os.path.abspath(temp_extracted) != os.path.abspath(file_path):
        os.remove(file_path)
        shutil.move(temp_extracted, file_path)
    log.info(f"    Extracted NetCDF successfully.")


def download_era5_chunk(
    client,
    year: int,
    month: int,
    day_range: list,
    chunk_tag: str,
    download_dir: str,
    bbox: list,
) -> str:
    """
    Download ERA5-Land data for a specific day range and bounding box.
    - Skips if a valid file already exists  (Level 1 resume)
    - Retries on rate-limit errors with exponential back-off
    Returns path to the NetCDF file, or None on failure.
    """
    os.makedirs(download_dir, exist_ok=True)
    fname       = f"era5_nepal_{year}_{month:02d}_{chunk_tag}.nc"
    output_file = os.path.join(download_dir, fname)

    # Level 1 — skip if already valid
    if is_valid_netcdf(output_file):
        log.info(f"    {year}-{month:02d} ({chunk_tag}): valid file exists — skipping download.")
        return output_file

    request_params = {
        "product_type": "reanalysis",
        "variable":     CONFIG["variables"],
        "year":         str(year),
        "month":        f"{month:02d}",
        "day":          day_range,
        "time":         [f"{h:02d}:00" for h in range(24)],
        "area":         bbox,
        "data_format":  "netcdf",
    }

    retries = 0
    wait    = CONFIG["retry_wait_seconds"]

    while retries <= CONFIG["max_retries"]:
        try:
            log.info(f"    Downloading {year}-{month:02d} {chunk_tag} (days {day_range[0]}–{day_range[-1]})...")
            client.retrieve("reanalysis-era5-land", request_params, output_file)

            # Windows holds a file lock briefly after download completes.
            # Retry up to 10 times with a 3s wait before giving up.
            for attempt in range(10):
                try:
                    ensure_netcdf(output_file, download_dir)
                    break  # file is accessible, move on
                except PermissionError:
                    log.warning(f"    Windows file lock — waiting 3s (attempt {attempt + 1}/10)...")
                    time.sleep(3)
            else:
                log.error(f"    File still locked after 10 attempts — skipping chunk.")
                return None

            return output_file

        except Exception as e:
            err = str(e).lower()

            if "licence" in err or "license" in err:
                log.error(
                    "\n[ERROR] License not accepted!\n"
                    "Visit: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download\n"
                    "Scroll to bottom, accept Terms of Use, then re-run.\n"
                )
                return None

            if any(k in err for k in ["rate limit", "too many", "429", "cost limit", "403"]):
                retries += 1
                if retries > CONFIG["max_retries"]:
                    log.error(f"    {year}-{month:02d} {chunk_tag}: max retries exceeded.")
                    return None
                log.warning(f"    Rate limit hit — waiting {wait}s (retry {retries}/{CONFIG['max_retries']})...")
                time.sleep(wait)
                wait *= 2
            else:
                log.error(f"    Download failed: {e}")
                return None

    return None


def get_months_to_download(start_year: int, end_year: int):
    """Return (year, month) pairs up to the last fully completed month."""
    now    = datetime.now()
    result = []
    for year in range(start_year, end_year + 1):
        last = 12 if year < now.year else now.month - 1
        for month in range(1, last + 1):
            result.append((year, month))
    return result


# ===================================================================
#  STEP 2: PHYSICS
# ===================================================================

def kelvin_to_celsius(k):
    return k - 273.15


def relative_humidity(t_k, td_k):
    """Magnus formula — returns RH in % clipped to [0, 100]."""
    t  = kelvin_to_celsius(t_k)
    td = kelvin_to_celsius(td_k)
    rh = 100.0 * (
        np.exp((17.625 * td) / (243.04 + td)) /
        np.exp((17.625 * t)  / (243.04 + t))
    )
    return np.clip(rh, 0.0, 100.0)


def heat_index(t_c, rh):
    """
    Steadman's Heat Index (°C).
    Falls back to raw temperature when T < 27°C or RH < 40%.
    """
    T, R = t_c, rh
    HI = (
        -8.78469475556
        + 1.61139411    * T
        + 2.33854883889 * R
        - 0.14611605    * T * R
        - 0.012308094   * T ** 2
        - 0.0164248277778 * R ** 2
        + 0.002211732   * T ** 2 * R
        + 0.00072546    * T * R ** 2
        - 0.000003582   * T ** 2 * R ** 2
    )
    return np.where((t_c < 27) | (rh < 40), t_c, HI)


def wind_speed(u, v):
    return np.sqrt(u ** 2 + v ** 2)


# ===================================================================
#  STEP 3: DATA QUALITY CHECK
# ===================================================================

def validate_chunk(df: pd.DataFrame, expected_dates: list) -> bool:
    """Run basic sanity checks and log any issues. Returns True if clean."""
    issues = []

    actual   = set(df["date"].astype(str).tolist())
    expected = set(expected_dates)
    missing  = expected - actual
    if missing:
        issues.append(f"Missing dates: {sorted(missing)}")

    if (df["temp_mean_celsius"] > 60).any() or (df["temp_mean_celsius"] < -30).any():
        issues.append("Unrealistic temperature values detected!")

    if (df["precipitation_mm"] < 0).any():
        issues.append("Negative precipitation values detected!")

    if (df["relative_humidity_pct"] > 100).any() or (df["relative_humidity_pct"] < 0).any():
        issues.append("Humidity out of 0–100% range!")

    pct_null = df.isnull().mean() * 100
    for col, pct in pct_null.items():
        if pct > 10:
            issues.append(f"High missing data in '{col}': {pct:.1f}%")

    if issues:
        for issue in issues:
            log.warning(f"    [QC] {issue}")
        return False

    log.info("    [QC] All checks passed.")
    return True


# ===================================================================
#  STEP 4: SPATIAL CLIPPING HELPER
# ===================================================================

def spatial_clip_and_mean(da: xr.DataArray, gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Clip a DataArray to the district polygon and return spatial mean.
    Falls back to a simple bounding-box average if rioxarray fails.
    """
    try:
        import rioxarray  # noqa
        da_rio = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        da_rio = da_rio.rio.write_crs("EPSG:4326")
        clipped = da_rio.rio.clip(gdf.geometry, gdf.crs, all_touched=True)
        return clipped.mean(dim=["latitude", "longitude"])
    except Exception as e:
        log.warning(f"    rioxarray clip failed ({e}) — using bbox spatial mean.")
        bounds = gdf.total_bounds  # minX, minY, maxX, maxY
        return da.sel(
            latitude=slice(bounds[3], bounds[1]),
            longitude=slice(bounds[0], bounds[2]),
        ).mean(dim=["latitude", "longitude"])


# ===================================================================
#  STEP 5: PROCESS ONE NetCDF CHUNK → DAILY DISTRICT ROWS
# ===================================================================

def process_netcdf(nc_path: str, district_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Load a NetCDF file, apply physics, clip to district, resample to daily.
    Returns a DataFrame with one row per day.
    """
    # Buffer very small districts so at least some grid cells are captured
    gdf = district_gdf.copy()
    try:
        area_km2 = float(gdf.to_crs("EPSG:32645").area.iloc[0]) / 1e6
        if area_km2 < 200:
            log.warning(f"    Small district ({area_km2:.1f} km²) — applying 0.05° buffer.")
            gdf["geometry"] = gdf.buffer(0.05)
    except Exception:
        pass

    # Open NetCDF — try netcdf4 then h5netcdf
    ds = None
    for engine in ["netcdf4", "h5netcdf"]:
        try:
            ds = xr.open_dataset(nc_path, engine=engine)
            break
        except Exception:
            continue

    if ds is None:
        log.error(f"    Cannot open NetCDF: {nc_path}")
        return pd.DataFrame()

    try:
        # Fix expver dimension if present (just take first slice)
        if "expver" in ds.dims:
            ds = ds.isel(expver=0, drop=True)

        # Identify time dimension name
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"

        # ── Physics ──────────────────────────────────────────────
        t2m_k = ds["t2m"]
        d2m_k = ds["d2m"]
        u10   = ds["u10"]
        v10   = ds["v10"]
        swvl  = ds["swvl1"]

        tc_da = kelvin_to_celsius(t2m_k)
        rh_da = xr.DataArray(
            relative_humidity(t2m_k.values, d2m_k.values),
            coords=t2m_k.coords, dims=t2m_k.dims,
        )
        ws_da = xr.DataArray(
            wind_speed(u10.values, v10.values),
            coords=u10.coords, dims=u10.dims,
        )
        hi_da = xr.DataArray(
            heat_index(tc_da.values, rh_da.values),
            coords=tc_da.coords, dims=tc_da.dims,
        )

        # ── Fix Precipitation (ERA5 tp is cumulative per forecast step) ──
        tp_raw    = ds["tp"] * 1000           # metres → mm
        tp_hourly = tp_raw.diff(dim=time_dim).clip(min=0)
        # Note: first timestep is dropped by diff — acceptable (1 hour lost per chunk)

        # ── Spatial clip + daily resample for each variable ──────
        def to_daily_df(da, col_name, method="mean"):
            """
            Clip to district, resample to daily, return a clean
            two-column DataFrame [date, col_name].
            Each variable is converted independently so different
            array lengths (e.g. tp after diff) never misalign rows.
            """
            spatial = spatial_clip_and_mean(da, gdf)
            if method == "sum":
                resampled = spatial.resample({time_dim: "1D"}).sum()
            elif method == "max":
                resampled = spatial.resample({time_dim: "1D"}).max()
            elif method == "min":
                resampled = spatial.resample({time_dim: "1D"}).min()
            else:
                resampled = spatial.resample({time_dim: "1D"}).mean()

            tmp = resampled.to_dataframe(name=col_name).reset_index()
            for c in ["valid_time", "time"]:
                if c in tmp.columns:
                    tmp = tmp.rename(columns={c: "date"})
                    break
            tmp["date"] = pd.to_datetime(tmp["date"].astype(str)).dt.strftime("%Y-%m-%d")
            return tmp[["date", col_name]]

        # ── Build DataFrame — merge on date so lengths never mismatch ──
        df = to_daily_df(tc_da,     "temp_mean_celsius",     "mean")
        df = df.merge(to_daily_df(tc_da,     "temp_max_celsius",      "max"),  on="date", how="left")
        df = df.merge(to_daily_df(tc_da,     "temp_min_celsius",      "min"),  on="date", how="left")
        df = df.merge(to_daily_df(hi_da,     "heat_index_celsius",    "mean"), on="date", how="left")
        df = df.merge(to_daily_df(rh_da,     "relative_humidity_pct", "mean"), on="date", how="left")
        df = df.merge(to_daily_df(tp_hourly, "precipitation_mm",      "sum"),  on="date", how="left")
        df = df.merge(to_daily_df(ws_da,     "wind_speed_ms",         "mean"), on="date", how="left")
        df = df.merge(to_daily_df(swvl,      "soil_moisture_vol",     "mean"), on="date", how="left")

        df["district"] = district_gdf.iloc[0][CONFIG["district_col"]]

        ds.close()

        # Keep only needed columns in a clean order
        cols = [
            "date", "district",
            "temp_mean_celsius", "temp_max_celsius", "temp_min_celsius",
            "heat_index_celsius", "relative_humidity_pct",
            "precipitation_mm", "wind_speed_ms", "soil_moisture_vol",
        ]
        return df[cols]

    except Exception as e:
        log.error(f"    Processing error in {os.path.basename(nc_path)}: {e}")
        if ds is not None:
            ds.close()
        return pd.DataFrame()


# ===================================================================
#  STEP 6: MERGE ALL DISTRICT CSVs INTO MASTER FILE
# ===================================================================

def _infer_district_from_path(p: Path) -> str:
    # Expected: weather_data/<district>/<year>/daily_weather.csv
    if p.parent.name.isdigit() and len(p.parent.name) == 4:
        return p.parent.parent.name
    return p.parent.name


def _iter_daily_weather_csvs(
    base_output_dir: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[Path]:
    base = Path(base_output_dir)
    if not base.exists():
        return []

    paths: list[Path] = []
    for district_dir in base.iterdir():
        if not district_dir.is_dir():
            continue

        direct = district_dir / "daily_weather.csv"
        if direct.exists():
            paths.append(direct)

        for year_dir in district_dir.iterdir():
            if not year_dir.is_dir():
                continue
            if not (year_dir.name.isdigit() and len(year_dir.name) == 4):
                continue

            year = int(year_dir.name)
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue

            p = year_dir / "daily_weather.csv"
            if p.exists():
                paths.append(p)

    paths.sort(key=lambda x: (str(_infer_district_from_path(x)), str(x.parent.name), str(x)))
    return paths


def merge_weather_dir_to_master(
    base_output_dir: str,
    master_csv_path: str,
    start_year: int | None = None,
    end_year: int | None = None,
):
    log.info("\nMerging all districts into master CSV...")
    paths = _iter_daily_weather_csvs(base_output_dir, start_year=start_year, end_year=end_year)

    if not paths:
        log.warning(f"  No data found under: {base_output_dir}")
        return

    all_dfs: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "district" not in df.columns:
                df["district"] = _infer_district_from_path(p)
            all_dfs.append(df)
        except Exception as e:
            log.warning(f"  Could not read {p}: {e}")

    if not all_dfs:
        log.warning("  No readable CSVs found to merge.")
        return

    master = (
        pd.concat(all_dfs, ignore_index=True)
        .drop_duplicates(subset=["date", "district"])
        .sort_values(["district", "date"])
        .reset_index(drop=True)
    )
    master.to_csv(master_csv_path, index=False)
    log.info(f"  Master CSV saved: {master_csv_path}  ({len(master):,} rows)")


def merge_all_to_master(district_names: list):
    # Keep signature for backwards compatibility; merge is now directory-driven.
    _ = district_names
    merge_weather_dir_to_master(CONFIG["base_output_dir"], CONFIG["master_csv"])


# ===================================================================
#  MAIN
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("Nepal ERA5-Land District Pipeline  (2019–2025)")
    log.info("=" * 60)

    # ── Load shapefile ───────────────────────────────────────────
    shapefile = CONFIG["shapefile_path"]
    if not os.path.exists(shapefile):
        log.error(f"Shapefile not found: {shapefile}")
        sys.exit(1)

    log.info(f"Loading shapefile: {shapefile}")
    all_districts = gpd.read_file(shapefile).to_crs("EPSG:4326")

    # Auto-detect district column
    district_col = CONFIG["district_col"]
    if district_col not in all_districts.columns:
        candidates = [
            c for c in all_districts.columns
            if any(k in c.lower() for k in ["adm2", "dist", "name"])
        ]
        if candidates:
            district_col = candidates[0]
            CONFIG["district_col"] = district_col
            log.warning(f"  Auto-selected district column: '{district_col}'")
        else:
            log.error("  Cannot find district column in shapefile.")
            sys.exit(1)

    # Normalize district names (lowercase, strip whitespace)
    all_districts[district_col] = (
        all_districts[district_col].astype(str).str.strip().str.lower().str.replace("_", " ")
    )
    district_names = sorted(all_districts[district_col].unique().tolist())
    log.info(f"  Found {len(district_names)} districts.")
    log.info(f"  Sample names: {district_names[:5]}")

    # ── Load resume progress ─────────────────────────────────────
    progress = load_progress()
    log.info(f"  Resume tracker: {len(progress)} chunks already completed.")

    # ── Connect to CDS API ───────────────────────────────────────
    cds_api_key = load_cds_api_key()
    if not cds_api_key:
        print(
            "\n[ERROR] CDS_API_KEY not found!\n"
            "Create a .env file with: CDS_API_KEY=your-token\n"
            "Get token at: https://cds.climate.copernicus.eu/profile\n"
        )
        sys.exit(1)

    log.info("Connecting to CDS API...")
    try:
        client = cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=cds_api_key,
        )
        log.info("  CDS API connected.")
    except Exception as e:
        log.error(f"CDS connection failed: {e}")
        sys.exit(1)

    # ── Main loop: Year → Month → Chunk → Districts ──────────────
    for year in range(CONFIG["start_year"], CONFIG["end_year"] + 1):
        log.info(f"\n{'=' * 60}")
        log.info(f"  YEAR: {year}")
        log.info(f"{'=' * 60}")

        for _, month in get_months_to_download(year, year):
            log.info(f"\n  Month: {year}-{month:02d}")

            _, num_days = calendar.monthrange(year, month)
            all_days    = [f"{d:02d}" for d in range(1, num_days + 1)]
            day_chunks  = [all_days[i:i + 10] for i in range(0, len(all_days), 10)]

            for c_idx, day_range in enumerate(day_chunks, 1):
                chunk_tag = f"part{c_idx}"
                log.info(f"\n  [Chunk {c_idx}/{len(day_chunks)}] Days {day_range[0]}–{day_range[-1]}")

                # Check if file already exists BEFORE download attempt
                # so we know whether we actually hit the API or not
                expected_nc = os.path.join(
                    CONFIG["shared_download_dir"],
                    f"era5_nepal_{year}_{month:02d}_{chunk_tag}.nc"
                )
                file_already_existed = is_valid_netcdf(expected_nc)

                # ── LEVEL 1: Download (shared Nepal-wide file) ───
                nc_file = download_era5_chunk(
                    client, year, month, day_range, chunk_tag,
                    CONFIG["shared_download_dir"],
                    bbox=CONFIG["nepal_bbox"],
                )
                if not nc_file:
                    log.error(f"  Skipping chunk {chunk_tag} — download failed.")
                    continue

                # ── LEVEL 2 & 3: Per-district processing ─────────
                for d_name in district_names:
                    dist_folder    = d_name.replace(" ", "_")
                    year_out_dir   = os.path.join(CONFIG["base_output_dir"], dist_folder, str(year))
                    output_path    = os.path.join(year_out_dir, "daily_weather.csv")

                    # Level 2 — check progress tracker (fastest)
                    if is_done(progress, d_name, year, month, chunk_tag):
                        continue

                    # Level 3 — check actual CSV dates (safety net)
                    if chunk_fully_in_csv(output_path, year, month, day_range):
                        log.info(f"    {d_name}: all dates in CSV — marking done.")
                        mark_done(progress, d_name, year, month, chunk_tag)
                        continue

                    log.info(f"    Processing: {d_name}...")
                    district_gdf = all_districts[all_districts[district_col] == d_name]

                    df_chunk = process_netcdf(nc_file, district_gdf)

                    if df_chunk.empty:
                        log.warning(f"    {d_name}: empty result — will retry next run.")
                        continue

                    # Quality check
                    expected_dates = [f"{year}-{month:02d}-{d}" for d in day_range]
                    # tp uses diff so it may have one fewer day — allow for that
                    validate_chunk(df_chunk, expected_dates)

                    # Save to district CSV
                    os.makedirs(year_out_dir, exist_ok=True)
                    if os.path.exists(output_path):
                        existing   = pd.read_csv(output_path)
                        df_chunk   = pd.concat([existing, df_chunk], ignore_index=True)
                        df_chunk   = (
                            df_chunk
                            .drop_duplicates(subset=["date", "district"])
                            .sort_values("date")
                            .reset_index(drop=True)
                        )
                    df_chunk.to_csv(output_path, index=False)

                    # Mark as done
                    mark_done(progress, d_name, year, month, chunk_tag)
                    log.info(f"    {d_name}: saved → {output_path}")

                # Only pause if we actually hit the API (file didn't exist before)
                if file_already_existed:
                    log.info(f"  File already existed — skipping pause.")
                else:
                    log.info(f"  Pausing {CONFIG['pause_between_chunks']}s before next chunk...")
                    time.sleep(CONFIG["pause_between_chunks"])

    # ── Merge everything into master CSV ─────────────────────────
    merge_all_to_master(district_names)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5-Land Nepal district pipeline + merge utility")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip downloads and only merge existing weather CSVs into the master file.",
    )
    parser.add_argument(
        "--base-output-dir",
        default=CONFIG["base_output_dir"],
        help="Directory containing per-district weather outputs (default: weather_data).",
    )
    parser.add_argument(
        "--master-csv",
        default=CONFIG["master_csv"],
        help="Path to write the merged master CSV (default: nepal_dengue_weather_daily.csv).",
    )
    parser.add_argument("--start-year", type=int, default=None, help="Optional year filter for merge-only.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional year filter for merge-only.")
    args = parser.parse_args()

    CONFIG["base_output_dir"] = args.base_output_dir
    CONFIG["master_csv"] = args.master_csv

    if args.merge_only:
        merge_weather_dir_to_master(
            CONFIG["base_output_dir"],
            CONFIG["master_csv"],
            start_year=args.start_year,
            end_year=args.end_year,
        )
    else:
        main()
