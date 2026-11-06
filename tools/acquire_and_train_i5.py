#!/usr/bin/env python3
"""
I5 Idea-1 data acquisition + training pipeline — fully automated.

Manual steps:
  1. Enter NASA Earthdata username/password when prompted
  2. Enter Copernicus username/password when prompted

Everything else is automatic:
  - Builds 1 km grid over southern Alberta / SW Saskatchewan mixed-grass prairie
  - Pulls Daymet climate (authenticated Earthdata)
  - Pulls GBIF invasion occurrence (public API)
  - Downloads NASADEM elevation tiles (authenticated Earthdata)
  - Downloads ESA WorldCover 2021 land-cover tiles (authenticated Copernicus)
  - Pulls NBAC burned-area labels via CWFIS WFS (best-effort public endpoints)
  - Samples all rasters at cell centroids → feature matrix
  - Builds train_i5 binary corpus (corpus.hdr + *.X.bin / *.L.bin / *.S.bin / *.A.bin)
  - Optionally runs train_i5 and exports .lancius model

Requirements:
  pip install requests rasterio numpy
"""

import os
import sys
import csv
import json
import math
import struct
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

# ---------------------------------------------------------------------------
# 0. Credentials — from env vars, creds file, or interactive prompt
# ---------------------------------------------------------------------------

def load_creds_file():
    path = Path.home() / ".lancius_earthdata"
    if not path.exists():
        return {}
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def prompt_credentials():
    env = {
        "earth_user": os.environ.get("EARTHDATA_USERNAME", ""),
        "earth_pass": os.environ.get("EARTHDATA_PASSWORD", ""),
        "cop_user": os.environ.get("COPERNICUS_USERNAME", ""),
        "cop_pass": os.environ.get("COPERNICUS_PASSWORD", ""),
    }
    file_creds = load_creds_file()
    for k in env:
        if not env[k] and k in file_creds:
            env[k] = file_creds[k]

    missing = [k for k in ("earth_user", "earth_pass") if not env[k]]
    if missing:
        print("=" * 64)
        print("  I5 / Idea-1 automated acquisition pipeline")
        print("  Missing credentials:", ", ".join(missing))
        print("  Set them as env vars (EARTHDATA_USERNAME, EARTHDATA_PASSWORD,")
        print("   COPERNICUS_USERNAME, COPERNICUS_PASSWORD) or in ~/.lancius_earthdata")
        print("=" * 64)
        sys.exit(1)

    if not env["cop_user"] or not env["cop_pass"]:
        print("WARNING: no Copernicus credentials — WorldCover will be skipped.")

    print("Credentials loaded from environment / creds file.")
    return env


CREDS = prompt_credentials()

SESSION_E = requests.Session()
SESSION_E.auth = (CREDS["earth_user"], CREDS["earth_pass"])
SESSION_E.headers.update({"User-Agent": "lancius-i5-acquire/1.0 (research)"})

SESSION_C = requests.Session()
if CREDS["cop_user"] and CREDS["cop_pass"]:
    SESSION_C.auth = (CREDS["cop_user"], CREDS["cop_pass"])
SESSION_C.headers.update({"User-Agent": "lancius-i5-acquire/1.0 (research)"})

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

OUTDIR = Path("data_i5_real")
# Pilot grid: small area over southern AB mixed-grass prairie (~10 km cells)
# Expand GRID_* and shrink CELL_DEG for full coverage once credentials work.
GRID_LON_MIN, GRID_LON_MAX = -112.0, -110.0
GRID_LAT_MIN, GRID_LAT_MAX = 49.5, 50.5
CELL_DEG = 0.1  # ~10 km pilot; set to 1.0/111.0 for ~1 km full coverage
FEAT_YEARS = list(range(2015, 2024))   # features 2015..2023
LABEL_YEARS = list(range(2016, 2025))  # labels 2016..2024
NCLASS = 2

GBIF_TAXA = {
    "Bromus_tectorum": None,      # resolved below via species match
    "Bromus_japonicus": None,
    "Ventenata_dubia": 2710914,
    "Aegilops_cylindrica": 2711190,
}

DAYMET_VARS = ["tmax", "tmin", "prcp", "srad", "vp", "dayl", "swe"]

OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "nasadem").mkdir(exist_ok=True)
(OUTDIR / "worldcover").mkdir(exist_ok=True)
(OUTDIR / "nbac").mkdir(exist_ok=True)

print(f"\nOutput directory: {OUTDIR.resolve()}")
print(f"Grid: {GRID_LON_MIN}..{GRID_LON_MAX} x {GRID_LAT_MIN}..{GRID_LAT_MAX} (~1 km cells)")

# ---------------------------------------------------------------------------
# 2. Build 1 km cell grid
# ---------------------------------------------------------------------------

def build_grid():
    lons = []
    lon = GRID_LON_MIN
    while lon <= GRID_LON_MAX + 1e-9:
        lons.append(round(lon, 6))
        lon += CELL_DEG
    lats = []
    lat = GRID_LAT_MIN
    while lat <= GRID_LAT_MAX + 1e-9:
        lats.append(round(lat, 6))
        lat += CELL_DEG
    cells = []
    cid = 0
    for lat in lats:
        for lon in lons:
            cells.append({"cell_id": cid, "lon": lon, "lat": lat})
            cid += 1
    print(f"Grid cells: {len(cells)}")
    return cells

CELLS = build_grid()

# ---------------------------------------------------------------------------
# 3. Resolve GBIF taxon keys (public, no auth)
# ---------------------------------------------------------------------------

def resolve_gbif_keys():
    for name in GBIF_TAXA:
        if GBIF_TAXA[name] is not None:
            continue
        try:
            r = requests.get(
                "https://api.gbif.org/v1/species/match",
                params={"name": name.replace("_", " ")},
                timeout=30,
            )
            r.raise_for_status()
            key = r.json().get("usageKey")
            if key:
                GBIF_TAXA[name] = key
                print(f"GBIF {name}: key {key}")
            else:
                print(f"GBIF {name}: no key found")
        except Exception as e:
            print(f"GBIF {name}: lookup failed — {e}")
        time.sleep(0.2)

resolve_gbif_keys()

# ---------------------------------------------------------------------------
# 4. Fetch GBIF invasion counts per cell-year (public API)
# ---------------------------------------------------------------------------

def gbif_count_in_cell(taxon_key, lat, lon, year, radius_deg=0.0045):  # ~500 m
    if taxon_key is None:
        return 0
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "taxon_key": taxon_key,
        "limit": 0,
        "has_coordinate": "true",
        "year": year,
        "decimalLatitude": f"{lat-radius_deg},{lat+radius_deg}",
        "decimalLongitude": f"{lon-radius_deg},{lon+radius_deg}",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("count", 0)
    except Exception:
        return 0

print("\nFetching GBIF invasion counts (this may take a while)...")
GBIF_CACHE = {}  # (taxon, lat, lon, year) -> count

def get_gbif_count(taxon_name, lat, lon, year):
    key = GBIF_TAXA[taxon_name]
    cache_key = (taxon_name, round(lat,4), round(lon,4), year)
    if cache_key in GBIF_CACHE:
        return GBIF_CACHE[cache_key]
    c = gbif_count_in_cell(key, lat, lon, year)
    GBIF_CACHE[cache_key] = c
    time.sleep(0.15)  # polite to GBIF
    return c

# ---------------------------------------------------------------------------
# 5. Fetch Daymet climate per cell-year (Earthdata authenticated)
# ---------------------------------------------------------------------------

def fetch_daymet(lat, lon, year):
    url = ("https://daymet.ornl.gov/single-pixel/api/data")
    params = {
        "lat": lat,
        "lon": lon,
        "vars": ",".join(DAYMET_VARS),
        "startYear": year,
        "endYear": year,
    }
    try:
        r = SESSION_E.get(url, params=params, timeout=90)
        if r.status_code == 401:
            print("  !! Earthdata 401 — check credentials")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  !! Daymet fail ({lat},{lon},{year}): {e}")
        return None

DAYMET_CACHE = {}

def get_daymet(lat, lon, year):
    key = (round(lat,4), round(lon,4), year)
    if key in DAYMET_CACHE:
        return DAYMET_CACHE[key]
    dm = fetch_daymet(lat, lon, year)
    DAYMET_CACHE[key] = dm
    time.sleep(0.1)
    return dm

# ---------------------------------------------------------------------------
# 6. NASADEM tile download (Earthdata authenticated)
# ---------------------------------------------------------------------------

NASADEM_TILES = []
for lat in range(int(GRID_LAT_MIN), int(GRID_LAT_MAX) + 2):
    for lon in range(int(GRID_LON_MIN), int(GRID_LON_MAX) + 2):
        lat_str = f"N{lat:02d}"
        lon_str = f"W{abs(lon):03d}"
        NASADEM_TILES.append((lat_str, lon_str))

print(f"\nNASADEM tiles needed: {len(NASADEM_TILES)}")

def download_nasadem(lat_str, lon_str):
    out = OUTDIR / "nasadem" / f"NASADEM_HGGT_{lat_str}{lon_str}.hgt"
    if out.exists() and out.stat().st_size > 1000:
        return out
    # Try Earthdata direct URL patterns
    candidates = [
        f"https://e4ftl01.cr.usgs.gov/NASADEM/NASADEM_HGGT.001/{lat_str}{lon_str}.zip",
        f"https://e4ftl01.cr.usgs.gov/NASADEM/NASADEM_HGGT.001/{datetime.now().year}.{datetime.now().month:02d}.{datetime.now().day:02d}/NASADEM_HGGT_N{lat_str[1:] if lat_str.startswith('N') else lat_str}S{abs(int(lon_str[1:])):03d}.zip",
    ]
    # Simplified: try the known USGS pattern
    url = f"https://e4ftl01.cr.usgs.gov/NASADEM/NASADEM_HGGT.001/{lat_str}{lon_str}.zip"
    try:
        r = SESSION_E.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            # zip may contain .hgt; for simplicity store raw if it's hgt, else skip
            out.write_bytes(r.content)
            return out
    except Exception:
        pass
    # Fallback: direct .hgt (older USGS pattern)
    url2 = f"https://e4ftl01.cr.usgs.gov/NASADEM/NASADEM_HGGT.001/{lat_str}{lon_str}.hgt"
    try:
        r = SESSION_E.get(url2, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            out.write_bytes(r.content)
            return out
    except Exception as e:
        print(f"  !! NASADEM {lat_str}{lon_str}: {e}")
    return None

print("\nDownloading NASADEM tiles (may take several minutes)...")
NASADEM_PATHS = []
for lat_s, lon_s in NASADEM_TILES:
    p = download_nasadem(lat_s, lon_s)
    if p:
        NASADEM_PATHS.append(p)

# ---------------------------------------------------------------------------
# 7. ESA WorldCover 2021 download (Copernicus authenticated)
# ---------------------------------------------------------------------------

# WorldCover 100x100 km tiles: http://maps-elg.esa.int/worldcover/2021/v100/2021/ESA_WorldCover_10m_2021_v100_<tile>.tif
# Tile naming like N49W113 etc.

def worldcover_tile_name(lat, lon):
    lat_s = "N" if lat >= 0 else "S"
    lon_s = "W" if lon < 0 else "E"
    return f"{lat_s}{abs(lat):02d}{lon_s}{abs(lon):03d}"

WORLDCOVER_TILES = []
for lat in range(int(GRID_LAT_MIN), int(GRID_LAT_MAX) + 2):
    for lon in range(int(GRID_LON_MIN), int(GRID_LON_MAX) + 2):
        WORLDCOVER_TILES.append((lat, lon))

print(f"\nWorldCover tiles needed: {len(WORLDCOVER_TILES)}")

def download_worldcover(lat, lon):
    tile = worldcover_tile_name(lat, lon)
    out = OUTDIR / "worldcover" / f"ESA_WorldCover_10m_2021_v100_{tile}.tif"
    if out.exists() and out.stat().st_size > 1000:
        return out
    # ESA direct tile server (public, may work without auth)
    url = f"https://maps-elg.esa.int/worldcover/2021/v100/2021/ESA_WorldCover_10m_2021_v100_{tile}.tif"
    try:
        r = SESSION_C.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            out.write_bytes(r.content)
            return out
    except Exception as e:
        print(f"  !! WorldCover {tile}: {e}")
    return None

print("\nDownloading WorldCover tiles (may take several minutes)...")
WORLDCOVER_PATHS = []
for lat, lon in WORLDCOVER_TILES:
    p = download_worldcover(lat, lon)
    if p:
        WORLDCOVER_PATHS.append(p)

# ---------------------------------------------------------------------------
# 8. NBAC burned-area labels via CWFIS WFS (best-effort)
# ---------------------------------------------------------------------------

# Try common CWFIS WFS endpoints for NBAC burned area
NBAC_WFS_CANDIDATES = [
    "https://cwfis.cfs.nrcan.gc.ca/datamart/wfs/NBAC",
    "https://cwfis.cfs.nrcan.gc.ca/wfs/NBAC",
    "https://maps.cfs.nrcan.gc.ca/wfs/NBAC",
    "https://cwfis.cfs.nrcan.gc.ca/datamart/wfs",
]

def test_wfs(url):
    try:
        r = requests.get(url, params={"service":"WFS","request":"GetCapabilities"}, timeout=30)
        return r.status_code == 200 and "WFS" in r.text[:1000]
    except Exception:
        return False

NBAC_WFS = None
for cand in NBAC_WFS_CANDIDATES:
    if test_wfs(cand):
        NBAC_WFS = cand
        print(f"NBAC WFS found: {cand}")
        break
if NBAC_WFS is None:
    print("WARNING: no public NBAC WFS endpoint found. Labels will be all zeros (placeholder).")

def get_nbac_burn_fraction(lat, lon, year):
    """Return burn fraction [0,1] for cell at (lat,lon) in year, or -1 if unknown."""
    if NBAC_WFS is None:
        return -1.0
    # Try a simple WFS GetFeature with bbox filter around the cell
    try:
        cell_deg = CELL_DEG
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typename": "nbac:burned_area",
            "srsname": "EPSG:4326",
            "bbox": f"{lon-cell_deg},{lat-cell_deg},{lon+cell_deg},{lat+cell_deg},EPSG:4326",
            "outputFormat": "application/json",
        }
        r = requests.get(NBAC_WFS, params=params, timeout=60)
        if r.status_code != 200:
            return -1.0
        data = r.json()
        # Sum burned area / cell area very roughly
        features = data.get("features", [])
        if not features:
            return 0.0
        # Heuristic: if any feature intersects, mark as burned (fraction unknown)
        return 1.0  # conservative: presence = burned
    except Exception:
        return -1.0

print("\nFetching NBAC burn labels (may take a while)...")
NBAC_CACHE = {}
def get_label(lat, lon, label_year):
    key = (round(lat,4), round(lon,4), label_year)
    if key in NBAC_CACHE:
        return NBAC_CACHE[key]
    frac = get_nbac_burn_fraction(lat, lon, label_year)
    NBAC_CACHE[key] = frac
    time.sleep(0.2)
    return frac

# ---------------------------------------------------------------------------
# 9. Raster sampling helpers (rasterio)
# ---------------------------------------------------------------------------

try:
    import rasterio
    from rasterio.warp import transform
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("WARNING: rasterio not installed. Install with: pip install rasterio")
    print("  Elevation and land-cover features will be zeroed.")

NASADEM_SRCS = []
WORLDCOVER_SRCS = []
if RASTERIO_OK:
    for p in NASADEM_PATHS:
        try:
            NASADEM_SRCS.append(rasterio.open(str(p)))
        except Exception as e:
            print(f"  !! could not open NASADEM {p}: {e}")
    for p in WORLDCOVER_PATHS:
        try:
            WORLDCOVER_SRCS.append(rasterio.open(str(p)))
        except Exception as e:
            print(f"  !! could not open WorldCover {p}: {e}")

def sample_raster(srcs, lon, lat):
    """Sample first matching raster that contains the point."""
    if not srcs:
        return 0.0
    for src in srcs:
        try:
            if (src.bounds.left <= lon <= src.bounds.right and
                src.bounds.bottom <= lat <= src.bounds.top):
                row, col = src.index(lon, lat)
                val = src.read(1)[row, col]
                if src.nodata is not None and val == src.nodata:
                    continue
                return float(val)
        except Exception:
            continue
    return 0.0

def compute_slope_from_nasadem(lon, lat):
    # Simple placeholder: slope from elevation difference is complex without
    # full DEM; return 0 unless we implement proper slope. For now return 0.
    return 0.0

# ---------------------------------------------------------------------------
# 10. Build feature matrix (one row per cell-year)
# ---------------------------------------------------------------------------

FEATURE_ORDER = [
    # static (sampled once per cell)
    "elevation", "slope", "tree_frac", "shrub_frac", "grass_frac", "crop_frac",
    "built_frac", "bare_frac", "water_frac", "wetland_frac", "rangeland_frac",
    "built_5km_frac", "dist_built_km",
    # climate (per cell-year)
    "c_tmax_mean", "c_tmin_mean", "c_tavg_mean", "c_prcp_mean", "c_srad_mean",
    "c_vp_mean", "c_daymet_ok",
    # modis phenology — placeholder zeros until MODIS creds added
    "m_ndvi_spring", "m_evi_spring", "m_ndvi_summer", "m_evi_summer",
    "m_ndvi_fall", "m_evi_fall", "m_ndvi_max", "m_evi_max",
    "m_ndvi_amp", "m_evi_amp", "m_ndvi_season", "m_evi_season",
    "m_greenup", "m_senescence", "m_length_of_season", "m_mid_greenup",
    "m_mid_senescence",
    # invasion (GBIF counts per cell-year)
    "i_count_bt", "i_count_bj", "i_count_vd", "i_count_ac",
    "i_count_all",
    # fire history (placeholder zeros — would need multi-year NBAC)
    "fh_fire_t", "fh_fire_t1", "fh_fire_t2", "fh_fire_count_3yr",
    "fh_fire_years_since", "fh_fire_density_3x3", "fh_fire_density_5x5",
]

STATIC_FEATS = ["elevation","slope","tree_frac","shrub_frac","grass_frac","crop_frac",
                "built_frac","bare_frac","water_frac","wetland_frac","rangeland_frac",
                "built_5km_frac","dist_built_km"]
CLIMATE_FEATS = ["c_tmax_mean","c_tmin_mean","c_tavg_mean","c_prcp_mean","c_srad_mean",
                 "c_vp_mean","c_daymet_ok"]
MODIS_FEATS = [f for f in FEATURE_ORDER if f.startswith("m_")]
INVASION_FEATS = [f for f in FEATURE_ORDER if f.startswith("i_")]
FIRE_FEATS = [f for f in FEATURE_ORDER if f.startswith("fh_")]

print("\nBuilding feature matrix (this is the long part)...")
rows = []
total = len(CELLS) * len(FEAT_YEARS)
done = 0

# Pre-sample static features per cell
static_cache = {}
for cell in CELLS:
    lon, lat = cell["lon"], cell["lat"]
    elev = sample_raster(NASADEM_SRCS, lon, lat)
    slope = compute_slope_from_nasadem(lon, lat)
    # WorldCover class fractions within cell (1 km) — approximate by center pixel class
    wc = sample_raster(WORLDCOVER_SRCS, lon, lat)
    # WorldCover classes: 10=trees,20=shrub,30=grassland,40=cropland,50=built,60=bare,70=water,80=wetland,90=moss,95=mangrove
    tree_frac = 1.0 if wc == 10 else 0.0
    shrub_frac = 1.0 if wc == 20 else 0.0
    grass_frac = 1.0 if wc == 30 else 0.0
    crop_frac = 1.0 if wc == 40 else 0.0
    built_frac = 1.0 if wc == 50 else 0.0
    bare_frac = 1.0 if wc == 60 else 0.0
    water_frac = 1.0 if wc == 70 else 0.0
    wetland_frac = 1.0 if wc == 80 else 0.0
    rangeland_frac = 1.0 if wc in (20,30,40,60) else 0.0
    built_5km_frac = built_frac  # simplified
    dist_built_km = 0.0 if built_frac > 0 else 5.0  # placeholder
    static_cache[cell["cell_id"]] = {
        "elevation": elev, "slope": slope,
        "tree_frac": tree_frac, "shrub_frac": shrub_frac, "grass_frac": grass_frac,
        "crop_frac": crop_frac, "built_frac": built_frac, "bare_frac": bare_frac,
        "water_frac": water_frac, "wetland_frac": wetland_frac,
        "rangeland_frac": rangeland_frac, "built_5km_frac": built_5km_frac,
        "dist_built_km": dist_built_km,
    }

for cell in CELLS:
    lon, lat = cell["lon"], cell["lat"]
    stat = static_cache[cell["cell_id"]]
    for year in FEAT_YEARS:
        row = {"cell_id": cell["cell_id"], "year": year, "lat": lat, "lon": lon,
               "label": 0}
        # Static
        row.update(stat)
        # Climate
        dm = get_daymet(lat, lon, year)
        if dm:
            try:
                data = dm["data"]
                vals = {v: [d[v] for d in data if d.get(v) is not None] for v in DAYMET_VARS}
                row["c_tmax_mean"] = np.mean(vals["tmax"]) if vals["tmax"] else -999
                row["c_tmin_mean"] = np.mean(vals["tmin"]) if vals["tmin"] else -999
                row["c_tavg_mean"] = (row["c_tmax_mean"] + row["c_tmin_mean"]) / 2 if row["c_tmax_mean"] > -900 and row["c_tmin_mean"] > -900 else -999
                row["c_prcp_mean"] = np.mean(vals["prcp"]) if vals["prcp"] else -999
                row["c_srad_mean"] = np.mean(vals["srad"]) if vals["srad"] else -999
                row["c_vp_mean"] = np.mean(vals["vp"]) if vals["vp"] else -999
                row["c_daymet_ok"] = 1.0
            except Exception:
                for f in CLIMATE_FEATS:
                    row[f] = -999
        else:
            for f in CLIMATE_FEATS:
                row[f] = -999
        # MODIS — zeros (placeholder)
        for f in MODIS_FEATS:
            row[f] = 0.0
        # Invasion (GBIF counts)
        row["i_count_bt"] = float(get_gbif_count("Bromus_tectorum", lat, lon, year))
        row["i_count_bj"] = float(get_gbif_count("Bromus_japonicus", lat, lon, year))
        row["i_count_vd"] = float(get_gbif_count("Ventenata_dubia", lat, lon, year))
        row["i_count_ac"] = float(get_gbif_count("Aegilops_cylindrica", lat, lon, year))
        row["i_count_all"] = row["i_count_bt"] + row["i_count_bj"] + row["i_count_vd"] + row["i_count_ac"]
        # Fire history — zeros (placeholder; needs multi-year NBAC)
        for f in FIRE_FEATS:
            row[f] = 0.0
        rows.append(row)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{total} rows built...")

print(f"\nFeature matrix: {len(rows)} rows")

# ---------------------------------------------------------------------------
# 11. Attach NBAC burn labels (year t+1)
# ---------------------------------------------------------------------------

print("Attaching NBAC burn labels...")
for row in rows:
    label_year = row["year"] + 1
    if label_year > 2024:
        row["label"] = -1  # no label available
        continue
    frac = get_label(row["lat"], row["lon"], label_year)
    if frac < 0:
        row["label"] = -1  # unknown
    elif frac >= 0.01:  # ≥1% burn fraction threshold (matches I4 spec)
        row["label"] = 1
    else:
        row["label"] = 0

npos = sum(1 for r in rows if r["label"] == 1)
nknown = sum(1 for r in rows if r["label"] >= 0)
print(f"Known labels: {nknown}, positives: {npos} ({100.0*npos/max(nknown,1):.2f}%)")

# ---------------------------------------------------------------------------
# 12. Build train/val/test splits by year (temporal holdout)
# ---------------------------------------------------------------------------

train_rows = [r for r in rows if r["year"] in (2016, 2017, 2018, 2019, 2020, 2021) and r["label"] >= 0]
val_rows = [r for r in rows if r["year"] == 2022 and r["label"] >= 0]
test_rows = [r for r in rows if r["year"] in (2023, 2024) and r["label"] >= 0]

print(f"Splits — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

# ---------------------------------------------------------------------------
# 13. Write train_i5 binary corpus
# ---------------------------------------------------------------------------

def write_binary_corpus(split_name, split_rows, feat_names):
    n = len(split_rows)
    feat = len(feat_names)
    with open(OUTDIR / f"{split_name}.X.bin", "wb") as fx, \
         open(OUTDIR / f"{split_name}.L.bin", "wb") as fl, \
         open(OUTDIR / f"{split_name}.S.bin", "wb") as fs, \
         open(OUTDIR / f"{split_name}.A.bin", "wb") as fa:
        for aidx, row in enumerate(split_rows):
            xvals = [float(row.get(name, 0.0) or 0.0) for name in feat_names]
            fx.write(struct.pack(f"<{feat}d", *xvals))
            fl.write(struct.pack("<i", int(row.get("label", 0))))
            fs.write(struct.pack("<i", int(row.get("cell_id", 0)) % 5))
            fa.write(struct.pack("<i", aidx))

write_binary_corpus("train", train_rows, FEATURE_ORDER)
write_binary_corpus("val", val_rows, FEATURE_ORDER)
write_binary_corpus("test", test_rows, FEATURE_ORDER)

with open(OUTDIR / "corpus.hdr", "w") as f:
    f.write(f"FEAT {len(FEATURE_ORDER)}\n")
    f.write(f"NCLASS {NCLASS}\n")
    f.write(f"NTRAIN {len(train_rows)}\n")
    f.write(f"NVAL {len(val_rows)}\n")
    f.write(f"NTEST {len(test_rows)}\n")

# Also save feature names for reference
with open(OUTDIR / "feature_names.json", "w") as f:
    json.dump(FEATURE_ORDER, f, indent=2)

print(f"\nBinary corpus written to {OUTDIR}/")
print(f"  FEAT={len(FEATURE_ORDER)} NCLASS={NCLASS}")
print(f"  NTRAIN={len(train_rows)} NVAL={len(val_rows)} NTEST={len(test_rows)}")
print(f"  Next: ./build/bin/train_i5 {OUTDIR} 64 32 20 64 --export-lancius {OUTDIR}/i5.lancius --verify-lancius")

# ---------------------------------------------------------------------------
# 14. Optionally train I5
# ---------------------------------------------------------------------------

TRAINER = Path("build/bin/train_i5")
TRAIN_I5 = os.environ.get("TRAIN_I5", "0") == "1"
if TRAINER.exists() and TRAIN_I5:
    cmd = [str(TRAINER), str(OUTDIR), "64", "32", "20", "64",
           "--export-lancius", str(OUTDIR / "i5.lancius"),
           "--verify-lancius", "--save-dir", str(OUTDIR / "models")]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
elif TRAINER.exists() and not TRAIN_I5:
    print("\nbuild/bin/train_i5 found. Set TRAIN_I5=1 to auto-train, or run manually:")
    print(f"  {TRAINER} {OUTDIR} 64 32 20 64 --export-lancius {OUTDIR}/i5.lancius --verify-lancius")
else:
    print("\nbuild/bin/train_i5 not found — run `make build/bin/train_i5` first.")

print("\nPipeline complete.")
