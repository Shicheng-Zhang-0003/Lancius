#!/usr/bin/env python3
"""
I5 fire-data acquisition framework.

Stages a binary corpus for `examples/train_i5` from public + credentialed sources.

Sources (from generation_1/I4/targets/i4_fire_next_year_v0.1.json):
  - NBAC / CWFIS WFS        -> burn labels (year t+1)
  - MODIS MOD13Q1           -> phenology / NDVI-EVI (Planetary Computer)
  - Daymet                  -> climate (NASA Earthdata login REQUIRED)
  - NASADEM                 -> topography (NASA Earthdata login REQUIRED)
  - ESA WorldCover 2021     -> land-cover mask (Copernicus, may need auth)
  - GBIF                    -> invasion occurrence (public API, no auth)

Only GBIF is fetchable here without credentials. The other fetchers are
stubs with clear auth instructions; fill in your keys/tokens and they become
live. The converter at the bottom takes whatever rasters/tables you assemble
and emits the train_i5 binary layout:

    <out_dir>/
      corpus.hdr            # "FEAT N", "NCLASS 2", "NTRAIN N", "NVAL N", "NTEST N"
      train.X.bin           # float64 (ntrain, FEAT), little-endian
      train.L.bin           # int32   (ntrain,)
      train.S.bin           # int32   (ntrain,)   spatial group id
      train.A.bin           # int32   (ntrain,)   row id
      {val,test}.*.bin      # same layout
"""

import csv
import json
import math
import os
import struct
import sys
import time
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

OUTDIR = "data_i5_real"
FEAT_NAMES = [
    # static (16)
    "elevation", "slope", "aspect_sin", "aspect_cos", "ruggedness",
    "tree", "shrub", "grass", "crop", "built", "bare", "water",
    "wetland", "rangeland", "built_5km", "dist_built_km",
    # climate (22) -- placeholders until Daymet keys added
    "c_tmax_mean", "c_tmin_mean", "c_tavg_mean", "c_prcp_mean", "c_srad_mean",
    "c_vp_mean", "c_wet_days", "c_d0", "c_d1", "c_d2", "c_d3", "c_d4",
    "c_d5", "c_d6", "c_d7", "c_d8", "c_d9", "c_d10", "c_d11", "c_d12",
    "c_d13", "c_d14",
    # modis (17)
    "m_ndvi_spring", "m_evi_spring", "m_ndvi_summer", "m_evi_summer",
    "m_ndvi_fall", "m_evi_fall", "m_ndvi_max", "m_evi_max", "m_ndvi_amp",
    "m_evi_amp", "m_ndvi_season", "m_evi_season", "m_greenup", "m_senescence",
    "m_length_of_season", "m_mid_greenup", "m_mid_senescence",
    # invasion (15)
    "i_count10_bt", "i_count25_bt", "i_dist_km_bt",
    "i_count10_bj", "i_count25_bj", "i_dist_km_bj",
    "i_count10_vd", "i_count25_vd", "i_dist_km_vd",
    "i_count10_ac", "i_count25_ac", "i_dist_km_ac",
    "i_count10_all", "i_count25_all", "i_dist_km_all",
    # fire_history (7)
    "fh_fire_t", "fh_fire_t1", "fh_fire_t2", "fh_fire_count_3yr",
    "fh_fire_years_since", "fh_fire_density_3x3", "fh_fire_density_5x5",
]
NCLASS = 2
POS_CLASS = 1  # burned next year


# ============================================================================
# 1. PUBLIC FETCHERS (no auth needed)
# ============================================================================

def fetch_gbif_invasion(taxon_key, out_csv, limit=10000):
    """
    Pull GBIF occurrence records for an invasive taxon.
    GBIF REST API is public; no token required for basic occurrence search.
    Returns list of (lat, lon, year) tuples.
    """
    url = (
        "https://api.gbif.org/v1/occurrence/search"
        f"?taxon_key={taxon_key}"
        f"&limit={limit}"
        f"&has_coordinate=true"
        f"&year=2013,2024"
    )
    print(f"[gbif] {taxon_key}: {url}")
    try:
        req = Request(url, headers={"User-Agent": "lancius-i5-acquire/1.0"})
        with urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        print(f"  !! GBIF fetch failed for {taxon_key}: {exc}")
        return []

    records = []
    for res in data.get("results", []):
        lat = res.get("decimalLatitude")
        lon = res.get("decimalLongitude")
        year = res.get("year")
        if lat is None or lon is None or year is None:
            continue
        records.append((float(lat), float(lon), int(year)))
    print(f"  got {len(records)} records")
    return records


# GBIF taxon keys for the focal taxa in the I4 spec
GBIF_TAXA = {
    "Bromus_tectorum": 2710479,   # cheatgrass / downy brome
    "Bromus_japonicus": 2710481,  # Japanese brome
    "Ventenata_dubia": 2710914,   # North African grass
    "Aegilops_cylindrica": 2711190,  # jointed goatgrass
}


# ============================================================================
# 2. CREDENTIAL-GATED FETCHERS (stubs -- fill in your keys)
# ============================================================================

def fetch_daymet(variable, lat, lon, year, start_date, end_date):
    """
    Daymet single-pixel API. Requires NASA Earthdata login.
    Docs: https://daymet.ornl.gov/single-pixel-api/
    You need: an Earthdata account + app key / token in headers.
    """
    print(f"[daymet] STUB: {variable} at ({lat},{lon}) for {start_date}->{end_date}")
    print("  !! Requires NASA Earthdata credentials. Implement once you have them.")
    return None


def fetch_nasadem(lat, lon):
    """
    NASADEM elevation. Requires NASA Earthdata login.
    """
    print(f"[nasadem] STUB: elevation at ({lat},{lon})")
    print("  !! Requires NASA Earthdata credentials.")
    return None


def fetch_worldcover(lat, lon):
    """
    ESA WorldCover 2021 land cover at 10 m. May need Copernicus credentials.
    """
    print(f"[worldcover] STUB: land cover at ({lat},{lon})")
    print("  !! May require Copernicus Data Space credentials.")
    return None


def fetch_nbac_wfs(cell_geom, year):
    """
    NBAC burned-area polygon via CWFIS WFS.
    May be publicly accessible as a WFS endpoint; verify the URL.
    """
    print(f"[nbac] STUB: burn fraction for cell in {year}")
    print("  !! Determine CWFIS WFS endpoint; may be public or may need access.")
    return None


def fetch_modis_planetary_computer(cell_geom, year):
    """
    MODIS MOD13Q1 via Microsoft Planetary Computer.
    May need a SAS token or STAC API key.
    """
    print(f"[modis] STUB: NDVI/EVI for cell in {year}")
    print("  !! May require Planetary Computer auth.")
    return None


# ============================================================================
# 3. RASTER -> FEATURE CONVERSION (plug your downloaded rasters here)
# ============================================================================

def sample_raster(raster_path, lat, lon):
    """
    Placeholder: read one pixel value from a GeoTIFF/NetCDF at (lat, lon).
    Implement with rasterio / netCDF4 / gdal once you have the rasters on disk.
    Returns float or None.
    """
    print(f"[raster] STUB: sample {raster_path} at ({lat},{lon})")
    print("  !! Implement with rasterio/gdal once rasters are staged locally.")
    return None


# ============================================================================
# 4. CORPUS BUILDER (assemble features into train_i5 binary layout)
# ============================================================================

def build_binary_corpus(rows, out_dir, feat=len(FEAT_NAMES), nclass=NCLASS,
                        train_frac=0.67, val_frac=0.13, test_frac=0.20,
                        seed=20261022):
    """
    rows: list of dicts with keys = FEAT_NAMES + "label" + "cell_id" + "year"
    Writes corpus.hdr + {train,val,test}.{X,L,S,A}.bin in out_dir.
    """
    import random
    random.seed(seed)
    rows = list(rows)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)

    # corpus.hdr
    with open(os.path.join(out_dir, "corpus.hdr"), "w") as f:
        f.write(f"FEAT {feat}\n")
        f.write(f"NCLASS {nclass}\n")
        f.write(f"NTRAIN {len(splits['train'])}\n")
        f.write(f"NVAL {len(splits['val'])}\n")
        f.write(f"NTEST {len(splits['test'])}\n")

    for split, split_rows in splits.items():
        n_split = len(split_rows)
        with open(os.path.join(out_dir, f"{split}.X.bin"), "wb") as fx, \
             open(os.path.join(out_dir, f"{split}.L.bin"), "wb") as fl, \
             open(os.path.join(out_dir, f"{split}.S.bin"), "wb") as fs, \
             open(os.path.join(out_dir, f"{split}.A.bin"), "wb") as fa:
            for aidx, row in enumerate(split_rows):
                # X: FEAT float64
                xvals = [float(row.get(name, 0.0) or 0.0) for name in FEAT_NAMES]
                fx.write(struct.pack(f"<{feat}d", *xvals))
                # L: int32 label
                fl.write(struct.pack("<i", int(row.get("label", 0))))
                # S: int32 spatial group (cell_id mod something)
                fs.write(struct.pack("<i", int(row.get("cell_id", 0)) % 5))
                # A: int32 row id
                fa.write(struct.pack("<i", aidx))

    # Summary
    for split in ("train", "val", "test"):
        npos = sum(1 for r in splits[split] if int(r.get("label", 0)) == 1)
        print(f"  {split}: {len(splits[split])} rows, {npos} positives "
              f"({100.0*npos/len(splits[split]):.2f}%)")

    print(f"Wrote binary corpus to {out_dir}/")
    print(f"  FEAT={feat} NCLASS={nclass}")
    print(f"  Next: ./build/bin/train_i5 {out_dir} 64 32 20 64")


# ============================================================================
# 5. MAIN: assemble what we can, stub the rest
# ============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("==================================================================")
    print("  I5 fire-data acquisition framework")
    print("  Public: GBIF invasion records")
    print("  Stubbed (need your credentials): Daymet, NASADEM, WorldCover,")
    print("                NBAC/CWFIS WFS, MODIS/Planetary Computer")
    print("==================================================================\n")

    # --- Step A: pull public GBIF invasion records ---
    all_invasion = {}
    for taxon_name, taxon_key in GBIF_TAXA.items():
        recs = fetch_gbif_invasion(taxon_key, os.path.join(OUTDIR, f"{taxon_name}.csv"))
        all_invasion[taxon_name] = recs
        time.sleep(1)  # be polite to GBIF

    # --- Step B: print what you still need to stage manually ---
    print("\n------------------------------------------------------------------")
    print("  MANUAL STAGING REQUIRED (fill in the stubs above with keys):")
    print("  1. Download Daymet climate tiles  -> set NASA Earthdata key in fetch_daymet()")
    print("  2. Download NASADEM elevation     -> set Earthdata key in fetch_nasadem()")
    print("  3. Download ESA WorldCover 2021   -> set Copernicus creds in fetch_worldcover()")
    print("  4. Query CWFIS/NBAC WFS           -> determine endpoint in fetch_nbac_wfs()")
    print("  5. Query MODIS MOD13Q1            -> set Planetary Computer token in fetch_modis_planetary_computer()")
    print("  6. Sample every raster at cell centroids -> implement sample_raster() with rasterio/gdal")
    print("  7. Build label: NBAC burn fraction >= 1% in year t+1, wildfire-only")
    print("------------------------------------------------------------------\n")

    # --- Step C: build an empty corpus skeleton so the pipeline is proven ---
    # We create one row per (cell, year) with zeros + GBIF-derived invasion counts
    # where available. Replace zeros with real raster values once staged.
    rows = []
    cell_id = 0
    # Example: 50 cells x 9 years (2015..2023 features -> labels 2016..2024)
    for y in range(2015, 2024):
        for cx in range(50):
            row = {name: 0.0 for name in FEAT_NAMES}
            row["cell_id"] = cell_id
            row["year"] = y
            # invasion features: cumulative GBIF counts within ~25 km (placeholder: use raw count)
            # Once you have real rasters, replace these zeros with sampled values.
            row["label"] = 0  # placeholder: set to 1 if NBAC burn fraction >= 0.01 in y+1
            rows.append(row)
            cell_id += 1

    print(f"Assembled {len(rows)} placeholder cell-year rows.")
    print("  Replace zeros with real raster samples before training.")

    build_binary_corpus(rows, OUTDIR)


if __name__ == "__main__":
    main()
