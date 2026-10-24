#!/usr/bin/env python3
"""Shared I4 grid, label and raster helpers.

The study area is the dry interior of southern BC and Alberta, the region the
HSBDC brief calls "southern BC/Alberta grasslands".  The working grid is an
equal-area 2 km grid in NAD83 / Conus Albers (EPSG:5070); all vectors and
rasters are reprojected to it before they meet.
"""
import json, math, os
import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from affine import Affine

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
DATA = os.path.join(I4, "data")
CRS = "EPSG:5070"
RES = 2000.0
# Southern BC + southern Alberta dry interior.  Covers the Okanagan,
# Kamloops, Kootenays, Cypress Hills and the Palliser triangle.
WEST, SOUTH, EAST, NORTH = -121.0, 49.0, -110.0, 51.2
FEATURE_YEARS = list(range(2015, 2024))   # features in t, labels in t+1
LABEL_YEARS = list(range(2016, 2025))       # target labels
ALL_LABEL_YEARS = list(range(2013, 2025))  # history + target
CLASSES = {10:"tree", 20:"shrub", 30:"grass", 40:"crop", 50:"built",
           60:"bare", 70:"snow", 80:"water", 90:"wetland", 95:"mangrove",
           100:"moss"}
RANGELAND = (20, 30, 40, 60)

def tr4326():
    return Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

def grid():
    p = os.path.join(DATA, "grid.json")
    if os.path.exists(p):
        g = json.load(open(p))
        if g.get("res") == RES and g.get("bbox_wgs84") == [WEST,SOUTH,EAST,NORTH]:
            g["transform"] = Affine(*g["transform"])
            return g
    tr = tr4326()
    minx = miny = 1e30; maxx = maxy = -1e30
    for i in range(201):
        for j in range(101):
            x, y = tr.transform(WEST + (EAST-WEST)*i/200,
                                SOUTH + (NORTH-SOUTH)*j/100)
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)
    minx = math.floor(minx/RES)*RES; maxx = math.ceil(maxx/RES)*RES
    miny = math.floor(miny/RES)*RES; maxy = math.ceil(maxy/RES)*RES
    w = int(round((maxx-minx)/RES)); h = int(round((maxy-miny)/RES))
    t = rasterio.transform.from_origin(minx, maxy, RES, RES)
    g = {"crs":CRS, "res":RES, "bbox_wgs84":[WEST,SOUTH,EAST,NORTH],
         "bounds":[minx,miny,maxx,maxy], "width":w, "height":h,
         "transform":[t.a,t.b,t.c,t.d,t.e,t.f]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(g, open(p,"w"), indent=2)
    g["transform"] = t
    return g

def study_mask(g=None):
    """Cells whose centers fall inside the WGS84 study bbox.

    The Albers rectangle is a bounding box of the projected WGS84 bbox, so its
    corners extend well beyond the intended study area.  Every corpus / climate
    consumer must intersect with this mask; otherwise labels outside the NBAC
    fetch are structural zeros and edge climate interpolation extrapolates.
    """
    _, _, lon, lat = cell_centers_lonlat(g)
    return ((lon >= WEST) & (lon <= EAST) &
            (lat >= SOUTH) & (lat <= NORTH))

def cell_centers_lonlat(g=None):
    """Return (x5070, y5070, lon, lat) rows-major for every grid cell."""
    g = g or grid()
    t = g["transform"]
    rows, cols = np.indices((g["height"], g["width"]))
    x = t.c + (cols + 0.5)*t.a
    y = t.f + (rows + 0.5)*t.e
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return x, y, lon, lat

def fetch_nbac(force=False):
    p = os.path.join(DATA, "nbac_bbox.geojson")
    if os.path.exists(p) and not force:
        d = json.load(open(p))
        if d.get("_i4_bbox") == [WEST,SOUTH,EAST,NORTH]:
            return d
    params = {"service":"WFS","request":"GetFeature","version":"2.0.0",
              "typeNames":"public:nbac","outputFormat":"application/json",
              "srsName":"EPSG:4326",
              "bbox":f"{WEST},{SOUTH},{EAST},{NORTH},EPSG:4326"}
    r = requests.get("https://cwfis.cfs.nrcan.gc.ca/geoserver/wfs",
                     params=params, timeout=300)
    r.raise_for_status()
    d = r.json(); d["_i4_bbox"] = [WEST,SOUTH,EAST,NORTH]
    os.makedirs(DATA, exist_ok=True)
    json.dump(d, open(p,"w"))
    print("NBAC", d.get("numberMatched"), "features ->", p)
    return d

def nbac_shapes_5070(d=None):
    d = d or fetch_nbac()
    tr = tr4326()
    out = []
    for feat in d["features"]:
        p = feat.get("properties", {})
        if p.get("year") not in ALL_LABEL_YEARS: continue
        if str(p.get("prescribed","")).upper() in ("Y","YES","TRUE"):
            continue
        try:
            geom = make_valid(shape(feat["geometry"]))
            if geom.is_empty: continue
            geom = shp_transform(lambda x,y,z=None: tr.transform(x,y), geom)
        except Exception:
            continue
        out.append((int(p["year"]), geom, p))
    return out

def burn_fractions(g=None, force=False):
    """Rasterize NBAC at 400 m, then block-average to the 2 km grid."""
    g = g or grid()
    p = os.path.join(DATA, "labels_2km.npz")
    if os.path.exists(p) and not force:
        z = np.load(p)
        return {int(k): z[k] for k in z.files}
    f = 5  # 400 m subcells per 2 km cell
    sub = rasterio.transform.from_origin(g["bounds"][0], g["bounds"][3],
                                         RES/f, RES/f)
    sw, sh = g["width"]*f, g["height"]*f
    shapes = nbac_shapes_5070()
    out = {}
    for y in ALL_LABEL_YEARS:
        geoms = [(geom,1) for (yy,geom,_) in shapes if yy == y]
        if geoms:
            a = rasterize(geoms, out_shape=(sh,sw), transform=sub,
                          fill=0, dtype="uint8", all_touched=True)
            a = a.reshape(g["height"], f, g["width"], f).mean(axis=(1,3))
            a = a.astype("float32")
        else:
            a = np.zeros((g["height"], g["width"]), "float32")
        out[y] = a
        print(f"label {y}: fires={len(geoms):4d} cells>=1%={int((a>=.01).sum()):5d}")
    np.savez_compressed(p, **{str(y):out[y] for y in ALL_LABEL_YEARS})
    return out

def worldcover_fractions(g=None, force=False):
    """ESA WorldCover 2021 class fractions per 2 km cell."""
    g = g or grid()
    p = os.path.join(DATA, "worldcover_frac_2km.npz")
    if os.path.exists(p) and not force:
        z = np.load(p)
        return {c: z[str(c)] for c in CLASSES}
    from pystac_client import Client
    import planetary_computer as pc
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = list(cat.search(collections=["esa-worldcover"],
                            bbox=[WEST,SOUTH,EAST,NORTH],
                            datetime="2021-01-01/2021-12-31", limit=50).items())
    print("worldcover tiles", len(items))
    dst = {c: np.zeros((g["height"], g["width"]), "float32") for c in CLASSES}
    for it in items:
        ds = rasterio.open(pc.sign(it.assets["map"]).href)
        ow = oh = 1800  # 3 deg / 200 m
        arr = ds.read(1, out_shape=(oh,ow), resampling=Resampling.nearest)
        st = ds.transform * Affine.scale(ds.width/ow, ds.height/oh)
        for c in CLASSES:
            src = (arr == c).astype("float32")
            if not src.any(): continue
            tmp = np.zeros_like(dst[c])
            reproject(src, tmp, src_transform=st, src_crs=ds.crs,
                      dst_transform=g["transform"], dst_crs=CRS,
                      resampling=Resampling.average, init_dest_nodata=0)
            dst[c] += tmp
        ds.close()
    # mosaic overlap can slightly exceed 1; renormalize the 11 classes.
    tot = sum(dst.values())
    for c in dst:
        dst[c] = np.clip(dst[c]/np.maximum(tot,1e-6), 0, 1)
    np.savez_compressed(p, **{str(c):dst[c] for c in CLASSES})
    return dst

if __name__ == "__main__":
    g = grid()
    print("grid", g["width"], "x", g["height"], "=", g["width"]*g["height"])
    print("bounds", g["bounds"])
    lab = burn_fractions(g)
    wc = worldcover_fractions(g)
    r = sum(wc[c] for c in RANGELAND)
    for th in (0.3,0.5,0.7):
        print("rangeland frac >=",th,int((r>=th).sum()), "cells")
    pos = sum((lab[y]>=.01) for y in LABEL_YEARS)
    print("positive cell-years (all land)", int(pos.sum()))
    for th in (0.3,0.5,0.7):
        m = r>=th
        print("positive cell-years (rangeland>=",th,")", int((pos*m).sum()),
              "prevalence", float((pos*m).sum()/(m.sum()*len(LABEL_YEARS))))
