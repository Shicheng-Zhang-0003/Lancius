#!/usr/bin/env python3
"""Freeze the I4 grid and audit NBAC fire labels (no model code)."""
import json, math, os, sys
from collections import Counter
import requests
import numpy as np
from pyproj import Transformer
from shapely.geometry import box, shape, mapping
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid
import rasterio
from rasterio.features import rasterize

HERE = os.path.dirname(os.path.abspath(__file__))
I4 = os.path.dirname(HERE)
DATA = os.path.join(I4, "data")
os.makedirs(DATA, exist_ok=True)

WEST, SOUTH, EAST, NORTH = -114.5, 49.0, -110.0, 51.2
CRS = "EPSG:5070"
RES = 1000.0
BBOX = box(WEST, SOUTH, EAST, NORTH)
TR = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

def projected_bounds():
    pts = [TR.transform(WEST + (EAST-WEST)*i/100, SOUTH + (NORTH-SOUTH)*j/100)
           for i in range(101) for j in range(101)]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)

def grid():
    minx, miny, maxx, maxy = projected_bounds()
    minx = math.floor(minx / RES) * RES
    miny = math.floor(miny / RES) * RES
    maxx = math.ceil(maxx / RES) * RES
    maxy = math.ceil(maxy / RES) * RES
    w = int(round((maxx-minx)/RES)); h = int(round((maxy-miny)/RES))
    return dict(crs=CRS, res=RES, transform=rasterio.transform.from_origin(minx, maxy, RES, RES),
                width=w, height=h, bounds=(minx,miny,maxx,maxy))

def fetch_nbac():
    p = os.path.join(DATA, "nbac_bbox.geojson")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return json.load(open(p))
    params = {
        "service":"WFS", "request":"GetFeature", "version":"2.0.0",
        "typeNames":"public:nbac", "outputFormat":"application/json",
        "srsName":"EPSG:4326",
        "bbox": f"{WEST},{SOUTH},{EAST},{NORTH},EPSG:4326",
    }
    r = requests.get("https://cwfis.cfs.nrcan.gc.ca/geoserver/wfs", params=params, timeout=180)
    r.raise_for_status()
    d = r.json()
    json.dump(d, open(p, "w"))
    print("NBAC", d.get("numberMatched"), "features ->", p)
    return d

def main():
    g = grid()
    print("grid", g["width"], "x", g["height"], "=", g["width"]*g["height"], "cells")
    print("bounds 5070", g["bounds"])
    json.dump({k:v for k,v in g.items() if k != "transform"}, open(os.path.join(DATA,"grid.json"),"w"), indent=2)

    d = fetch_nbac()
    years = list(range(2016, 2025))
    # 100 m raster for burn fraction, then 10x10 block sum.
    f = 10
    sub = rasterio.transform.from_origin(g["bounds"][0], g["bounds"][3], RES/f, RES/f)
    subw, subh = g["width"]*f, g["height"]*f
    frac = {}
    total_ha = {}
    for y in years:
        geoms=[]
        ha=0.0
        for feat in d["features"]:
            p=feat.get("properties", {})
            if p.get("year") != y: continue
            if p.get("prescribed") in ("Y","YES","true","True"): continue
            try:
                geom=make_valid(shape(feat["geometry"]))
            except Exception:
                continue
            if geom.is_empty: continue
            geom=shp_transform(lambda x,y,z=None: TR.transform(x,y), geom)
            geoms.append(geom)
            ha += float(p.get("adj_ha") or p.get("poly_ha") or 0.0)
        if geoms:
            a=rasterize([(geom,1) for geom in geoms], out_shape=(subh,subw), transform=sub,
                        fill=0, dtype="uint8", all_touched=True)
            a=a.reshape(g["height"],f,g["width"],f).mean(axis=(1,3)).astype("float32")
        else:
            a=np.zeros((g["height"],g["width"]),"float32")
        frac[y]=a; total_ha[y]=ha
        print(y, "fires",len(geoms),"ha",round(ha,1),"cells>=1%",int((a>=0.01).sum()),
              "cells>=10%",int((a>=0.10).sum()),"maxfrac",float(a.max()))
    np.savez_compressed(os.path.join(DATA,"labels_1km.npz"), **{str(y): frac[y] for y in years})
    json.dump({str(y):{"fires":int(sum(1 for f in d["features"] if f["properties"].get("year")==y)),
                        "ha":total_ha[y],
                        "cells_ge1pct":int((frac[y]>=0.01).sum()),
                        "cells_ge10pct":int((frac[y]>=0.10).sum())}
               for y in years}, open(os.path.join(DATA,"label_audit.json"),"w"), indent=2)

if __name__ == "__main__":
    main()
