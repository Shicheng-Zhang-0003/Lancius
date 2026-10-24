#!/usr/bin/env python3
"""Cumulative invasive-annual-grass occurrence features (GBIF)."""
import os, sys, json, time
import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

SPECIES = ["Bromus tectorum", "Bromus japonicus", "Ventenata dubia",
           "Aegilops cylindrica"]

def fetch():
    p=os.path.join(C.DATA,"gbif_invasives.json")
    if os.path.exists(p): return json.load(open(p))
    out=[]
    for name in SPECIES:
        m=requests.get("https://api.gbif.org/v1/species/match",params={"name":name},timeout=60).json()
        key=m.get("usageKey"); print(name,key)
        off=0
        while True:
            params={"taxonKey":key,"decimalLatitude":f"{C.SOUTH},{C.NORTH}",
                    "decimalLongitude":f"{C.WEST},{C.EAST}","hasCoordinate":"true",
                    "hasGeospatialIssue":"false","occurrenceStatus":"PRESENT",
                    "limit":300,"offset":off}
            r=requests.get("https://api.gbif.org/v1/occurrence/search",params=params,timeout=120)
            r.raise_for_status(); d=r.json()
            for rec in d.get("results",[]):
                y=rec.get("year")
                if y is None:
                    y=2000  # pre-study default; documented in the target spec
                out.append({"species":name,"lat":rec["decimalLatitude"],
                            "lon":rec["decimalLongitude"],"year":int(y)})
            print("  offset",off,"got",len(d.get("results",[])),"total",d.get("count"))
            off += 300
            if off >= d.get("count",0) or not d.get("results"): break
            time.sleep(0.05)
    json.dump(out,open(p,"w"))
    print("records",len(out),"->",p)
    return out

def cell_index(g, lon, lat):
    # nearest grid cell for a point
    tr=Transformer.from_crs(4326,C.CRS,always_xy=True)
    x,y=tr.transform(lon,lat)
    t=g["transform"]
    col=np.floor((x-t.c)/t.a).astype(int)
    row=np.floor((y-t.f)/t.e).astype(int)
    ok=(col>=0)&(col<g["width"])&(row>=0)&(row<g["height"])
    return row, col, ok

def disk(radius_cells):
    y,x=np.ogrid[-radius_cells:radius_cells+1,-radius_cells:radius_cells+1]
    return ((x*x+y*y)<=radius_cells*radius_cells).astype("float32")

def main():
    g=C.grid(); rec=fetch()
    h,w=g["height"],g["width"]
    # cell centers for distance to nearest occurrence
    x,y,lon,lat=C.cell_centers_lonlat(g)
    xy=np.column_stack([x.ravel(),y.ravel()])
    species=sorted(set(r["species"] for r in rec))
    out={}
    for sp in species:
        rows=[]
        for r in rec:
            if r["species"]!=sp: continue
            rr,cc,ok=cell_index(g,r["lon"],r["lat"])
            if ok: rows.append((int(r["year"]),float(x[rr,cc]),float(y[rr,cc])))
        print(sp,"in grid",len(rows))
        for fy in C.FEATURE_YEARS:
            pts=[(xx,yy) for (yy,xx,zz) in [(0,0,0)]]
            pts=[(xx,yy) for (yr,xx,yy) in rows if yr<=fy]
            if pts:
                tree=cKDTree(np.array(pts))
                d,_=tree.query(xy,k=1)
                dist=(d/1000.0).reshape(h,w).astype("float32")
            else:
                dist=np.full((h,w),999.0,"float32")
            # cumulative occurrence count by cell, then disk sums
            cnt=np.zeros((h,w),"float32")
            for (yr,xx,yy) in rows:
                if yr>fy: continue
                tr=Transformer.from_crs(C.CRS,C.CRS)  # no-op
            # build count from records up to fy
            for r in rec:
                if r["species"]!=sp or r["year"]>fy: continue
                rr,cc,ok=cell_index(g,r["lon"],r["lat"])
                if ok: cnt[rr,cc]+=1
            c25=ndimage.convolve(cnt,disk(int(round(25.0/C.RES))),mode="constant")
            c10=ndimage.convolve(cnt,disk(int(round(10.0/C.RES))),mode="constant")
            key=sp.lower().replace(" ","_")
            out[f"{key}_dist_km_{fy}"]=dist
            out[f"{key}_count25_{fy}"]=c25.astype("float32")
            out[f"{key}_count10_{fy}"]=c10.astype("float32")
    # aggregate all species
    for fy in C.FEATURE_YEARS:
        ds=[out[f"{sp.lower().replace(' ','_')}_dist_km_{fy}"] for sp in species]
        c25=sum(out[f"{sp.lower().replace(' ','_')}_count25_{fy}"] for sp in species)
        c10=sum(out[f"{sp.lower().replace(' ','_')}_count10_{fy}"] for sp in species)
        out[f"inv_dist_km_{fy}"]=np.minimum.reduce(ds).astype("float32")
        out[f"inv_count25_{fy}"]=c25.astype("float32")
        out[f"inv_count10_{fy}"]=c10.astype("float32")
    np.savez_compressed(os.path.join(C.DATA,"gbif_features_2km.npz"),**out)
    print("saved",len(out),"layers")

if __name__=="__main__": main()
