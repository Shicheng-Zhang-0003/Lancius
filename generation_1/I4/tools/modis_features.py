#!/usr/bin/env python3
"""MODIS MOD13Q1 16-day NDVI/EVI -> 2 km seasonal metrics."""
import os, sys, re, time
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY","1")
os.environ.setdefault("GDAL_HTTP_TIMEOUT","60")
import numpy as np
import rasterio
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

YEARS = list(range(2015, 2024))  # feature years; labels use t+1
DAYS = (105, 289)                # 15 Apr -- 16 Oct

def one_item(item):
    m=re.search(r"\.A(\d{4})(\d{3})\.", item.id)
    if not m: return None
    yr, day = int(m.group(1)), int(m.group(2))
    if day < DAYS[0] or day > DAYS[1]: return None
    it=pc.sign(item)
    g=C.grid()
    def read(key, resampling, nodata, dst_nodata):
        last=None
        for attempt in range(5):
            try:
                with rasterio.open(it.assets[key].href) as ds:
                    b=transform_bounds(C.CRS,ds.crs,*g["bounds"],densify_pts=21)
                    win=from_bounds(*b,transform=ds.transform).round_offsets().round_lengths()
                    win=win.intersection(Window(0,0,ds.width,ds.height))
                    if win.width<=0 or win.height<=0: return None
                    src=ds.read(1,window=win)
                    src_transform=ds.window_transform(win)
                    dst=np.full((g["height"],g["width"]),dst_nodata,"float32")
                    reproject(src,dst,src_transform=src_transform,src_crs=ds.crs,
                              dst_transform=g["transform"],dst_crs=C.CRS,resampling=resampling,
                              src_nodata=nodata,dst_nodata=dst_nodata)
                    return dst
            except Exception as e:
                last=e; time.sleep(2.0*(attempt+1))
        print("WARN read failed",item.id,key,last,flush=True)
        return None
    nd=read("250m_16_days_NDVI",Resampling.average,-3000,np.nan)
    ev=read("250m_16_days_EVI",Resampling.average,-3000,np.nan)
    rel=read("250m_16_days_pixel_reliability",Resampling.nearest,255,255)
    if nd is None or ev is None or rel is None: return None
    ok=(rel<=1)&np.isfinite(nd)&np.isfinite(ev)
    return yr,day,np.where(ok,nd*1e-4,np.nan).astype("float32"),np.where(ok,ev*1e-4,np.nan).astype("float32")

def year_arrays(year):
    c=Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    s=c.search(collections=["modis-13Q1-061"],bbox=[C.WEST,C.SOUTH,C.EAST,C.NORTH],
               datetime=f"{year}-04-01/{year}-10-20",limit=1000)
    items=list(s.items()); print(year,"items",len(items),flush=True)
    byday={}
    def safe_one(item):
        try:
            return one_item(item)
        except Exception as e:
            print("WARN item failed",item.id,e,flush=True)
            return None
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        done=0
        for res in ex.map(safe_one,items):
            done+=1
            if done%10==0: print(year,"done",done,"/",len(items),flush=True)
            if res is None: continue
            yr,day,nd,ev=res
            if yr!=year: continue
            if day not in byday:
                byday[day]=[np.full_like(nd,np.nan),np.full_like(ev,np.nan)]
            a=byday[day]
            a[0]=np.where(np.isfinite(nd),nd,a[0])
            a[1]=np.where(np.isfinite(ev),ev,a[1])
    days=sorted(byday)
    nd=np.stack([byday[d][0] for d in days]) if days else np.empty((0,0,0))
    ev=np.stack([byday[d][1] for d in days]) if days else np.empty((0,0,0))
    return np.array(days),nd,ev

def metrics(days,nd,ev):
    g=C.grid(); h,w=g["height"],g["width"]
    out={}
    if len(days)==0:
        for k in ["ndvi_max","ndvi_mean","ndvi_amp","ndvi_spring","ndvi_summer",
                  "ndvi_fall","ndvi_sd","ndvi_doy_max","ndvi_spring_anom",
                  "ndvi_senescence","evi_max","evi_mean","evi_amp","evi_spring",
                  "evi_summer","evi_sd","valid_count"]:
            out[k]=np.full((h,w),np.nan,"float32")
        return out
    def band(a):
        return {
            "max":np.nanmax(a,axis=0),"mean":np.nanmean(a,axis=0),
            "amp":np.nanmax(a,axis=0)-np.nanmin(a,axis=0),
            "spring":np.nanmean(a[(days>=105)&(days<=151)],axis=0),
            "summer":np.nanmean(a[(days>=152)&(days<=243)],axis=0),
            "fall":np.nanmean(a[(days>=244)&(days<=289)],axis=0),
            "sd":np.nanstd(a,axis=0),
            "count":np.isfinite(a).sum(axis=0).astype("float32"),
        }
    n=band(nd); e=band(ev)
    out["ndvi_max"]=n["max"]; out["ndvi_mean"]=n["mean"]; out["ndvi_amp"]=n["amp"]
    out["ndvi_spring"]=n["spring"]; out["ndvi_summer"]=n["summer"]; out["ndvi_fall"]=n["fall"]
    out["ndvi_sd"]=n["sd"]; out["valid_count"]=n["count"]
    # day of maximum: argmax over valid composites, ignoring all-NaN cells
    filled=np.where(np.isfinite(nd),nd,-np.inf)
    idx=np.argmax(filled,axis=0).astype("float32")
    out["ndvi_doy_max"]=np.where(np.isfinite(filled).any(axis=0),days[idx.astype(int)],np.nan).astype("float32")
    out["ndvi_spring_anom"]=n["spring"]-n["mean"]
    out["ndvi_senescence"]=n["summer"]-n["fall"]
    out["evi_max"]=e["max"]; out["evi_mean"]=e["mean"]; out["evi_amp"]=e["amp"]
    out["evi_spring"]=e["spring"]; out["evi_summer"]=e["summer"]; out["evi_sd"]=e["sd"]
    return {k:v.astype("float32") for k,v in out.items()}

def main():
    p=os.path.join(C.DATA,"modis_features_2km.npz")
    if os.path.exists(p): print("cached",p); return
    out={}
    for y in YEARS:
        yp=os.path.join(C.DATA,f"modis_year_{y}.npz")
        if os.path.exists(yp):
            z=np.load(yp)
            for k in z.files: out[f"{k}_{y}"]=z[k]
            print("modis",y,"cached",len(z.files),flush=True); continue
        d,nd,ev=year_arrays(y)
        m=metrics(d,nd,ev)
        np.savez_compressed(yp,**m)
        for k,v in m.items(): out[f"{k}_{y}"]=v
        print("modis",y,"days",len(d),"layers",len(m),flush=True)
    np.savez_compressed(p,**out)
    print("saved",len(out),"modis layers")

if __name__=="__main__": main()
