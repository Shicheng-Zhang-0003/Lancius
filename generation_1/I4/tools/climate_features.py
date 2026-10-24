#!/usr/bin/env python3
"""Daymet daily -> 2 km annual/seasonal climate features.

Daymet is sampled through its single-pixel API on a 0.1 degree lattice
(about 11 km) and the point metrics are bilinearly interpolated to the 2 km
analysis grid.  This keeps the climate fetch bounded (2.5k point requests)
while preserving regional gradients; it is deliberately coarser than the
native 1 km Daymet product and is recorded as a limitation in the target spec.
"""
import os, sys, json, time, math
import numpy as np
import requests
import concurrent.futures as cf
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

START, END = "2014-01-01", "2024-12-31"
STEP = 0.1
LON1D = np.round(np.arange(C.WEST, C.EAST+1e-9, STEP), 3)
LAT1D = np.round(np.arange(C.SOUTH, C.NORTH+1e-9, STEP), 3)
LON, LAT = np.meshgrid(LON1D, LAT1D)
PLON=LON.ravel(); PLAT=LAT.ravel()
VARS="tmax,tmin,prcp,vp,srad,dayl,swe"

def point_metrics(lat, lon):
    params={"lat":f"{lat:.4f}","lon":f"{lon:.4f}","vars":VARS,
            "start":START,"end":END,"format":"json"}
    for attempt in range(6):
        try:
            r=requests.get("https://daymet.ornl.gov/single-pixel/api/data",
                           params=params,timeout=180)
            if r.status_code in (429,500,502,503,504):
                time.sleep(2*(attempt+1)); continue
            r.raise_for_status(); d=r.json()["data"]; break
        except Exception:
            time.sleep(2*(attempt+1))
    else:
        raise RuntimeError(f"daymet failed {lat},{lon}")
    year=np.array(d["year"],dtype=int); yday=np.array(d["yday"],dtype=int)
    def g(k):
        a=np.array(d[k],dtype="float32")
        a[a<=-9998]=np.nan
        return a
    tmax=g("tmax (deg c)"); tmin=g("tmin (deg c)"); prcp=g("prcp (mm/day)")
    vp=g("vp (Pa)"); srad=g("srad (W/m^2)"); dayl=g("dayl (s)"); swe=g("swe (kg/m^2)")
    es=0.6108*np.exp(17.27*tmax/(tmax+237.3))  # kPa
    vpd=es-vp/1000.0
    rh=np.clip(vp/(es*1000.0)*100.0,0,100)
    names=None; rows=[]
    for y in C.FEATURE_YEARS:
        Y=(year==y)
        spring=Y&(yday>=60)&(yday<=151)
        summer=Y&(yday>=152)&(yday<=243)
        fall=Y&(yday>=244)&(yday<=304)
        season=Y&(yday>=105)&(yday<=273)
        snowmelt=Y&(yday>=60)&(yday<=151)
        winter=((year==y-1)&(yday>=305))|((year==y)&(yday<=59))
        f={}
        f["tmax_mean"]=np.nanmean(tmax[Y]); f["tmax_p95"]=np.nanpercentile(tmax[Y],95)
        f["tmin_mean"]=np.nanmean(tmin[Y]); f["tmin_min"]=np.nanmin(tmin[Y])
        f["precip_sum"]=np.nansum(prcp[Y]); f["precip_spring"]=np.nansum(prcp[spring])
        f["precip_summer"]=np.nansum(prcp[summer]); f["precip_fall"]=np.nansum(prcp[fall])
        f["precip_winter"]=np.nansum(prcp[winter])
        f["gdd"]=np.nansum(np.maximum(0.5*(tmax+tmin)-5.0,0)[Y])
        f["frost_days"]=np.nansum((tmin[Y]<0).astype("float32"))
        f["dry_days"]=np.nansum((prcp[Y]<1.0).astype("float32"))
        spell=cur=0.0
        for di in np.where(season)[0]:
            cur=cur+1.0 if prcp[di]<1.0 else 0.0
            spell=max(spell,cur)
        f["dry_spell"]=spell
        f["vpd_mean"]=np.nanmean(vpd[season]); f["vpd_max"]=np.nanmax(vpd[season])
        f["rh_mean"]=np.nanmean(rh[season])
        f["srad_mean"]=np.nanmean(srad[season]); f["srad_summer"]=np.nanmean(srad[summer])
        f["dayl_mean"]=np.nanmean(dayl[season])
        f["swe_mean"]=np.nanmean(swe[winter]); f["swe_max"]=np.nanmax(swe[winter])
        f["snow_free_days"]=np.nansum((swe[snowmelt]<=0.0).astype("float32"))
        if names is None: names=sorted(f)
        rows.append([f[k] for k in names])
    return np.array(rows,dtype="float32"), names

def fetch_all():
    p=os.path.join(C.DATA,"daymet_point_metrics.npz")
    if os.path.exists(p):
        z=dict(np.load(p,allow_pickle=True)); return z["names"].tolist(), z["values"]
    n=len(PLON); vals=None; names=None
    def work(i):
        v,nm=point_metrics(PLAT[i],PLON[i]); return i,v,nm
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(work,i) for i in range(n)]
        done=0
        for fu in cf.as_completed(futs):
            i,v,nm=fu.result(); done+=1
            if vals is None:
                names=nm; vals=np.full((n,len(C.FEATURE_YEARS),len(nm)),np.nan,"float32")
            vals[i]=v
            if done%100==0: print("daymet",done,"/",n,flush=True)
    np.savez_compressed(p,lon=PLON,lat=PLAT,names=np.array(names),values=vals)
    return names,vals

def to_grid(names,vals):
    g=C.grid(); h,w=g["height"],g["width"]
    _,_,lon,lat=C.cell_centers_lonlat(g)
    inside=C.study_mask(g)
    pts=np.column_stack([lat[inside],lon[inside]])
    out={}
    for yi,y in enumerate(C.FEATURE_YEARS):
        for mi,m in enumerate(names):
            z=vals[:,yi,mi].reshape(LAT.shape)
            f=RegularGridInterpolator((LAT[:,0],LON[0,:]),z,method="linear",
                                      bounds_error=False,fill_value=np.nan)
            flat=np.full(h*w,np.nan,dtype="float32")
            flat[inside.ravel()]=f(pts).astype("float32")
            out[f"{m}_{y}"]=flat.reshape(h,w)
    np.savez_compressed(os.path.join(C.DATA,"climate_features_2km.npz"),**out)
    print("climate grid layers",len(out),"inside cells",int(inside.sum()))

if __name__=="__main__":
    names,vals=fetch_all(); print(len(names),names)
    to_grid(names,vals)
