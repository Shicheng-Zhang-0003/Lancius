#!/usr/bin/env python3
"""Static 2 km features: topography (NASADEM) and land cover (WorldCover)."""
import os, sys, json
import numpy as np
import rasterio
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import reproject, Resampling
from affine import Affine
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

SUB_RES = 500.0

def subgrid(g):
    w = g["width"]*int(C.RES/SUB_RES); h = g["height"]*int(C.RES/SUB_RES)
    t = rasterio.transform.from_origin(g["bounds"][0], g["bounds"][3], SUB_RES, SUB_RES)
    return w,h,t

def nasadem_elevation(g):
    p=os.path.join(C.DATA,"nasadem_500m.npy")
    if os.path.exists(p): return np.load(p)
    sw,sh,st=subgrid(g)
    out=np.full((sh,sw),np.nan,"float32")
    cat=Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items=list(cat.search(collections=["nasadem"],bbox=[C.WEST,C.SOUTH,C.EAST,C.NORTH],
                          limit=100).items())
    print("nasadem tiles",len(items))
    for it in items:
        ds=rasterio.open(pc.sign(it.assets["elevation"]).href)
        ow=oh=225
        arr=ds.read(1,out_shape=(oh,ow),resampling=Resampling.bilinear).astype("float32")
        arr[arr==ds.nodata]=np.nan
        st_src=ds.transform*Affine.scale(ds.width/ow,ds.height/oh)
        tmp=np.full((sh,sw),np.nan,"float32")
        reproject(arr,tmp,src_transform=st_src,src_crs=ds.crs,
                  dst_transform=st,dst_crs=C.CRS,resampling=Resampling.bilinear,
                  src_nodata=np.nan,dst_nodata=np.nan)
        m=np.isfinite(tmp)&~np.isfinite(out)
        out[m]=tmp[m]
        ds.close()
    # fill small holes so gradients at tile seams are defined
    bad=~np.isfinite(out)
    if bad.any():
        idx=ndimage.distance_transform_edt(bad,return_distances=False,return_indices=True)
        out=out[tuple(idx)]
    np.save(p,out)
    return out

def topography(g=None):
    g=g or C.grid()
    elev=nasadem_elevation(g).astype("float64")
    dzdy,dzdx=np.gradient(elev,SUB_RES)
    slope=np.degrees(np.arctan(np.hypot(dzdx,dzdy))).astype("float32")
    aspect=np.arctan2(-dzdy,dzdx).astype("float32")
    f=int(C.RES/SUB_RES)
    def agg(a,how="mean"):
        h,w=g["height"],g["width"]
        a=a[:h*f,:w*f].reshape(h,f,w,f)
        return getattr(np,how)(a,axis=(1,3))
    return {
        "elevation":agg(elev,"mean").astype("float32"),
        "slope":agg(slope,"mean").astype("float32"),
        "aspect_sin":agg(np.sin(aspect),"mean").astype("float32"),
        "aspect_cos":agg(np.cos(aspect),"mean").astype("float32"),
        "ruggedness":agg(elev,"std").astype("float32"),
    }

def landcover(g=None):
    g=g or C.grid()
    wc=C.worldcover_fractions(g)
    out={name:wc[c].astype("float32") for c,name in C.CLASSES.items() if c in (10,20,30,40,50,60,80,90)}
    out["rangeland"]=sum(wc[c] for c in C.RANGELAND).astype("float32")
    built=out["built"]
    out["built_5km"]=ndimage.gaussian_filter(built,2.5).astype("float32")
    out["dist_built_km"]=(ndimage.distance_transform_edt(built<0.02)*C.RES/1000.0).astype("float32")
    return out

if __name__=="__main__":
    g=C.grid()
    topo=topography(g)
    lc=landcover(g)
    np.savez_compressed(os.path.join(C.DATA,"static_features_2km.npz"),**topo,**lc)
    print({k:(float(np.nanmin(v)),float(np.nanmax(v))) for k,v in {**topo,**lc}.items()})
