#!/usr/bin/env python3
"""Past-fire features for the I4 next-year fire target.

All features are known by the end of feature year t: fire in t, t-1, t-2,
a 3-year count, the capped years-since-fire, and local 3x3/5x5 fire density.
Labels are 2 km NBAC burn fractions; a cell counts as burned at >=1%.
"""
import os, sys
import numpy as np
from scipy.ndimage import uniform_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i4_common as C

OUT = os.path.join(C.DATA, "fire_history_features_2km.npz")
THRESH = 0.01

def build():
    z = np.load(os.path.join(C.DATA, "labels_2km.npz"))
    lab = {int(k): z[k] for k in z.files}
    inside = C.study_mask().astype("float32")
    out = {}
    for t in C.FEATURE_YEARS:
        f0 = (lab[t] >= THRESH).astype("float32")
        f1 = (lab[t-1] >= THRESH).astype("float32")
        f2 = (lab[t-2] >= THRESH).astype("float32")
        ysf = np.full(f0.shape, 3.0, dtype="float32")
        ysf[f2 > 0] = 2.0
        ysf[f1 > 0] = 1.0
        ysf[f0 > 0] = 0.0
        out[f"fire_t_{t}"] = f0
        out[f"fire_t1_{t}"] = f1
        out[f"fire_t2_{t}"] = f2
        out[f"fire_count_3yr_{t}"] = f0 + f1 + f2
        out[f"fire_years_since_{t}"] = ysf
        def density(size):
            num = uniform_filter(f0*inside, size=size, mode="constant", cval=0.0)
            den = uniform_filter(inside, size=size, mode="constant", cval=0.0)
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0).astype("float32")
        out[f"fire_density_3x3_{t}"] = density(3)
        out[f"fire_density_5x5_{t}"] = density(5)
        print(f"fire {t}: fires={int(f0.sum())} cells, density_max={out[f'fire_density_5x5_{t}'].max():.3f}", flush=True)
    np.savez_compressed(OUT, **out)
    print("saved", OUT, len(out), "layers")

if __name__ == "__main__":
    build()
