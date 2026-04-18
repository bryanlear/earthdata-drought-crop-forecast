
```torch.utils.data.Dataset``` 

Input window must capture **Soil Moisture Memory** of region $\rightarrow$ Amount of time it takes fo an anomaly (e.g., heavcy rainstorm, extreme heatwave) to dissipate and for the soil to return to **climatological baseline**.

Upper top centimeters of soil respond much faster to weather than root zone. SUrface soil moisture changes on timescales of days to a few weeks (depending on other factors such as atmospheric demand, vegetation, soil texture). Root zone moisture can persist for much longer (weeks to months).



SMAP L3 36km channels live on differents scales and distributions (e.g., soil moisture L $0-0.6m^3/m^3$, surface temp: $200-323K$)
- Normalize channels before feeding to SatMAE 

methods:
```
.npz (imputed)          numpy_to_tensor.py          split_tensors.py
─────────────── ──────────────────────────► ──────────────────────────────────────►

smap_multifeature       multifeature_tensor         multifeature_{train,val,test}.pt
 _west_arsi_3day         _T_C_Lat_Lon.pt            ├─ tensor      (raw, clipped)
 _imputed.npz           (1324, 8, 64, 64)           ├─ tensor_norm (z-scored)
                         float32                     ├─ dates
│ np arrays             │ PyTorch tensors            ├─ spatial_mask
│ with 0-filled         │ unchanged from .npz        └─ feature_names
│ ocean/lake pixels     │
                                                    multifeature_norm_stats.pt
                        ┌──────────────────────────► ├─ channel_stats {mean, std,
                        │  1. Spatial mask:          │    clip_lo, clip_hi}
                        │     always-zero pixels     ├─ spatial_mask (8,64,64) bool
                        │     identified per channel └─ feature_names
                        │
                        │  2. Clip Ch4,Ch5 at
                        │     train p1/p99
                        │
                        │  3. Per-channel z-score
                        │     (train non-zero μ/σ)
                        │
                        │  4. Split by year:
                        │     train 2015-2022
                        │     val   2023-2024
                        │     test  2025-2026
                        └──────────────────────────►
```
