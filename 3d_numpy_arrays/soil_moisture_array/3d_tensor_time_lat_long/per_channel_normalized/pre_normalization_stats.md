### Pre-Normalization Per-Channel Statistics

Source: `multifeature_tensor_T_C_Lat_Lon.pt` — shape $(T{=}1324,\ C{=}8,\ Lat{=}64,\ Lon{=}64)$

Elements per channel: $5{,}423{,}104$

| Ch | Feature | Min | p1 | p5 | p25 | Median | p75 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | soil_moisture_am | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1006 | 0.1688 | 0.3315 | 0.4508 | 0.6213 |
| 1 | soil_moisture_pm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0857 | 0.1534 | 0.3133 | 0.4241 | 0.6008 |
| 2 | surface_temp_am | 0.0000 | 0.0000 | 0.0000 | 295.72 | 300.90 | 303.87 | 307.95 | 310.13 | 315.64 |
| 3 | surface_temp_pm | 0.0000 | 0.0000 | 0.0000 | 299.58 | 305.42 | 309.18 | 314.21 | 317.04 | 323.41 |
| 4 | vegetation_water | 0.0000 | 0.0000 | 0.0000 | 0.0015 | 0.4905 | 2.2800 | 5.8947 | 17.8981 | 18.3304 |
| 5 | vegetation_opacity | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0679 | 0.2556 | 0.6002 | 1.2376 | 2.4857 |
| 6 | tb_polarization_diff | -39.6061 | 1.7895 | 8.3261 | 15.5240 | 21.5363 | 30.9830 | 40.0324 | 45.9210 | 130.4632 |
| 7 | bulk_density | 0.0000 | 0.0000 | 0.0000 | 1.1683 | 1.3404 | 1.4516 | 1.5348 | 1.6080 | 1.6892 |

#### Non-zero statistics

| Ch | Feature | Non-zero mean | Non-zero std | IQR | Zeros | Always-zero pixels |
|---|---|---|---|---|---|---|
| 0 | soil_moisture_am | 0.1630 | 0.0981 | 0.1688 | 30.81% | 1259/4096 (30.7%) |
| 1 | soil_moisture_pm | 0.1490 | 0.0946 | 0.1534 | 30.81% | 1259/4096 (30.7%) |
| 2 | surface_temp_am | 301.09 | 4.6618 | 8.1513 | 12.82% | 523/4096 (12.8%) |
| 3 | surface_temp_pm | 305.97 | 5.4139 | 9.5997 | 12.80% | 523/4096 (12.8%) |
| 4 | vegetation_water | 2.1695 | 3.4794 | 2.2785 | 22.65% | 809/4096 (19.8%) |
| 5 | vegetation_opacity | 0.2397 | 0.2633 | 0.2556 | 29.78% | 789/4096 (19.3%) |
| 6 | tb_polarization_diff | 23.5121 | 10.8371 | 15.4591 | 0.05% | 0/4096 (0.0%) |
| 7 | bulk_density | 1.3559 | 0.1375 | 0.2833 | 15.60% | 637/4096 (15.6%) |

#### Outlier summary (IQR method)

| Ch | Feature | Mild (1.5×IQR) | Extreme (3×IQR) | Tail |
|---|---|---|---|---|
| 0 | soil_moisture_am | 1.56% high | 0% | Mild right |
| 1 | soil_moisture_pm | 1.88% high | 0% | Mild right |
| 2 | surface_temp_am | 12.82% low (zeros) | 12.82% low (zeros) | Zeros only |
| 3 | surface_temp_pm | 12.80% low (zeros) | 12.80% low (zeros) | Zeros only |
| 4 | vegetation_water | 5.14% high | 3.79% high | **Heavy right** |
| 5 | vegetation_opacity | 4.28% high | 2.45% high | **Heavy right** |
| 6 | tb_polarization_diff | 0.35% | ~0% | Clean |
| 7 | bulk_density | 15.62% low (zeros) | 15.60% low (zeros) | Zeros only |

Zeros are spatially fixed (permanently unobserved pixels: ocean, lakes, coverage gaps) — not missing-at-random.
