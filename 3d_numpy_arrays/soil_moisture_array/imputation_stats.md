# Imputation Stats — `smap_multifeature_africa_3day_imputed.npz`

**Source:** `smap_multifeature_africa_3day.npz`
**Shape:** (1324, 8, 241, 187) — T × C × H × W
**Total elements:** 477,349,664

## Pre-Imputation NaN Coverage

| Channel              | NaN % |
|----------------------|------:|
| soil_moisture_am     | 55.8% |
| soil_moisture_pm     | 55.2% |
| surface_temp_am      | 39.9% |
| surface_temp_pm      | 40.1% |
| vegetation_water     | 42.4% |
| vegetation_opacity   | 44.8% |
| tb_polarization_diff |  3.1% |
| bulk_density         | 42.0% |
| **Overall**          | **40.4%** (192,947,324 / 477,349,664) |

## Post-Imputation

| Channel              | NaN | Value Range            |
|----------------------|-----|------------------------|
| soil_moisture_am     |   0 | [0.0200, 0.6279]       |
| soil_moisture_pm     |   0 | [0.0200, 0.6208]       |
| surface_temp_am      |   0 | [268.7490, 318.0892] K |
| surface_temp_pm      |   0 | [270.2556, 326.4023] K |
| vegetation_water     |   0 | [0.0000, 18.4737]      |
| vegetation_opacity   |   0 | [-0.0000, 2.4857]      |
| tb_polarization_diff |   0 | [-82.4685, 142.2126] K |
| bulk_density         |   0 | [0.0272, 1.7166]       |

**Filled:** 192,947,324 pixels (100.0% of NaNs)
**Output size:** 945.2 MB
