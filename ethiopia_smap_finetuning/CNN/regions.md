### Somalia
- Northwest agro-pastoral belt
- Northeast dry pastoral belt
- Central pastoral belt
- Juba riverine belt
- Shabelle riverine belt
- Southern rainfed agro-pastoral belt

### Eritrea
- Central highlands
- Western lowlands
- Eastern lowlands
- Southwestern lowlands
- Escarpment / Green belt 
  
### Kenya
- Northern arid pastoral belt
- Eastern semi-arid belt
- Central highlands
- Rift valley highlands
- Western high-rainfall farming belt
- Coastal lowllands

### Sudan
- Desert belt
- Semi-desert belt
- Low-rainfall savanna
- High-rainfall savanna
- Gezira / Irrigated nile belt
- Blue Nile / Rainfed mechanized belt

### South Sudan
- Greenbelt
- Ironstone plateau
- Flood plains
- Nile-sobat river zone
- Eastern pastoral drylands
- Hills and Mountains

### Ethiopia
- Afar
- Amhara
- Oromia
- Somali
- West Arsi

### Djubouti
- Coastal arid belt
- Inland pastoral drylands
- Pasis / wasi irrigated pockets
---


```CNN/SMAP_regions/build_monthly_smap.py```

* **Crop**: 
  * Dissolve regions admin-1 polygons into 1 boundary
  * Find smallest axis-aligned SMAP row/col bounding box that contains the polygon
  * Extract rectangular window from every SMAP feature layer
  * Each region gets its own native window size:
    * Eritrea ```western_lowlands``` $\rightarrow$ ```5 x 7```
    * Kenya ```coastal_lowlands``` $\rightarrow$ ```8x11```
    * Somalia ```central_pastoral``` $\rightarrow$ ```11x14```
    * Sudann ```desert``` $\rightarrow$ ```15 x 30```
    * ...
* **Mask**:
  * v  



---


`CNN/SMAP_regions/build_country_cubes.py`

* **Crop**" per-country 64×64 SMAP window:
  * Build composite (latitute|longitude) grids by sampling 30 SMAP files spread across AM + PM passes for ~100% grid coverage
  * For each country, find natural SMAP pixel bounding box enclosing all its admin-1 polygons (with 0.8° margin)
  * Expand symmetrically to a fixed 64×64 window by padding the shorter dimension; clamp to grid edges if needed
  * Hard-coded row/col anchors on 406×964 EASE-Grid 2.0 (original SMAP).

    | Country | r0 | r1 | c0 | c1 | lat range | lon range |
    |---|---|---|---|---|---|---|
    | eritrea | 118 | 182 | 557 | 621 | [6.1, 24.6] | [28.2, 51.7] |
    | kenya | 171 | 235 | 552 | 616 | [−8.9, 8.9] | [26.3, 49.9] |
    | somalia | 153 | 217 | 574 | 638 | [−3.8, 14.1] | [34.5, 58.1] |
    | south_sudan | 144 | 208 | 531 | 595 | [−1.3, 16.8] | [18.5, 42.0] |
    | sudan | 116 | 180 | 531 | 595 | [6.7, 25.2] | [18.5, 42.0] |

* **SMAP channels**: 8 features / daily file composited to monthly ```nanmean```:
  * `soil_moisture_am` / `soil_moisture_pm` — quality-filtered (retrieval_qual_flag bit 0)
  * `surface_temp_am` / `surface_temp_pm`
  * `vegetation_water_content`, `vegetation_opacity`, `bulk_density`
  * `tb_polarization_diff` — derived: `tb_v_corrected − tb_h_corrected` (AM)
  * Fill value −9999 → NaN before compositing

* **CHIRPs**: Match monthly 0.05° TIFs to the SMAP 64×64 grid:
  * Read only geographic sub-window of TIF that overlaps SMAP window (`rasterio.windows.from_bounds`)
  * For each SMAP cell center (lat, lon), compute NaN-aware mean of 7×7 CHIRPS pixel block centered on cell (~1 SMAP footprint at 0.05° res)
  * Sum/count decomposition using `scipy.ndimage.uniform_filter`
  * Output: `(T, 64, 64)` float32 `chirps_cube`, units mm/month

* **Region masks**: pixel-level membership for each agro-ecological zone:
  * GADM level-1 admin units into zone polygons (`COUNTRIES` dict)
  * For each zone collect all polygon exterior rings from GeoJSON files
  * Test all valid SMAP cell centers (lon, lat) against each ring via `matplotlib.path.Path.contains_points`
  * Output: `(R, 64, 64)` bool array `region_mask`; minimum cells per zone across all 27 regions = 16 (Eritrea `southwestern_lowlands`)

* **Output**: `.npz` / country (`CNN/SMAP_regions/`)

  | Array | Shape | dtype | Description |
  |---|---|---|---|
  | `smap_cube` | `(T, 8, 64, 64)` | float32 | Monthly SMAP composites |
  | `chirps_cube` | `(T, 64, 64)` | float32 | Precip mm resampled to SMAP grid |
  | `region_mask` | `(R, 64, 64)` | bool | Pixel membership per zone |
  | `region_names` | `(R,)` | str | Slugs matching CHIRPS CSV filenames |
  | `feature_names` | `(8,)` | str | SMAP channel names |
  | `dates` | `(T,)` | str | ISO date strings |
  | `lat_grid` / `lon_grid` | `(64, 64)` | float32 | SMAP cell-centered coordinates |

  T = 132 months (April 2015 – March 2026) for all 5 countries

  --- If labels are noisy, model will memorize visual patterns

### Model (before implementing time-series CV 5 fold)

- Current reference setup uses one `(country, month, region)` crop per sample, 10 input channels `(8 SMAP + 1 CHIRPS + 1 region mask)`, month-of-year sine/cosine encoding, and `width=0.25`
- Current sample counts for this setup: `3255 train / 420 val / 945 test`
- A 3-seed sweep with month encoding on (`seeds 43, 44, 45`) averaged: `macro F1=0.647`, `drought precision=0.343`, `drought recall=0.577`, `drought F1=0.430`, `accuracy=0.783`
- Removing month encoding hurt on average over the same seeds: `macro F1=0.620`, `drought precision=0.310`, `drought recall=0.530`, `drought F1=0.390`, `accuracy=0.757`
- Increasing width from `0.25` to `0.5` gave only a small average change: `macro F1 0.647 -> 0.660`, `drought F1 0.430 -> 0.433`, `accuracy 0.783 -> 0.813`, but shifted the operating point toward higher drought precision `0.343 -> 0.420` and lower drought recall `0.577 -> 0.500`
- Decision: keep month encoding enabled and keep `width=0.25` as the default because the `0.5` model is only marginally better on aggregate score while missing more drought cases\

