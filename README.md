### Problem:

From 'Poor Economics' by Banerjee and Duflo

Drought $\rightarrow$ lower agricultural output / labor demand $\rightarrow$ lower household income $\rightarrow$ worse nutrition/debt/school dropout/migration/stress/social breakdowns

**Climate change exacerbates the logic**

* Low income households in many countries rely more on rainfed agriculture, farm labor, climate-sensitive food markets.
* Higher income households/farmers with irrigation, storage, insurance, savings, access too credit, are less exposed.
* Applying predictive models to such problem $\rightarrow$ help farmers, insurers, water managers, governments act promptly and selectively $\rightarrow$ Prioritize resouce allocation, trigger emergency seed/financial support, index-based payout by insurer $\rightarrow$ help reduce severity of chain of effects

### Drought:
1. Rainfall deficits
2. Soil moisture drops
3. Plant stress rises
4. Yield potential declines
5. Losses

---

* Ethiopia $\rightarrow$ IMF projects real GPD growth rate of $> 7$% for 2026 while government projects $10.2$% agricultural yields
* Agriculture makes up $35$% of GDP and represents $~75$% total export earnings with coffee as best trade
* Agricultural sector employs $~70$% of total population
* $~80$% population lives in rural areas
  * $~75$% of farmers are small family farmers
* Stable crops grown to meet subsistence needs are teff, maize, wheat, barley, sorghum

---

[DATA SOURCE: SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture V009](data_processing/data_source.md)

$1$ pixel $=$ $36km$ x $36km$ $(1296km^2)$

$1$ degree latitude/longitude ~$111km$

![SMAP soil moisture](data_processing/plots/sm_single_channel.png)

---

### West Arsi, Ethiopia:

Located in the Oromia region and part of ***Ethiopia's wheat belt***. It relies heavily on rain-fed agriculture rather than irrigation therefore crop yields in the zone are extremely sensitive to root-zone soil moisture deficits

*Root zone soil moisture*: Water that is available to plants usually considered to be in the upper 200 cm of soil. An accurate depiction can provide valuable insights for agricultural monitoring, weather, prediction, drought/flood warnings. 

*Surface soil moisture*: Shallow near-surface layer, often upper 5-10 cm.

Soil moisture is also dependent on soil type and vegetation.

[Source](https://www.drought.gov/topics/soil-moisture)

---

* **Swath gaps:** Spaces between satellite observation tracks where no data is collected during a given pass or compositing period.
* Retrieval ~6AM local overpass for passive soil-moisture as early morning conditions are thermally more uniform.

---

### Data pre-processing:

**1. Spatial window:** Extract a $64 \times 64$ pixel window ($\approx 2{,}300$ km) centered on West Arsi, Ethiopia (lat $7.25°$N, lon $39.0°$E) from global EASE-2 grid. Center pixel located using squared Euclidean distance on lat/lon arrays after masking $-9999$ (INVALID) fill values.

**2. Multi feature subset:** $8$ channels/pixel/day:

| Channel | Source group | Description |
|---|---|---|
| `soil_moisture_am` | AM | 6 AM passive retrieval ($cm^3/cm^3$) |
| `soil_moisture_pm` | PM | 6 PM passive retrieval ($cm^3/cm^3$) |
| `surface_temp_am` | AM | Effective soil temperature (K) |
| `surface_temp_pm` | PM | Effective soil temperature (K) |
| `vegetation_water` | AM | Vegetation water content ($kg/m^2$) |
| `vegetation_opacity` | AM | Vegetation optical depth (unitless) |
| `tb_polarization_diff` | AM | $T_{b,V} - T_{b,H}$ brightness-temperature difference (K) |
| `bulk_density` | AM | Soil bulk density ($g/cm^3$) |

Quality filtering: soil-moisture channels masked if `retrieval_qual_flag` bit-0 $= 1$. All channels masked at $-9999$ fill-value sentinel.

**3. Temporal compose:** Non-overlapping $3$-day `nanmean` composites ($3{,}971$ daily $\rightarrow$ $1{,}324$ composites). Improves per pixel coverage from $\sim 20$–$27$% to $\sim 60$–$65$% by aggregating across orbital swath gaps.

**4. Temporal imputation:** Per-pixel linear interpolation along time axis to fill remaining composite gaps using *nearest valid observations* before and after. Edge composites forward/backward filled up to $5$ composites ($15$ days because 3-day layer used for composites). Permanently unobserved pixels (ocean, lakes) 0-filled.

**5. Array to Tensor:** NumPy arrays converted to PyTorch `float32` tensors. Single-channel tensor shape $(T, Lat, Lon)$; multi feature tensor shape $(T, C, Lat, Lon)$. $T = 1{,}324$, $C = 8$, $Lat = Lon = 64$. 

Stored .npz and .pt files

---

### Temporal Train/Validation/Test splits

* **Train set**(gradients and weights): Yrs 2015 - 2022 
* **Validation set**(hyperparameter tuning, stop training before overfitting): Yrs 2023 - 2024
* **Test**: Yrs 2025-2026

```torch.utils.data.Dataset``` 

Input window must capture **Soil Moisture Memory** of region $\rightarrow$ Amount of time it takes fo an anomaly (e.g., heavcy rainstorm, extreme heatwave) to dissipate and for the soil to return to **climatological baseline**.

Upper top centimeters of soil respond much faster to weather than root zone. SUrface soil moisture changes on timescales of days to a few weeks (depending on other factors such as atmospheric demand, vegetation, soil texture). Root zone moisture can persist for much longer (weeks to months).

Input Processing:

**Pixel-wise normalization**: helps put pixel intensities on a scale so model sees inputs with a distribution similar to what it was trained on 

