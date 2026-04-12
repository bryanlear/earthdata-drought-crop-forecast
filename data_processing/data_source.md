
## Data Source

- **Product**: SPL3SMP (SMAP Level 3 Soil Moisture, Passive)
- **Version**: V009 (R19240)
- **Provider**: NASA NSIDC DAAC
- **Satellite**: SMAP (Soil Moisture Active Passive), L-band (1.41 GHz) radiometer
- **Temporal coverage**: 2015-03-31 to 2026-04-09 (daily)
- **Spatial coverage**: Global
- **Spatial resolution**: 36 km (EASE-Grid 2.0)
- **Grid dimensions**: 406 × 964

## File Format

Each `.h5` (HDF5) file = one day of global data. Patter name:
```
SMAP_L3_SM_P_{YYYYMMDD}_R19240_{VVV}.h5
```
- `YYYYMMDD`: observation date
- `R19240`: processing baseline
- `VVV`: version number (001, 002, etc. — higher = reprocessed)

## Structure

$3$ top-level groups per file:

### `Metadata/`
Acquisition info, data quality, grid definition, processing lineage.

### `Soil_Moisture_Retrieval_Data_AM/` (6 AM descending pass)

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `soil_moisture` | (406, 964) | float32 | Primary soil moisture retrieval (cm³/cm³) |
| `soil_moisture_error` | (406, 964) | float32 | Uncertainty estimate |
| `latitude` / `longitude` | (406, 964) | float32 | Geolocation |
| `surface_temperature` | (406, 964) | float32 | Surface temp (from GEOS-5 model) |
| `vegetation_water_content` | (406, 964) | float32 | Vegetation water content |
| `vegetation_opacity` | (406, 964) | float32 | Vegetation optical depth |
| `tb_h_corrected` / `tb_v_corrected` | (406, 964) | float32 | Brightness temperature (H/V polarization) |
| `retrieval_qual_flag` | (406, 964) | uint16 | Quality flag (bitfield) |
| `surface_flag` | (406, 964) | uint16 | Surface type classification |
| `freeze_thaw_fraction` | (406, 964) | float32 | Frozen ground fraction |
| `clay_fraction` | (406, 964) | float32 | Clay fraction (static ancillary) |
| `bulk_density` | (406, 964) | float32 | Soil bulk density (static ancillary) |
| `landcover_class` | (406, 964, 3) | uint8 | IGBP land cover (top 3 classes) |

### `Soil_Moisture_Retrieval_Data_PM/` (6 PM ascending pass)
Same variables as AM with `_pm` suffix. Uses dual-channel algorithm (`_dca`) for the primary retrieval.

Higher version supersede earlier ones.

---



