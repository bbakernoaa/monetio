# Models

MONETIO supports reading output from several major atmospheric chemistry and transport models.

## Unified Access

We recommend using the unified `monetio.load()` function:

```python
import monetio as mio

# Load CMAQ data
ds = mio.load("cmaq", files="aqm.t12z.aconc.ncf")
```

## Supported Models

### CMAQ

Community Multiscale Air Quality Model.

- **Source ID**: `cmaq`

### HYSPLIT

Hybrid Single-Particle Lagrangian Integrated Trajectory model.

- **Source ID**: `hysplit` (Concentration) or `hytraj` (Trajectories)
- **Features**:
    - Lazy loading via Dask.
    - Automatic grid continuity fixing.
    - Optimized mass loading calculations.

### WRF-Chem

Weather Research and Forecasting model coupled with Chemistry.

- **Source ID**: `wrfchem`

### UFS-AQM

Unified Forecast System with Chemistry.

- **Source ID**: `ufs`

### NCEP GRIB

National Centers for Environmental Prediction (NCEP) GRIB2 model outputs.

- **Source ID**: `ncep_grib`: Generic NCEP GRIB2 reader.
- **Source ID**: `gfs`: Global Forecast System.
- **Source ID**: `gefs`: Global Ensemble Forecast System.
- **Source ID**: `gdas`: Global Data Assimilation System.
- **Source ID**: `rrfs`: Rapid Refresh Forecast System.
- **Source ID**: `grib2`: Standard GRIB2 files.
- **Recommended API**: Use `open_dataset(...)` with the grib2io xarray backend
    options (`filters`, `use_icechunk`, `storage_options`, etc.).

### NOAA verification precipitation and snowfall grids

The `prism`, `ccpa`, and `nohrsc` readers open local product files or URLs
directly through the shared Xarray driver. `files` accepts a path, list, glob,
HTTP URL, or supported object-store path; callers can therefore use either
archived local inputs or directly address a published file. These readers do
not search or enumerate remote archives.

```python
prism = mio.load("prism", files="/data/prism_daily.nc")
ccpa = mio.load("ccpa", files="/data/ccpa_06h.nc")
snow = mio.load("nohrsc", files="/data/nohrsc_snowfall.grib2", engine="grib2io")
```

Common product variables are given MONETIO names (`precipitation_amount`,
daily PRISM temperature fields, and `snowfall_amount` or
`snow_water_equivalent`). Native values and units, grid mapping, and available
validity or accumulation bounds are preserved. Use the product's variable
attributes to distinguish accumulation periods and snowfall depth from snow
water equivalent. GRIB inputs require an installed Xarray GRIB backend.

### ICAP-MME

International Cooperative for Aerosol Prediction Multi-Model Ensemble.

- **Source ID**: `icap_mme`

### PARDUMP

HYSPLIT Particle Dump files.

- **Source ID**: `pardump`

### Other Models

- **CAMx**: Comprehensive Air Quality Model with extensions (`camx`)
- **CHIMERE**: Multi-scale chemistry-transport model (`chimere`)
- **RAQMS**: Realtime Air Quality Modeling System (`raqms`)
