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

### Other Models

- **CAMx**: Comprehensive Air Quality Model with extensions (`camx`)
- **CHIMERE**: Multi-scale chemistry-transport model (`chimere`)
- **RAQMS**: Realtime Air Quality Modeling System (`raqms`)
