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

- **Source ID**: `hysplit`

### WRF-Chem

Weather Research and Forecasting model coupled with Chemistry.

- **Source ID**: `wrfchem`

### FV3-Chem (UFS-Chem)

Unified Forecast System with Chemistry.

- **Source ID**: `fv3chem` or `ufs`

### Other Models

- **CAMx**: Comprehensive Air Quality Model with extensions (`camx`)
- **CHIMERE**: Multi-scale chemistry-transport model (`chimere`)
- **RAQMS**: Realtime Air Quality Modeling System (`raqms`)
