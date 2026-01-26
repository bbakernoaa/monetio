# Observations

This section describes how to use MONETIO to load supported observational datasets.

## Unified Access

We recommend using the unified `monetio.load()` function for all observational data:

```python
import monetio as mio
import pandas as pd

# Load AirNow data
dates = pd.date_range(start='2018-05-01', end='2018-05-05', freq='h')
df = mio.load("airnow", files=dates)
```

## Supported Observation Networks

### AirNow
Near real-time air quality data for the United States.
- **Source ID**: `airnow`
- **Variables**: `O3`, `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, etc.

### EPA AQS
Historical air quality data from the U.S. EPA.
- **Source ID**: `aqs`

### AERONET
Aerosol Robotic Network (global).
- **Source ID**: `aeronet`

### OpenAQ
Global air quality data platform.
- **Source ID**: `openaq`

### Integrated Surface Database (ISH)
Global hourly and synoptic surface observations.
- **Source ID**: `ish` or `ish_lite`

### Other Networks
- **NADP**: National Atmospheric Deposition Program (`nadp`)
- **CRN**: Climate Reference Network (`crn`)
- **CEMS**: Continuous Emission Monitoring Systems (`cems`)
- **IMPROVE**: Interagency Monitoring of Protected Visual Environments (`improve`)
