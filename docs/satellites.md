# Satellites

MONETIO provides readers for several satellite instruments and reanalysis products.

## Unified Access

We recommend using the unified `monetio.load()` function:

```python
import monetio as mio

# Load MERRA2 data
ds = mio.load("merra2", files="MERRA2_400.tavg1_2d_slv_Nx.20230101.nc4")
```

## Supported Satellites and Reanalysis

### MERRA-2

Modern-Era Retrospective analysis for Research and Applications, Version 2.

- **Source ID**: `merra2`

### MODIS

Moderate Resolution Imaging Spectroradiometer.

- **Source ID**: `modis_l2`, `modis_ornl`, `nasa_modis`

### NESDIS VIIRS

National Environmental Satellite, Data, and Information Service (NESDIS) Visible Infrared Imaging Radiometer Suite (VIIRS).

- **Source ID**: `nesdis_edr_viirs`, `nesdis_eps_viirs`, `nesdis_viirs_jrr` (or `viirs_jrr`), `nesdis_frp`

### OMPS

Ozone Mapping and Profiler Suite.

- **Source ID**: `omps`, `omps_nadir`

### TROPOMI

Tropospheric Monitoring Instrument.

- **Source ID**: `tropomi`

### TEMPO

Tropospheric Emissions: Monitoring of Pollution.

- **Source ID**: `tempo`

### GOES

Geostationary Operational Environmental Satellite.

- **Source ID**: `goes`

### MOPITT

Measurements Of Pollution In The Troposphere.

- **Source ID**: `mopitt`
