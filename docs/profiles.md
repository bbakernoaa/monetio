# Profiles

MONETIO supports various vertical profile data sources, including lidars, ozonesondes, and aircraft measurements.

## Unified Access

We recommend using the unified `monetio.load()` function:

```python
import monetio as mio

# Load IAGOS aircraft data
ds = mio.load("iagos", files="iagos_data.nc")
```

## Supported Profile Networks

### IAGOS

In-service Aircraft for a Global Observing System.

- **Source ID**: `iagos`

### ACTRIS/EBAS

European Research Infrastructure for the observation of Aerosol, Clouds and Trace Gases.

- **Source ID**: `actris`

### GML Ozonesonde

Global Monitoring Laboratory (GML) Ozonesondes.

- **Source ID**: `gml_ozonesonde`

### EARLINET

European Aerosol Research Lidar Network.

- **Source ID**: `earlinet`

### MPLNET

Micro-Pulse Lidar Network.

- **Source ID**: `mplnet`

### TOLNET

Tropospheric Ozone Lidar Network.

- **Source ID**: `tolnet`

### IGRA2

Integrated Global Radiosonde Archive Version 2.

- **Source ID**: `igra2`

### GEOMS

Generic Earth Observation Metadata Standard (common format for many NDACC/Pandora instruments).

- **Source ID**: `geoms`

### ICARTT

International Consortium for Atmospheric Research on Transport and Transformation (common airborne format).

- **Source ID**: `icartt`

### UMBC Aerosol

UMBC Aerosol lidar data.

- **Source ID**: `umbc_aerosol`
