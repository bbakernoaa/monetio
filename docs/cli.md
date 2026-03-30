# Command Line Interface (CLI)

MONETIO provides a unified Command Line Interface (CLI) for retrieving and processing atmospheric data from various sources. The CLI allows you to download data, apply preprocessing, and save results to NetCDF or CSV formats without writing any Python code.

## Installation

The CLI is installed automatically when you install MONETIO. You can access it using the `monetio` command:

```bash
monetio --help
```

## Unified Loading

The `load` command is the primary entry point for all supported data sources. It dispatches to the internal `monetio.load` function.

### Basic Usage

```bash
monetio load SOURCE [OPTIONS]
```

Where `SOURCE` is any registered reader name (e.g., `cmaq`, `airnow`, `goes`, `aeronet`).

### Common Options

- `-d, --dates TEXT`: Date or date range. Supports single dates (`2023-01-01`) or ranges (`2023-01-01:2023-01-05`).
- `-o, --output PATH`: Path to save the output file.
- `-f, --files TEXT`: File path(s) or glob pattern(s) to read.
- `--lazy`: Use Dask for lazy loading (recommended for large satellite or model datasets).
- `--as-pandas`: Process and save the data as a CSV (tabular) instead of NetCDF (gridded).
- `-k, --kwargs KEY=VALUE`: Additional keyword arguments passed to the reader. Can be specified multiple times.

### Examples

#### Loading Model Data (CMAQ)

Retrieve surface data from CMAQ files and save as NetCDF:

```bash
monetio load cmaq -f "CMAQ_output_*.nc" -o combined_surface.nc -k surf_only=True
```

#### Retrieving Satellite Data (GOES)

Download and preprocess GOES-16 AOD data for a specific day:

```bash
monetio load goes -d 2023-06-01 -o goes_aod.nc --lazy
```

#### Loading Observation Data (ISH)

Retrieve Integrated Surface Database (ISH) lite data for multiple sites and save as CSV:

```bash
monetio load ish_lite -d 2023-01-01 -o observations.csv --as-pandas
```

#### Passing List Arguments

Some readers accept lists for certain parameters. You can pass multiple values for the same key:

```bash
monetio load cmaq -f "data.nc" -k var_list=O3 -k var_list=NO2 -k var_list=PM25
```

## Specialized Commands

For frequently used observation networks, MONETIO provides specialized commands with easier access to common parameters.

### AERONET

```bash
monetio aeronet -d 2023-01-01 --siteid GSFC --product AOD15 -o gsfc_aeronet.nc
```

### AirNow

```bash
monetio airnow -d 2023-01-01:2023-01-02 --daily -o airnow_daily.csv --as-pandas
```

### EPA AQS

```bash
monetio aqs -d 2023-01-01 -p OZONE -p PM2.5 -o epa_data.nc
```

### OpenAQ

```bash
monetio openaq -d 2023-01-01 -o openaq_global.csv --as-pandas
```


## Pre-processing Large Datasets (VirtualiZarr)

For very large gridded datasets (e.g., MERRA2, GFS, RRFS), `xarray.open_mfdataset` can be slow to build the initial coordinate map. MONETIO provides a command to pre-process these files into a single VirtualiZarr reference JSON file. This allows instant loading of the virtual Zarr dataset later.

```bash
monetio virtualizarr "s3://noaa-rrfs-pds/rrfs_a/20230101/00/control/rrfs.t00z.prslev.f*.nc" -o rrfs_cache.json
```

Options:
- `-o, --output PATH`: (Required) Path to save the output JSON reference file.
- `--concat-dim TEXT`: Dimension to concatenate the files along (default: `time`).

Once generated, you can load the data natively using the Python API by passing `use_virtualizarr=True` and `virtualizarr_file="rrfs_cache.json"` to the reader.

## Data Handling

### NetCDF vs. CSV

By default, the CLI attempts to return an `xarray.Dataset` and save it as a NetCDF file. This is appropriate for gridded data (Models, Satellites) and UGRID-compliant point observations.

If you prefer tabular data, use the `--as-pandas` flag. This will convert the result to a `pandas.DataFrame` and save it as a CSV. Note that metadata and coordinates are handled differently in CSV format.

### Laziness and Performance

For large datasets, always use the `--lazy` flag. This enables Dask-backed loading, which avoids reading the entire dataset into memory at once. The data will only be processed and written to disk when the final save operation occurs.
