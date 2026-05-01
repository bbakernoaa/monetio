# Virtualization Support

MONETIO provides a uniform and optimized way to create and use virtual datasets using **Kerchunk** and **Icechunk**. This is particularly beneficial for large geospatial datasets (e.g., MERRA-2, GFS, RRFS, CMAQ) where opening hundreds of files via `xarray.open_mfdataset` can be slow.

## Benefits

- **Performance**: Bypass the overhead of scanning and mapping many files at runtime. Virtual datasets open almost instantly once the reference metadata is created.
- **Cloud-Optimized**: Access archival formats (NetCDF, HDF5, GRIB) as if they were cloud-native Zarr stores.
- **Unified Interface**: Same virtualization API across all gridded readers in MONETIO.

## Usage

### Python API

All gridded readers support virtualization parameters in `open_dataset`:

- `use_virtualizarr=True`: Enables Kerchunk-based virtualization.
- `virtualizarr_file="path/to/ref.json"`: Path to save/load the reference JSON.
- `use_icechunk=True`: Enables Icechunk-based virtualization.
- `icechunk_url="s3://bucket/store"`: URL for the Icechunk store.

#### Example: Virtualizing MERRA-2

```python
import monetio as mio

# This will scan the files, create a JSON reference, and return the virtual dataset
ds = mio.load("merra2", dates="2024-01-01", use_virtualizarr=True, virtualizarr_file="merra2_2024.json")

# Subsequent calls with the same file will load the reference instantly
ds = mio.load("merra2", use_virtualizarr=True, virtualizarr_file="merra2_2024.json")
```

#### Unified `virtualize` Function

For convenience, a top-level `virtualize` function is available to generate these references without loading the full dataset:

```python
import monetio as mio

mio.virtualize("cmaq", files="CMAQ_output_*.nc", output="cmaq_ref.json", backend="kerchunk")
```

### Command Line Interface

The `virtualize` command allows you to generate virtual datasets from the terminal:

```bash
# Create a Kerchunk reference for RRFS data on AWS
monetio virtualize -s rrfs -d 2023-01-01 -o rrfs_ref.json

# Create an Icechunk store for MERRA-2
monetio virtualize -s merra2 -d 2024-01-01 -o s3://my-bucket/merra2_ice --backend icechunk
```

Options:
- `-s, --source`: MONETIO source name.
- `-o, --output`: Output file path or Icechunk URL.
- `--backend`: `kerchunk` (default) or `icechunk`.
- `-d, --dates`: Date or date range for source URL building.
- `-k, --kwargs`: Additional keyword arguments for the reader.
