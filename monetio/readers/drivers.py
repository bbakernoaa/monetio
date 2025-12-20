from typing import List, Union

import fsspec
import pandas as pd
import xarray as xr


class FileUtility:
    """
    Helper class to manage file path expansion (Local + S3 + HTTP).
    """

    @staticmethod
    def get_fs(path: str):
        """
        Returns the correct filesystem (local, s3, or http) based on the protocol.
        """
        if path.startswith("s3://"):
            import s3fs

            # anon=True means public bucket. Use anon=False to use your AWS credentials.
            return s3fs.S3FileSystem(anon=True)
        elif path.startswith("http://") or path.startswith("https://"):
            return fsspec.filesystem("http")
        return fsspec.filesystem("file")

    @staticmethod
    def expand_paths(path_input: Union[str, List[str]]) -> List[str]:
        """
        Converts a string (with wildcards), a single path, or a list of paths
        into a guaranteed list of file paths.
        """
        if isinstance(path_input, str):
            paths_to_process = [path_input]
        elif isinstance(path_input, list):
            paths_to_process = path_input
        else:
            raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")

        all_files = []
        for path in paths_to_process:
            fs = FileUtility.get_fs(path)
            if any(char in path for char in ["*", "?"]):
                # Expand glob pattern
                globbed_files = sorted(fs.glob(path))
                if path.startswith("s3://") and globbed_files and not globbed_files[0].startswith("s3://"):
                    globbed_files = [f"s3://{f}" for f in globbed_files]
                all_files.extend(globbed_files)
            else:
                # It's a single file path, verify existence
                if not path.startswith("http") and not fs.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
                all_files.append(path)

        if not all_files:
            raise FileNotFoundError(f"No files found matching pattern: {path_input}")

        return all_files


class XarrayDriver:
    """
    The unified driver for opening gridded data (NetCDF, GRIB, HDF).
    Supports S3 via fsspec.
    """

    def open(self, files: Union[str, List[str]], use_dask: bool = True, **kwargs) -> xr.Dataset:

        # Expand wildcards (supports S3 globbing now)
        file_list = FileUtility.expand_paths(files)

        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()
        if use_dask and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = {}

        # Extract preprocess if present
        preprocess = xr_kwargs.get("preprocess", None)

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                filename = file_list[0]

                # 'open_dataset' does not support 'preprocess', so we must remove it
                if "preprocess" in xr_kwargs:
                    del xr_kwargs["preprocess"]

                # Remove open_mfdataset specific arguments
                for k in [
                    "combine",
                    "concat_dim",
                    "parallel",
                    "compat",
                    "data_vars",
                    "coords",
                    "ids",
                    "infer_order",
                    "join",
                ]:
                    if k in xr_kwargs:
                        del xr_kwargs[k]

                # If S3 or HTTP, we open a file-like object to pass to xarray
                if filename.startswith("s3://") or filename.startswith("http"):
                    ds = xr.open_dataset(filename, engine="h5netcdf", **xr_kwargs)
                else:
                    ds = xr.open_dataset(filename, **xr_kwargs)

                # Apply preprocess manually
                if preprocess:
                    ds = preprocess(ds)

                return ds

            # Case B: Multiple Files (dataset)
            else:
                # xr.open_mfdataset handles URLs intelligently if 'parallel=True'
                # But generally, passing a list of S3 URLs works if backend supports it.
                if file_list[0].startswith("s3://"):
                    # For S3, open_mfdataset often prefers fsspec objects explicitly
                    # Create list of file objects (buffers)
                    # Note: This can be slow for 1000s of files;
                    # optimization: pass s3://.../*.nc directly to open_mfdataset if engine supports it
                    return xr.open_mfdataset(file_list, engine="h5netcdf", **xr_kwargs)
                else:
                    return xr.open_mfdataset(file_list, **xr_kwargs)

        except Exception:
            raise


class PandasDriver:
    """
    The unified driver for opening tabular/point data.
    """

    def open(
        self, files: Union[str, List[str]], read_method: str = "read_csv", **kwargs
    ) -> pd.DataFrame:

        file_list = FileUtility.expand_paths(files)

        # Get the actual pandas function
        if not hasattr(pd, read_method):
            raise ValueError(f"Pandas method '{read_method}' not found.")
        reader_func = getattr(pd, read_method)

        data_frames = []

        try:
            for f in file_list:
                if f.startswith("s3://"):
                    # Pandas can read S3 URLs directly if s3fs is installed!
                    # We just pass the URL string "s3://bucket/file.csv"
                    # optionally storage_options={'anon': True} can be passed in kwargs
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"anon": True}  # Default to public
                    df = reader_func(f, **kwargs)
                else:
                    df = reader_func(f, **kwargs)
                data_frames.append(df)

            if not data_frames:
                return pd.DataFrame()

            return pd.concat(data_frames, ignore_index=True)

        except Exception as e:
            raise OSError(f"PandasDriver failed to open files. Error: {e}")
