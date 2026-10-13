from collections.abc import Callable
from typing import Union

import fsspec
import pandas as pd
import xarray as xr

from .errors import (
    ContentInvalidError,
    MalformedInputError,
    OptionUnsupportedError,
    Tolerance,
    annotate,
    format_tolerance_record,
    infer_category,
)

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


# Exception types that mean "this engine cannot read this file, try another".
# ``h5netcdf`` signals a non-HDF5 file with a plain
# ``OSError`` ("file signature not found"); ``ImportError`` means the engine
# package is absent; ``ValueError`` covers xarray rejecting the format. Genuine
# read errors — ``PermissionError``/``FileNotFoundError`` (both ``OSError``
# subclasses) — are deliberately EXCLUDED so they propagate with their original
# type instead of being masked by a second engine attempt.
def _engine_not_applicable(exc: BaseException) -> bool:
    if isinstance(exc, ImportError | ValueError):
        return True
    return isinstance(exc, OSError) and not isinstance(exc, PermissionError | FileNotFoundError)


def _annotate_files(exc: BaseException, file_list) -> BaseException:
    """Same-typed copy of ``exc`` whose message names the file(s) involved.

    An error must identify the specific file and keep its original
    exception type (the old blanket ``raise OSError(...)`` flattened everything).
    Falls back to the untouched exception if the type cannot be reconstructed.
    """
    return annotate(exc, *file_list)


def _reject_unsupported_options(exc: BaseException, source: str | None) -> BaseException:
    """Turn a backend ``TypeError`` about an unexpected keyword into a named rejection.

    When a keyword reaches a storage backend that does not define it, the backend
    raises ``TypeError: ... got an unexpected keyword argument 'X'``. That names
    neither the source nor the option in the caller's terms; this re-raises it as
    an error naming both, so an option in a source's namespace is never silently
    accepted or reported as a backend internal.
    """
    import re

    if not isinstance(exc, TypeError):
        return exc
    match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
    if match is None:
        return exc
    return OptionUnsupportedError(match.group(1), source or "source")


def _validate_nonempty_result(result, file_list, source: str | None):
    """Refuse an empty result in place of a failure.

    A load that opened every requested input and produced a result carrying no
    information — a dataset with no variables, a dataset whose dimensions are
    all empty, or a frame with neither rows nor columns — is indistinguishable
    from a successful load unless it says so. If the inputs were genuinely empty
    that is still reported here rather than passed through as success.
    """
    if not file_list:
        return result
    if isinstance(result, xr.Dataset):
        if len(result.data_vars) == 0:
            raise ContentInvalidError(
                "load produced a dataset with no data variables",
                *file_list,
                cause=(
                    f"{len(file_list)} input(s) were opened but contributed no "
                    "variables; the result would be indistinguishable from a "
                    "successful load"
                ),
            )
        if result.dims and all(size == 0 for size in result.sizes.values()):
            # Variables exist but every dimension is empty — a schema-only
            # shell (a text reader whose parse silently matched nothing, for
            # instance) is just as indistinguishable from success as having no
            # variables at all.
            raise ContentInvalidError(
                "load produced a dataset with no data points",
                *file_list,
                cause=(
                    f"{len(file_list)} input(s) were opened but contributed no "
                    f"rows on any dimension ({dict(result.sizes)}); the result "
                    "would be indistinguishable from a successful load"
                ),
            )
    elif isinstance(result, pd.DataFrame):
        # A frame with columns but no rows is left to the reader: some sources
        # treat "the query was valid but returned no records" as a documented
        # outcome with its own message, and refusing it here would preempt that
        # report with a generic one. A frame with neither rows nor columns
        # carried no information at all and is refused. A lazy (dask) frame is
        # not validated — nothing has been read yet.
        if len(result) == 0 and len(result.columns) == 0:
            raise ContentInvalidError(
                "load produced an empty result",
                *file_list,
                cause=(
                    f"{len(file_list)} input(s) were opened but contributed no "
                    "rows or columns; the result would be indistinguishable from "
                    "a successful load"
                ),
            )
    return result


def _tolerant_per_file(file_list, open_one, tolerance):
    """Open inputs one at a time, skipping failures under input-scope tolerance.

    Returns ``(results, record)``, where ``record`` is the tolerance record for
    everything skipped. Without tolerance the first failure is raised, annotated
    to name its input. When every input fails, the first failure is raised even
    under tolerance: partial success is tolerated, empty-as-success never is.
    """
    from .errors import ToleranceRecord

    record = ToleranceRecord()
    results = []
    first_error = None
    for f in file_list:
        try:
            results.append(open_one(f))
        except Exception as exc:  # noqa: BLE001 - re-raised or recorded below; categories vary
            annotated = annotate(exc, f)
            if not tolerance:
                raise annotated from exc
            if first_error is None:
                first_error = annotated
            record = record.record_skip(f, str(exc), infer_category(exc))
    if not results and first_error is not None:
        raise first_error
    return results, record


def _file_signature(path: str):
    """Cheap metadata-only structural signature of one file, or ``None``.

    Opens the file lazily (no data is read) and hashes its variable names,
    dimension sizes, per-variable attributes, and coordinate dtypes — the
    three things a combine can disagree on (variables, coordinates,
    attributes). ``decode_times=False`` keeps the signature independent of
    time-decoding differences so only genuine structural mismatches stand out.
    """
    for engine in ("h5netcdf", None):
        try:
            ds = xr.open_dataset(path, decode_times=False, **({"engine": engine} if engine else {}))
        except Exception as exc:  # noqa: BLE001
            if engine is None or not _engine_not_applicable(exc):
                return None  # unopenable, or a real error: this file stands out
            continue
        with ds:
            coord_names = set(ds.coords)

            # ``units``/``calendar`` on coordinate variables legitimately
            # differ per file in a multi-file set (the time axis advances), so
            # they are excluded; every other attribute difference is structural.
            def _attrs(name, var):
                return tuple(
                    sorted(
                        (k, str(v))
                        for k, v in var.attrs.items()
                        if not (k in ("units", "calendar") and name in coord_names)
                    )
                )

            return (
                tuple(sorted(ds.data_vars)),
                tuple(sorted((str(k), int(v)) for k, v in ds.sizes.items())),
                tuple(sorted((name, _attrs(name, var)) for name, var in ds.variables.items())),
                tuple(sorted((str(c), str(ds[c].dtype)) for c in ds.coords)),
            )
    return None


def _mismatch_diagnostic(file_list) -> str | None:
    """Name the file(s) whose structure deviates from the set's majority.

    When a multi-file combine fails (mismatched variables, coordinates, or
    attributes), xarray's own error identifies the conflicting dimension or
    variable but not the *file*. This re-opens each file's metadata only — no
    data payload — and reports the minority-signature paths. Returns ``None``
    when no clear majority exists, so the caller keeps xarray's original error
    instead of a guess.
    """
    from collections import Counter

    files = list(file_list)
    if len(files) < 2:
        return None
    try:
        sigs = {f: _file_signature(f) for f in files}
    except Exception:  # noqa: BLE001 - diagnosis must never mask the real error
        return None
    counts = Counter(s for s in sigs.values() if s is not None)
    if not counts:
        return None
    majority, n = counts.most_common(1)[0]
    if n * 2 <= len(files):  # no majority signature: do not guess a culprit
        return None
    bad = [f for f, s in sigs.items() if s != majority]
    if not bad:
        return None
    preview = ", ".join(str(f) for f in bad[:5])
    if len(bad) > 5:
        preview += ", ..."
    return preview


def _open_mfdataset(file_list, xr_kwargs: dict, engine: str | None = None):
    """``xr.open_mfdataset`` with a file-level diagnostic on combine failures.

    A combine error (``ValueError``, including xarray's ``AlignmentError``)
    gets the mismatched file(s) appended to its message, keeping the original
    exception type. Errors that are not combine-shaped, or sets with no clear
    majority signature, re-raise untouched. A ``_monetio_diagnosed`` marker
    prevents the engine-fallback retry from re-running the diagnosis.
    """
    try:
        if engine is not None:
            return xr.open_mfdataset(file_list, engine=engine, **xr_kwargs)
        return xr.open_mfdataset(file_list, **xr_kwargs)
    except ValueError as exc:
        if getattr(exc, "_monetio_diagnosed", False):
            raise
        diag = _mismatch_diagnostic(file_list)
        if diag is None:
            raise
        annotated = type(exc)(f"{exc} [mismatched file(s): {diag}]")
        annotated._monetio_diagnosed = True
        raise annotated from exc


def _path_protocol(path: str) -> str | None:
    """Return the ``scheme`` of a ``scheme://`` path, or ``None`` for local."""
    if "://" in path:
        return path.split("://", 1)[0]
    return None


def _store_bucket(path: str) -> str:
    """Best-effort store identifier from a path (bucket for s3, root for others)."""
    protocol = _path_protocol(path)
    if not protocol:
        return path
    remainder = path.split("://", 1)[1]
    if protocol == "s3":
        bucket = remainder.split("/")[0]
        return f"s3://{bucket}"
    return f"{protocol}://{remainder.split('/')[0]}"


# botocore error codes / classes that mean "the store rejected our credentials".
_CREDENTIAL_ERROR_CODES = {
    "AccessDenied",
    "InvalidAccessKeyId",
    "ExpiredToken",
    "SignatureDoesNotMatch",
}


def _is_credential_error(exc: BaseException) -> bool:
    """True when ``exc`` is a missing/expired/rejected-credential failure.

    Detected by botocore class name (``NoCredentialsError``) or a ``ClientError``
    whose response error code is in the credential set. A bare network error or
    a genuine "no such bucket" is *not* a credential error and is left untouched
    so it is never mislabeled.
    """
    name = type(exc).__name__
    if name in {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError"}:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
        if code in _CREDENTIAL_ERROR_CODES:
            return True
    return False


def _store_error(exc: BaseException, path: str) -> BaseException:
    """Return ``exc`` with the object store named, when it is a credential error.

    A missing or expired credential must identify the store and the credential
    problem rather than surfacing as a generic file-not-found. The original
    exception type is preserved (botocore errors are annotated in place via
    ``.args``, which their ``str`` reflects); non-credential errors pass
    through unchanged.
    """
    if not _is_credential_error(exc):
        return exc
    store = _store_bucket(str(path))
    message = str(exc)
    if store not in message:
        message = f"{message} [store: {store}; credential problem]"
    try:
        exc.args = (message,)
    except (AttributeError, TypeError):  # pragma: no cover - exotic exceptions
        return type(exc)(message) if _reconstructable(exc) else exc
    return exc


def _reconstructable(exc: BaseException) -> bool:
    try:
        type(exc)(str(exc))
    except Exception:  # noqa: BLE001
        return False
    return True


def _build_s3_config(storage_options: dict) -> dict:
    """Build an S3Store config dict from fsspec-style storage_options."""
    s3_config = {}
    if storage_options.get("anon", True):
        s3_config["skip_signature"] = "true"
    if "client_kwargs" in storage_options and "region_name" in storage_options["client_kwargs"]:
        s3_config["region"] = storage_options["client_kwargs"]["region_name"]
    return s3_config


def _select_store(file_list: list[str], storage_options: dict) -> tuple:
    """Select the appropriate object store based on file protocol.

    Parameters
    ----------
    file_list : list[str]
        List of file paths (all assumed to share the same protocol).
    storage_options : dict
        fsspec-style storage options (e.g. ``{"anon": True}``).

    Returns
    -------
    tuple[ObjectStoreRegistry, list[str]]
        The configured registry and (possibly updated) file list.
    """
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import HTTPStore, LocalStore, S3Store

    registry = ObjectStoreRegistry()

    if file_list[0].startswith("s3://"):
        bucket = file_list[0].replace("s3://", "").split("/")[0]
        config = _build_s3_config(storage_options)
        store = S3Store(bucket, config=config)
        registry.register(f"s3://{bucket}", store)
    elif file_list[0].startswith("http://") or file_list[0].startswith("https://"):
        store = HTTPStore()
        registry.register("http://", store)
        registry.register("https://", store)
    else:
        store = LocalStore(prefix="/")
        registry.register("file:///", store)
        file_list = [f"file://{f}" if not f.startswith("file://") else f for f in file_list]

    return registry, file_list


def _open_via_icechunk(vds, icechunk_repo: str, virtualizarr_file: str | None) -> xr.Dataset:
    """Store virtual references in Icechunk and return the dataset.

    Parameters
    ----------
    vds : virtualizarr.VirtualDataset
        The virtual dataset to persist.
    icechunk_repo : str
        Path (local or remote) to the Icechunk repository.
    virtualizarr_file : str | None
        Unused for Icechunk but accepted for interface consistency.

    Returns
    -------
    xr.Dataset
        The dataset opened from the Icechunk store.
    """
    try:
        import icechunk
    except ImportError:
        raise ImportError(
            "Icechunk backend requires 'icechunk'. Install with: pip install monetio[icechunk]"
        )

    repo = icechunk.Repository.open_or_create(icechunk_repo)
    session = repo.writable_session("main")
    store = session.store

    vds.virtualize.to_icechunk(store)
    session.commit("VirtualiZarr references")

    # Re-open for reading
    session = repo.readonly_session()
    return xr.open_zarr(session.store, consolidated=False)


class FileUtility:
    """
    Helper class to manage file path expansion (Local + S3 + HTTP).
    """

    @staticmethod
    def get_fs(path: str, storage_options: dict | None = None):
        """
        Returns the correct filesystem (local, s3, http, or other fsspec
        protocol) based on the path's protocol.

        ``storage_options`` are forwarded to the fsspec backend. For ``s3://``
        paths, ``anon`` defaults to ``True`` (public bucket) exactly as before;
        passing ``anon=False`` uses AWS credentials, so a missing/expired
        credential surfaces as its own error rather than being masked as an
        empty listing. Any other ``protocol://`` prefix (``memory://``,
        ``gcs://``, ``abfs://``, …) resolves to that fsspec backend; plain
        paths use the local filesystem.
        """
        options = dict(storage_options or {})
        if path.startswith("s3://"):
            # anon=True means public bucket. Use anon=False to use your AWS credentials.
            anon = options.pop("anon", True)
            return fsspec.filesystem("s3", anon=anon, **options)
        elif path.startswith("http://") or path.startswith("https://"):
            return fsspec.filesystem("http", **options)
        elif path.startswith("ftp://"):
            return fsspec.filesystem("ftp", **options)
        elif "://" in path:
            protocol = path.split("://", 1)[0]
            return fsspec.filesystem(protocol, **options)
        return fsspec.filesystem("file")

    @staticmethod
    def expand_paths(
        path_input: str | list[str], fs=None, storage_options: dict | None = None
    ) -> list[str]:
        """
        Converts a string (with wildcards), a single path, or a list of paths
        into a guaranteed list of file paths/objects.
        """
        # Convert Path objects to string
        if hasattr(path_input, "__fspath__"):
            path_input = str(path_input)

        # Case 1: It's a list already
        if isinstance(path_input, list):
            return sorted([str(p) if hasattr(p, "__fspath__") else p for p in path_input])

        # Case 2: It's a single string (S3 or Local)
        if isinstance(path_input, str):
            # If no specific filesystem provided, guess it from the path
            if fs is None:
                fs = FileUtility.get_fs(path_input, storage_options)

            # Use fsspec/s3fs to glob wildcards (works for s3://bucket/data/*.nc too!)
            if any(char in path_input for char in ["*", "?"]):
                # HTTP globbing is generally not supported by fsspec without specific implementation
                # For S3/Local it works.
                if path_input.startswith("http"):
                    # Fallback: treat as single file if glob chars present but http (unlikely to work)
                    pass

                try:
                    files = sorted(fs.glob(path_input))
                except Exception as exc:  # noqa: BLE001
                    raise _store_error(exc, path_input) from exc
                # fs.glob usually returns paths without the protocol
                # (e.g. 'bucket/file.nc'); restore it so downstream open()
                # calls resolve the same backend the glob used.
                protocol = _path_protocol(path_input)
                if protocol and files and not str(files[0]).startswith(f"{protocol}://"):
                    files = [f"{protocol}://{f}" for f in files]

                if not files:
                    raise FileNotFoundError(f"No files found matching pattern: {path_input}")
                return files
            else:
                # It is a specific single file
                if not path_input.startswith("http") and not fs.exists(path_input):
                    raise FileNotFoundError(f"File not found: {path_input}")
                return [path_input]

        raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")

    @staticmethod
    def expand_paths_iter(
        path_input, fs=None, chunk_size: int = 10_000, storage_options: dict | None = None
    ):
        """Streaming, order-preserving variant of :meth:`expand_paths`.

        Yields file paths in the **same globally sorted order** as
        ``expand_paths`` — so ``concat``/``combine_nested`` results are
        identical — but never materializes a single full sorted collection:
        paths are consumed in bounded ``chunk_size`` runs, each sorted, then
        merged lazily with :func:`heapq.merge`. This lets a lazy consumer such
        as the ``dask.delayed`` fan-out (``concurrent_files=True``) build its
        task graph over ≥100,000 paths without an O(n) ``sorted`` list.

        The default :meth:`expand_paths` (a plain list) is unchanged.

        Raises the same ``FileNotFoundError`` as ``expand_paths`` when a glob
        matches nothing or a named single file does not exist (spec edge case).
        """
        import heapq
        from itertools import islice

        if hasattr(path_input, "__fspath__"):
            path_input = str(path_input)

        glob_mode = False
        if isinstance(path_input, list):
            raw = iter(str(p) if hasattr(p, "__fspath__") else p for p in path_input)
        elif isinstance(path_input, str):
            if fs is None:
                fs = FileUtility.get_fs(path_input, storage_options)
            if any(char in path_input for char in ["*", "?"]):
                glob_mode = True
                if path_input.startswith("http"):
                    raise FileNotFoundError(f"No files found matching pattern: {path_input}")
                protocol = _path_protocol(path_input)

                def _raw_glob(fs=fs, pattern=path_input, protocol=protocol):
                    # fsspec's glob materializes its own list; we stream it into
                    # bounded sorted runs below rather than sorting the whole set.
                    try:
                        globbed = fs.glob(pattern)
                    except Exception as exc:  # noqa: BLE001
                        raise _store_error(exc, pattern) from exc
                    for f in globbed:
                        f = str(f)
                        if protocol and not f.startswith(f"{protocol}://"):
                            f = f"{protocol}://{f}"
                        yield f

                raw = _raw_glob()
            else:
                if not path_input.startswith("http") and not fs.exists(path_input):
                    raise FileNotFoundError(f"File not found: {path_input}")
                yield path_input
                return
        else:
            raise TypeError(f"Invalid path type: {type(path_input)}. Must be str or list.")

        # Bounded-memory sorted merge: sort each chunk_size run (working set
        # O(chunk_size)) and lazily k-way-merge the runs. The merged stream is
        # globally sorted — identical to expand_paths' output, so concat order
        # is unchanged — while no single full sorted list is ever built and a
        # lazy consumer can overlap with later runs.
        runs = []
        while True:
            chunk = sorted(islice(raw, chunk_size))
            if not chunk:
                break
            runs.append(chunk)

        if not runs:
            if glob_mode:
                raise FileNotFoundError(f"No files found matching pattern: {path_input}")
            return  # empty list input -> empty stream (matches expand_paths)

        yield from heapq.merge(*runs)


class XarrayDriver:
    """
    The unified driver for opening gridded data (NetCDF, GRIB, HDF).
    Supports S3 via fsspec.
    """

    def __init__(self, source: str | None = None):
        #: Manifest source name, when the driver belongs to a registered
        #: reader. Errors name it (an unsupported option must identify the
        #: source, not the backend engine) and scale-capability declarations
        #: are looked up through it. A bare ``XarrayDriver()`` used directly
        #: (e.g. in tests) has no name and skips the capability check.
        self.source = source

    def _require_defer_support(self) -> None:
        """Refuse ``defer=True`` for a source that has not earned it.

        Enforced here rather than only in the reader base class because
        readers forward their keyword arguments down to the driver: checking
        at the driver means ``defer`` can never reach a storage backend as an
        unknown keyword (which surfaced there as a ``TypeError`` naming
        neither the source nor the option).
        """
        from .._manifest import resolve
        from .errors import CapabilityUnavailableError

        if self.source is None:
            return
        try:
            spec = resolve(self.source)
        except KeyError:
            return
        if not spec.capabilities.deferred:
            reason = spec.capabilities.reason or "no reason recorded"
            raise CapabilityUnavailableError(spec.name, "defer", reason)

    def open(
        self,
        files: str | list[str],
        use_dask: bool = False,
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        virtualizarr_backend: str = "kerchunk",
        icechunk_repo: str | None = None,
        parallel: bool = False,
        concurrent_files: bool = False,
        defer: bool = False,
        tolerate=None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Open gridded data backend-agnostically.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s), URL(s), or glob pattern.
        use_dask : bool, optional
            Whether to use Dask for lazy loading, by default False.
        use_virtualizarr : bool, optional
            Whether to use VirtualiZarr to create a virtual Zarr dataset, by default False.
            Useful for large datasets to avoid xarray.open_mfdataset overhead.
        virtualizarr_file : str, optional
            Path to save/load the VirtualiZarr reference JSON file. If provided and the file
            exists, the references will be loaded from it. If the file does not exist,
            the references will be computed and saved to this path.
        virtualizarr_backend : str, optional
            Backend for VirtualiZarr references. Must be ``"kerchunk"`` (default) or
            ``"icechunk"``. When ``"icechunk"`` is selected, references are stored in
            an Icechunk repository instead of a kerchunk JSON file.
        icechunk_repo : str, optional
            Path to the Icechunk repository. Required when
            ``virtualizarr_backend="icechunk"``.
        parallel : bool, optional
            Passed through to :func:`xarray.open_mfdataset` to parallelize per-file
            metadata parsing. Default ``False`` reproduces today's
            single-threaded behavior; the result is identical either way.
        concurrent_files : bool, optional
            For the multi-file custom ``read_method`` path, fan the per-file reads
            out with :func:`dask.delayed` instead of a serial comprehension,
            preserving file order. Default ``False`` is the serial path.
        defer : bool, optional
            Return an unevaluated, dask-backed dataset. Implies
            ``use_dask=True``; raises for a source declaring
            ``capabilities.deferred = False``.
        tolerate : str or bool, optional
            Explicit tolerance for skipped inputs (``"input"``): a failing file
            of several is skipped and recorded in provenance instead of failing
            the load. Never the default; with no tolerance the first failure
            raises. Requires opening files one at a time, so the multi-file
            standard path combines per-file datasets rather than delegating to
            :func:`xarray.open_mfdataset`.
        **kwargs : dict
            Additional arguments passed to xarray open functions.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        from .sat_utils import update_history

        tolerance = Tolerance.parse(tolerate)
        if defer:
            self._require_defer_support()
            # defer wins over use_dask=False.
            use_dask = True
        # Validate virtualizarr_backend parameter
        if virtualizarr_backend not in ("kerchunk", "icechunk"):
            raise ValueError(
                f"Invalid virtualizarr_backend '{virtualizarr_backend}'. "
                "Must be 'kerchunk' or 'icechunk'."
            )
        # Prepare kwargs for xarray
        xr_kwargs = kwargs.copy()

        # Parallel metadata parsing. Only inject when requested so
        # the default path is byte-identical to before (xarray's own default is
        # False). Reaches xr.open_mfdataset; stripped for single-file/virtual paths.
        if parallel:
            xr_kwargs["parallel"] = True

        # Expand wildcards (supports S3 globbing now). storage_options (e.g.
        # {"anon": False}) reach the fsspec backend so credential problems
        # surface as themselves instead of an empty listing.
        file_list = FileUtility.expand_paths(
            files, storage_options=dict(xr_kwargs.get("storage_options", {})) or None
        )

        # Handle 'lazy' keyword: Eager by default.
        if "lazy" in xr_kwargs:
            use_dask = xr_kwargs.pop("lazy")

        # If laziness or specific chunking is requested, ensure auto-chunking is used.
        if (use_dask or "chunks" in xr_kwargs) and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = {}

        # Extract MONETIO-specific keywords
        preprocess = xr_kwargs.pop("preprocess", None)
        read_method = xr_kwargs.pop("read_method", None)

        if use_virtualizarr:
            try:
                import ujson  # noqa: F401
                import zarr  # noqa: F401
                from virtualizarr import open_virtual_mfdataset
                from virtualizarr.parsers import HDFParser
            except ImportError:
                raise ImportError(
                    "VirtualiZarr support requires additional packages. "
                    "Install with: pip install monetio[virtualizarr]"
                )

            import os

            # --- Kerchunk cache: load existing refs if available ---
            refs = None
            if (
                virtualizarr_backend == "kerchunk"
                and virtualizarr_file is not None
                and os.path.exists(virtualizarr_file)
            ):
                try:
                    with open(virtualizarr_file) as f_ref:
                        refs = ujson.load(f_ref)
                except Exception as e:  # noqa: BLE001 - cache read is best-effort (malformed, content_invalid); recompute below
                    import warnings

                    warnings.warn(f"Failed to load virtualizarr_file {virtualizarr_file}: {e}")
                    refs = None

            if refs is None:
                storage_options = dict(xr_kwargs.get("storage_options", {}))
                registry, file_list = _select_store(file_list, storage_options)

                concat_dim = xr_kwargs.get("concat_dim", "time")
                try:
                    vds = open_virtual_mfdataset(
                        file_list,
                        registry=registry,
                        parser=HDFParser(),
                        combine="nested",
                        concat_dim=concat_dim,
                    )
                except ValueError:
                    vds = open_virtual_mfdataset(
                        file_list, registry=registry, parser=HDFParser(), combine="by_coords"
                    )

                # --- Branch on backend ---
                if virtualizarr_backend == "icechunk":
                    ds = _open_via_icechunk(vds, icechunk_repo, virtualizarr_file)
                    if preprocess:
                        ds = preprocess(ds)
                    return ds

                # Kerchunk path: export refs and optionally cache them
                refs = vds.vz.to_kerchunk()

                if virtualizarr_file is not None:
                    try:
                        with open(virtualizarr_file, "w") as f_ref:
                            ujson.dump(refs, f_ref)
                    except Exception as e:
                        # The caller asked for the references at this path; a
                        # failed write must not pass as success (that is the
                        # silent-discard failure mode). categories: malformed,
                        # permission_denied, not_found.
                        raise MalformedInputError(
                            virtualizarr_file,
                            f"could not write virtualizarr references: {e}",
                        ) from e

            remote_protocol = "file"
            remote_options = {}
            if file_list[0].startswith("s3://"):
                remote_protocol = "s3"
                remote_options = dict(xr_kwargs.get("storage_options", {}))
                if "anon" not in remote_options:
                    remote_options["anon"] = True
            elif file_list[0].startswith("http"):
                remote_protocol = "http"
                # file_list for fsspec mapper should not start with file:// if they are local
            elif file_list[0].startswith("file://"):
                pass

            mapper = fsspec.get_mapper(
                "reference://",
                fo=refs,
                remote_protocol=remote_protocol,
                remote_options=remote_options,
            )

            # Clean up xr_kwargs for open_dataset
            mfdataset_keys = [
                "combine",
                "concat_dim",
                "parallel",
                "compat",
                "data_vars",
                "coords",
                "ids",
                "infer_order",
                "join",
                "engine",
                "storage_options",
            ]
            for k in mfdataset_keys:
                xr_kwargs.pop(k, None)

            ds = xr.open_dataset(
                mapper,
                engine="zarr",
                backend_kwargs={"consolidated": False},
                consolidated=False,
                **xr_kwargs,
            )

            if preprocess:
                ds = preprocess(ds)

            return ds

        try:
            # Case A: Single File (Optimized)
            if len(file_list) == 1:
                ds = self._open_single_file(file_list[0], xr_kwargs, read_method, preprocess)
                return _validate_nonempty_result(ds, file_list, self.source)

            # Case B: Multiple Files (dataset)
            else:
                if read_method:
                    # Custom read_method path (e.g. TOLNet)
                    if concurrent_files:
                        # Fan out per-file reads with dask.delayed, mirroring
                        # PandasDriver's lazy path. dask.compute
                        # returns results in submission order, so file order —
                        # which combine_nested/concat depend on — is preserved.
                        import dask

                        delayed_ds = [dask.delayed(read_method)(f, **xr_kwargs) for f in file_list]
                        dsets = list(dask.compute(*delayed_ds))
                    else:
                        dsets = [read_method(f, **xr_kwargs) for f in file_list]

                    if preprocess:
                        dsets = [preprocess(ds) for ds in dsets]

                    # Combine logic (backend-agnostic)
                    concat_dim = xr_kwargs.get("concat_dim")
                    if concat_dim is not None:
                        # If a explicit dimension is given, we use nested combination
                        # or direct concatenation if nested combine fails.
                        try:
                            ds = xr.combine_nested(
                                dsets,
                                concat_dim=concat_dim,
                                data_vars="minimal",
                                coords="minimal",
                                compat="override",
                            )
                        except (ValueError, TypeError):
                            ds = xr.concat(
                                dsets, dim=concat_dim, coords="different", data_vars="minimal"
                            )
                    else:
                        try:
                            ds = xr.combine_by_coords(
                                dsets,
                                data_vars="minimal",
                                coords="minimal",
                                compat="override",
                            )
                        except (ValueError, TypeError):
                            # Fallback to concat if combine_by_coords fails.
                            ds = xr.concat(
                                dsets, dim="time", coords="different", data_vars="minimal"
                            )
                    return _validate_nonempty_result(ds, file_list, self.source)

                # Tolerated multi-file loads open one input at a time so a single
                # bad file can be skipped and recorded; xarray's own multi-file
                # open cannot report which file failed inside a combine error.
                if tolerance and tolerance.scope == "input":

                    def _open_one(f):
                        return self._open_single_file(f, xr_kwargs, None, preprocess)

                    dsets, record = _tolerant_per_file(file_list, _open_one, tolerance)
                    ds = self._combine_many(dsets, xr_kwargs)
                    line = format_tolerance_record(record, requested=True)
                    if line:
                        ds = update_history(ds, line)
                    return _validate_nonempty_result(ds, file_list, self.source)

                # Standard path: use xr.open_mfdataset
                if preprocess:
                    xr_kwargs["preprocess"] = preprocess

                # If concat_dim is provided, ensure we use nested combine to avoid xarray errors
                if "concat_dim" in xr_kwargs and "combine" not in xr_kwargs:
                    xr_kwargs["combine"] = "nested"

                if "engine" not in xr_kwargs:
                    try:
                        ds = _open_mfdataset(file_list, xr_kwargs, engine="h5netcdf")
                    except Exception as first:  # noqa: BLE001
                        if not _engine_not_applicable(first):
                            raise
                        ds = _open_mfdataset(file_list, xr_kwargs)
                else:
                    ds = _open_mfdataset(file_list, xr_kwargs)
                return _validate_nonempty_result(ds, file_list, self.source)

        except Exception as e:
            # Preserve the original exception type and name the files,
            # instead of flattening everything into a generic OSError.
            raise _reject_unsupported_options(_annotate_files(e, file_list), self.source) from e

    def _open_single_file(self, filename, xr_kwargs, read_method, preprocess) -> xr.Dataset:
        """Open (and preprocess) exactly one input, without combining.

        The per-file half of the multi-file path: used for a single-file load,
        for tolerated loads (where a failure must be attributable to one file),
        and by readers that combine datasets themselves.
        """
        # Remove open_mfdataset specific arguments to prevent TypeError in xr.open_dataset
        xr_kwargs = {
            k: v
            for k, v in xr_kwargs.items()
            if k
            not in (
                "combine",
                "concat_dim",
                "parallel",
                "compat",
                "data_vars",
                "coords",
                "ids",
                "infer_order",
                "join",
            )
        }

        if read_method:
            ds = read_method(filename, **xr_kwargs)
        else:
            # Logic for standard engine/remote access
            if _path_protocol(filename) and not filename.startswith("file://"):
                fs = FileUtility.get_fs(filename, xr_kwargs.get("storage_options"))
                file_obj = fs.open(filename)
            else:
                file_obj = filename

            if "engine" not in xr_kwargs:
                try:
                    ds = xr.open_dataset(file_obj, engine="h5netcdf", **xr_kwargs)
                except Exception as first:  # noqa: BLE001
                    # Only fall back when the preferred engine genuinely
                    # cannot read this format; a real read error must not
                    # be masked by a second, misleading attempt.
                    if not _engine_not_applicable(first):
                        raise
                    ds = xr.open_dataset(file_obj, **xr_kwargs)
            else:
                ds = xr.open_dataset(file_obj, **xr_kwargs)

        # Apply preprocess manually
        if preprocess:
            ds = preprocess(ds)

        return ds

    @staticmethod
    def _combine_many(dsets, xr_kwargs) -> xr.Dataset:
        """Combine per-file datasets with the same logic the multi-file paths use."""
        concat_dim = xr_kwargs.get("concat_dim")
        if concat_dim is not None:
            try:
                return xr.combine_nested(
                    dsets,
                    concat_dim=concat_dim,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
            except (ValueError, TypeError):
                return xr.concat(dsets, dim=concat_dim, coords="different", data_vars="minimal")
        try:
            return xr.combine_by_coords(
                dsets, data_vars="minimal", coords="minimal", compat="override"
            )
        except (ValueError, TypeError):
            return xr.concat(dsets, dim="time", coords="different", data_vars="minimal")


class PandasDriver:
    """
    The unified driver for opening tabular/point data.
    """

    def __init__(self, source: str | None = None):
        #: Manifest source name; see :attr:`XarrayDriver.source`.
        self.source = source

    def open(
        self,
        files: str | list[str],
        read_method: str | Callable = "read_csv",
        lazy: bool = False,
        meta: pd.DataFrame | pd.Series | dict | tuple | None = None,
        tolerate=None,
        **kwargs,
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """Open tabular inputs.

        ``tolerate`` (``"input"``) applies to the eager path only: the lazy path
        defers reading to Dask, where a per-file failure surfaces at compute
        time and cannot be attributed to one file here. Tolerating a lazy load
        would record skips that never happened, so the request is rejected
        rather than quietly ignored.
        """
        from .sat_utils import update_history

        tolerance = Tolerance.parse(tolerate)
        file_list = FileUtility.expand_paths(
            files, storage_options=dict(kwargs.get("storage_options", {})) or None
        )

        # Get the actual reading function
        if callable(read_method):
            reader_func = read_method
        elif hasattr(pd, read_method):
            reader_func = getattr(pd, read_method)
        else:
            raise ValueError(f"Pandas method '{read_method}' not found and not callable.")

        if lazy:
            import dask
            import dask.dataframe as dd

            if tolerance:
                raise OptionUnsupportedError(
                    "tolerate",
                    self.source or "source",
                    cause=(
                        "a lazy load defers reading to Dask, so a failing input "
                        "cannot be identified and recorded here; use the eager "
                        "path (lazy=False) to tolerate bad inputs"
                    ),
                )

            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)

            delayed_dfs = []
            for f in file_list:
                if f.startswith("s3://"):
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"anon": True}

                d = dask.delayed(reader_func)(f, **kwargs)
                if preprocess:
                    d = dask.delayed(preprocess)(d)
                delayed_dfs.append(d)

            if not delayed_dfs:
                return dd.from_pandas(pd.DataFrame(), npartitions=1)

            return dd.from_delayed(delayed_dfs, meta=meta)

        data_frames = []
        # Reuse our filesystem logic
        try:
            # Extract preprocess if present
            preprocess = kwargs.pop("preprocess", None)

            def _read_one(f):
                if f.startswith("s3://"):
                    # Pandas can read S3 URLs directly if s3fs is installed!
                    per_file = kwargs
                    if "storage_options" not in per_file:
                        per_file = {**per_file, "storage_options": {"anon": True}}
                else:
                    per_file = kwargs
                df = reader_func(f, **per_file)
                if preprocess:
                    df = preprocess(df)
                return df

            if tolerance and tolerance.scope == "input":
                # One input at a time so a failure is attributable to one file
                # and can be skipped and recorded rather than failing the load.
                data_frames, record = _tolerant_per_file(file_list, _read_one, tolerance)
                if not data_frames:
                    return pd.DataFrame()
                df = pd.concat(data_frames, ignore_index=True)
                line = format_tolerance_record(record, requested=True)
                if line:
                    df = update_history(df, line)
                return _validate_nonempty_result(df, file_list, self.source)

            for f in file_list:
                data_frames.append(_read_one(f))

            if not data_frames:
                return pd.DataFrame()

            df = pd.concat(data_frames, ignore_index=True)
            return _validate_nonempty_result(df, file_list, self.source)

        except (RuntimeError, ValueError, MalformedInputError):
            raise
        except Exception as e:
            # Preserve the original exception type and name the files,
            # instead of flattening everything into a generic OSError.
            raise _reject_unsupported_options(annotate(e, *file_list), self.source) from e
