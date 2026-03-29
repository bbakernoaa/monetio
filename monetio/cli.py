"""MONETIO Command Line Interface."""

import click
import pandas as pd
import xarray as xr

import monetio


def parse_dates(dates):
    """Parse dates from CLI input."""
    if not dates:
        return None
    # Support 'start:end' range
    if len(dates) == 1 and ":" in dates[0]:
        try:
            start, end = dates[0].split(":")
            return pd.date_range(start, end)
        except Exception as e:
            raise click.BadParameter(f"Invalid date range format: {dates[0]}. Error: {e}")
    # Otherwise return list
    return list(dates)


def handle_save(obj, output, as_pandas):
    """Handle saving the loaded data object."""
    if output:
        if as_pandas or isinstance(obj, (pd.DataFrame, pd.Series)):
            # Handle dask dataframe
            if hasattr(obj, "compute"):
                obj = obj.compute()
            # Handle xarray dataset to pandas
            if isinstance(obj, xr.Dataset):
                obj = obj.to_dataframe().reset_index()

            obj.to_csv(output, index=False)
            click.echo(f"Saved to {output} (CSV)")
        else:
            if isinstance(obj, pd.DataFrame):
                click.echo("Error: Cannot save DataFrame as NetCDF. Use --as-pandas for CSV.")
                return
            obj.to_netcdf(output)
            click.echo(f"Saved to {output} (NetCDF)")
    else:
        # Just print a summary if no output file specified
        click.echo(obj)


@click.group()
def cli():
    """MONETIO Command Line Interface for atmospheric data."""
    pass


def common_options(f):
    """Decorator for common CLI options."""
    f = click.option(
        "--dates",
        "-d",
        multiple=True,
        help="Date or date range (e.g., '2023-01-01' or '2023-01-01:2023-01-05').",
    )(f)
    f = click.option("--output", "-o", help="Output file path.")(f)
    f = click.option("--n-procs", default=1, help="Number of processors for parallel loading.")(f)
    f = click.option("--lazy", is_flag=True, help="Use Dask for lazy loading.")(f)
    f = click.option("--as-pandas", is_flag=True, help="Process and save as CSV (Pandas).")(f)
    return f


@cli.command()
@click.argument("source")
@common_options
@click.option("--files", "-f", multiple=True, help="File path(s) or glob pattern(s).")
@click.option(
    "--kwargs",
    "-k",
    multiple=True,
    help="Additional keyword arguments in key=value format.",
)
def load(source, dates, output, n_procs, lazy, as_pandas, files, kwargs):
    """
    Generic command to load data from any registered source.

    Available sources include: aeronet, airnow, aqs, goes, ish, ish_lite, etc.
    See monetio.load documentation for full list.
    """
    d = parse_dates(dates)
    f = list(files) if files else None

    # Parse additional kwargs
    extra_kwargs = {}
    for kw in kwargs:
        if "=" in kw:
            key, val = kw.split("=", 1)
            # Try to convert to int or float or bool if possible
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass

            if key in extra_kwargs:
                if not isinstance(extra_kwargs[key], list):
                    extra_kwargs[key] = [extra_kwargs[key]]
                extra_kwargs[key].append(val)
            else:
                extra_kwargs[key] = val

    click.echo(f"Loading {source} data...")

    # Build arguments dynamically to avoid passing defaults that might conflict
    load_kwargs = {"as_xarray": not as_pandas}
    if d is not None:
        load_kwargs["dates"] = d
    if f is not None:
        load_kwargs["files"] = f
    if n_procs != 1:
        load_kwargs["n_procs"] = n_procs
    if lazy:
        load_kwargs["lazy"] = lazy

    obj = monetio.load(source, **load_kwargs, **extra_kwargs)
    handle_save(obj, output, as_pandas)


@cli.command()
@common_options
@click.option("--siteid", help="AERONET site ID.")
@click.option("--product", default="AOD15", help="AERONET product (e.g., 'AOD15', 'SDA20').")
@click.option("--daily", is_flag=True, help="Load daily averages.")
@click.option("--inv-type", help="Inversion type (e.g., 'ALM15').")
def aeronet(dates, output, n_procs, lazy, as_pandas, siteid, product, daily, inv_type):
    """Retrieve and process AERONET data."""
    d = parse_dates(dates)
    click.echo(f"Loading AERONET data for {d if d is not None else 'default dates'}...")
    obj = monetio.load(
        "aeronet",
        dates=d,
        siteid=siteid,
        product=product,
        daily=daily,
        inv_type=inv_type,
        n_procs=n_procs,
        lazy=lazy,
        as_xarray=not as_pandas,
    )
    handle_save(obj, output, as_pandas)


@cli.command()
@common_options
@click.option("--daily", is_flag=True, help="Load daily data.")
@click.option(
    "--wide-fmt/--long-fmt",
    default=True,
    help="Whether to return data in wide format (pollutants as columns).",
)
def airnow(dates, output, n_procs, lazy, as_pandas, daily, wide_fmt):
    """Retrieve and process AirNow data."""
    d = parse_dates(dates)
    click.echo(f"Loading AirNow data for {d if d is not None else 'default dates'}...")
    obj = monetio.load(
        "airnow",
        dates=d,
        daily=daily,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        lazy=lazy,
        as_xarray=not as_pandas,
    )
    handle_save(obj, output, as_pandas)


@cli.command()
@common_options
@click.option("--param", "-p", multiple=True, help="AQS parameter(s) to retrieve.")
@click.option("--daily", is_flag=True, help="Load daily data.")
@click.option("--network", help="Filter by network name.")
def aqs(dates, output, n_procs, lazy, as_pandas, param, daily, network):
    """Retrieve and process EPA AQS data."""
    d = parse_dates(dates)
    p = list(param) if param else None
    click.echo(f"Loading AQS data for {d if d is not None else 'default dates'}...")
    obj = monetio.load(
        "aqs",
        dates=d,
        param=p,
        daily=daily,
        network=network,
        n_procs=n_procs,
        lazy=lazy,
        as_xarray=not as_pandas,
    )
    handle_save(obj, output, as_pandas)


@cli.command()
@common_options
@click.option(
    "--wide-fmt/--long-fmt",
    default=True,
    help="Whether to return data in wide format.",
)
def openaq(dates, output, n_procs, lazy, as_pandas, wide_fmt):
    """Retrieve and process OpenAQ data."""
    d = parse_dates(dates)
    click.echo(f"Loading OpenAQ data for {d if d is not None else 'default dates'}...")
    obj = monetio.load(
        "openaq",
        dates=d,
        wide_fmt=wide_fmt,
        n_procs=n_procs,
        lazy=lazy,
        as_xarray=not as_pandas,
    )
    handle_save(obj, output, as_pandas)


@cli.command()
@click.argument("files", nargs=-1)
@click.option(
    "--output", "-o", required=True, help="Output JSON file path for the Kerchunk reference."
)
@click.option(
    "--concat-dim", default="time", help="Dimension to concatenate along (default 'time')."
)
def kerchunk(files, output, concat_dim):
    """
    Pre-process files into a single Kerchunk reference JSON map.
    Especially useful for large geospatial datasets (MERRA2, GFS, etc.).
    """
    if not files:
        click.echo("Error: No files provided.")
        return

    # Expand any glob strings (e.g. s3://bucket/*.nc)
    from monetio.readers.drivers import FileUtility

    expanded_files = []
    for f in files:
        try:
            expanded_files.extend(FileUtility.expand_paths(f))
        except FileNotFoundError:
            pass

    if not expanded_files:
        click.echo("Error: No valid files found matching patterns.")
        return

    files = expanded_files

    click.echo(f"Kerchunking {len(files)} file(s) into {output}...")
    from monetio.readers.drivers import XarrayDriver

    # We use XarrayDriver.open to handle the parsing natively and drop the dataset payload.
    # It automatically saves the references to `kerchunk_file`.
    xd = XarrayDriver()
    try:
        xd.open(list(files), use_kerchunk=True, kerchunk_file=output, concat_dim=concat_dim)
        click.echo(f"Successfully generated {output}")
    except Exception as e:
        click.echo(f"Error generating Kerchunk reference: {e}")


if __name__ == "__main__":
    cli()
