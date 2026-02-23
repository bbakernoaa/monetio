"""Utilities for EPA AQS and IMPROVE data."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

from ..util import force_object_strings

if TYPE_CHECKING:
    import dask.dataframe as dd

logger = logging.getLogger(__name__)


def convert_statenames_to_abv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert full state names to two-letter abbreviations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a 'state_name' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'state_name' converted to abbreviations.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'state_name': ['Alabama', 'California']})
    >>> convert_statenames_to_abv(df)
      state_name
    0         AL
    1         CA
    """
    d = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "Canada": "CC",
        "Mexico": "MM",
    }
    if "state_name" in df.columns:
        df["state_name"] = df["state_name"].replace(d)
    return df


@lru_cache(maxsize=1)
def read_monitor_file(
    network: Optional[str] = None, airnow: bool = False, drop_latlon: bool = False
) -> pd.DataFrame:
    """
    Read the EPA/AQS monitor location file.

    Parameters
    ----------
    network : str, optional
        Filter by network (e.g., 'IMPROVE', 'CSN'), by default None.
    airnow : bool, optional
        Whether to load the AirNow monitor file instead, by default False.
    drop_latlon : bool, optional
        Whether to drop latitude and longitude columns, by default False.

    Returns
    -------
    pd.DataFrame
        The monitor metadata.

    Examples
    --------
    >>> # Read general AQS monitors
    >>> df = read_monitor_file()
    >>> # Read specific network monitors
    >>> df_improve = read_monitor_file(network='IMPROVE')
    """
    import os

    df = pd.DataFrame()
    if airnow:
        monitor_airnow_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/today/monitoring_site_locations.dat"
        colsinuse = [
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        ]
        try:
            df = pd.read_csv(
                monitor_airnow_url,
                delimiter="|",
                header=None,
                usecols=colsinuse,
                dtype={0: str},
                encoding="ISO-8859-1",
            )
            df.columns = [
                "siteid",
                "site_code",
                "site_name",
                "status",
                "agency",
                "agency_name",
                "epa_region",
                "latitude",
                "longitude",
                "elevation",
                "gmt_offset",
                "country_code",
                "cmsa_code",
                "cmsa_name",
                "msa_code",
                "msa_name",
                "state_code",
                "state_name",
                "county_code",
                "county_name",
                "city_code",
            ]
            df["airnow_flag"] = "AIRNOW"
            df.columns = [i.lower() for i in df.columns]
        except Exception as e:
            logger.error(f"Failed to read AirNow monitor file: {e}")
            df = pd.DataFrame()
    else:
        # Try local HDF first
        try:
            # Look in monetio/data/
            import monetio

            basedir = os.path.dirname(monetio.__file__)
            fname = os.path.join(basedir, "data", "monitoring_site_locations.hdf")
            if os.path.exists(fname):
                df = pd.read_hdf(fname)
        except Exception as e:
            logger.debug(f"Local monitor file not found or unreadable: {e}")

        if df.empty:
            # Fallback to downloading from AQS
            logger.info("Downloading monitor metadata from AQS...")
            baseurl = "https://aqs.epa.gov/aqsweb/airdata/"
            site_url = baseurl + "aqs_sites.zip"
            monitor_url = baseurl + "aqs_monitors.zip"

            try:
                # Read EPA Site file
                site = pd.read_csv(site_url, encoding="ISO-8859-1")
                # read epa monitor file
                monitor = pd.read_csv(monitor_url, encoding="ISO-8859-1")

                # make siteid column
                for d in [site, monitor]:
                    d["siteid"] = (
                        d["State Code"].astype(str).str.zfill(2)
                        + d["County Code"].astype(str).str.zfill(3)
                        + d["Site Number"].astype(str).str.zfill(4)
                    )

                site.columns = [i.replace(" ", "_") for i in site.columns]
                df = monitor.merge(
                    site[["siteid", "Land Use", "Location Setting", "GMT Offset"]],
                    on=["siteid"],
                    how="left",
                )
                df.columns = [i.replace(" ", "_").lower() for i in df.columns]

                monitor_drop = [
                    "state_code",
                    "county_code",
                    "site_number",
                    "extraction_date",
                    "parameter_code",
                    "parameter_name",
                    "poc",
                    "last_sample_date",
                    "pqao",
                    "reporting_agency",
                    "exclusions",
                    "monitoring_objective",
                    "last_method_code",
                    "last_method",
                    "naaqs_primary_monitor",
                    "qa_primary_monitor",
                ]
                df = df.drop(columns=monitor_drop, errors="ignore")
                df = convert_statenames_to_abv(df).dropna(subset=["latitude", "longitude"])
            except Exception as e:
                logger.error(f"Failed to download/process AQS monitor metadata: {e}")
                df = pd.DataFrame()

    if not df.empty:
        if network is not None:
            if "networks" in df.columns:
                df = df.loc[df.networks.str.contains(network, na=False, case=False)]

        if drop_latlon:
            df = df.drop(columns=["latitude", "longitude"], errors="ignore")

        df = force_object_strings(df).drop_duplicates()

    return df


def convert_epa_unit(
    df: Union[pd.DataFrame, "dd.DataFrame"],
    obscolumn: str = "obs",
    unit_column: str = "units",
    to_unit: str = "ug/m3",
    species: str = "SO2",
) -> Union[pd.DataFrame, "dd.DataFrame"]:
    """
    Convert EPA units (e.g., ppb to ug/m3).

    Parameters
    ----------
    df : Union[pd.DataFrame, "dd.DataFrame"]
        Input dataframe.
    obscolumn : str, optional
        Name of column with observation data, by default 'obs'.
    unit_column : str, optional
        Name of column with unit info, by default 'units'.
    to_unit : str, optional
        Target unit, by default 'ug/m3'.
    species : str, optional
        Species name for factor selection, by default 'SO2'.

    Returns
    -------
    Union[pd.DataFrame, "dd.DataFrame"]
        Dataframe with converted values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'obs': [10.0], 'units': ['ppb']})
    >>> df = convert_epa_unit(df, species='SO2', to_unit='ug/m3')
    >>> df['units'].iloc[0]
    'ug/m3'
    """
    # SO2 factor: 2.6178 (from legacy code)
    # Generic implementation could be expanded
    factors = {"SO2": 2.6178, "O3": 1.96, "NO2": 1.88, "CO": 1.145}
    factor = factors.get(species.upper(), 1.0)

    ppb = "ppb"
    ugm3 = "ug/m3"

    # Normalize input unit casing for comparison
    df[unit_column] = df[unit_column].astype(str)

    if to_unit.lower() == ugm3:
        mask = df[unit_column].str.lower() == ppb
        df[obscolumn] = df[obscolumn].mask(mask, df[obscolumn] * factor)
        df[unit_column] = df[unit_column].mask(mask, ugm3)
    elif to_unit.lower() == ppb:
        mask = df[unit_column].str.lower() == ugm3
        df[obscolumn] = df[obscolumn].mask(mask, df[obscolumn] / factor)
        df[unit_column] = df[unit_column].mask(mask, ppb)

    # Update history
    from .sat_utils import update_history

    df = update_history(df, f"Converted {species} to {to_unit}.")

    return df
