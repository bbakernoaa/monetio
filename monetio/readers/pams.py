"""PAMS Reader"""

import json
import pandas as pd
from .base import PointReader, register_reader

@register_reader("pams")
class PAMSReader(PointReader):
    def open_dataset(self,
                     files,
                     **kwargs):
        """
        Reads PAMS JSON files.
        """
        # Expand paths
        from .drivers import FileUtility
        file_list = FileUtility.expand_paths(files)

        dfs = []
        for f in file_list:
            df = add_data_pams(f)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/pams.py
# -----------------------------------------------------------------------------

def open_json(filename):
    with open(filename) as f:
        jsonf = json.load(f)
    return jsonf

def add_data_pams(filename):
    jsonf = open_json(filename)
    dataf = jsonf.get("Data", [])
    data = pd.DataFrame.from_dict(dataf)

    if data.empty:
        return data

    data["siteid"] = (
        data.state_code.astype(str).str.zfill(2)
        + data.county_code.astype(str).str.zfill(3)
        + data.site_number.astype(str).str.zfill(4)
    )

    data["datetime_local"] = pd.to_datetime(data["date_local"] + " " + data["time_local"])
    data["datetime_utc"] = pd.to_datetime(data["date_gmt"] + " " + data["time_gmt"])

    data = data.rename(
        columns={
            "sample_measurement": "obs",
            "units_of_measure": "units",
            "units_of_measure_code": "unit_code",
        }
    )

    cols_to_drop = [
        "state_code", "county_code", "site_number", "datum", "qualifier", "uncertainty",
        "county", "state", "date_of_last_change", "date_local", "time_local",
        "date_gmt", "time_gmt", "poc", "unit_code", "sample_duration_code", "method_code",
    ]
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Reorder if columns exist
    # cols = data.columns.tolist()
    # Logic to insert siteid etc at start is cosmetic, skipping strict reorder to avoid key errors

    units = data.units.unique()
    for i in units:
        con = data.units == i
        if i.upper() == "Parts per billion Carbon".upper():
            data.loc[con, "units"] = "ppbC"
        if i == "Parts per billion":
            data.loc[con, "units"] = "ppb"
        if i == "Parts per million":
            data.loc[con, "units"] = "ppm"

    return data
