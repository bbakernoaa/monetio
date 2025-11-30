"""IMPROVE Reader"""

import pandas as pd
from numpy import nan
from .base import PointReader, register_reader
from monetio.obs.epa_util import read_monitor_file

@register_reader("improve")
class IMPROVEReader(PointReader):
    def open_dataset(self,
                     files,
                     add_meta=False,
                     delimiter="\t",
                     **kwargs):
        """
        Reads IMPROVE data files.
        """
        # Expand paths
        from .drivers import FileUtility
        file_list = FileUtility.expand_paths(files)

        dfs = []
        for f in file_list:
            imp = IMPROVE()
            df = imp.add_data(f, add_meta=add_meta, delimiter=delimiter)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/improve_mod.py
# -----------------------------------------------------------------------------

class IMPROVE:
    def __init__(self):
        self.datestr = []
        self.df = None
        self.daily = True

    def add_data(self, fname, add_meta=False, delimiter="\t"):
        f = open(fname)
        lines = f.readlines()
        f.close()
        skiprows = 0
        skip = False
        for i, line in enumerate(lines):
            if line == "Data\n":
                skip = True
                skiprows = i + 1
                break

        if skip:
            df = pd.read_csv(
                fname,
                delimiter=delimiter,
                parse_dates=[2],
                infer_datetime_format=True,
                dtype={"EPACode": str},
                skiprows=skiprows,
            )
        else:
            df = pd.read_csv(
                fname,
                delimiter=delimiter,
                parse_dates=[2],
                infer_datetime_format=True,
                dtype={"EPACode": str},
            )

        df.rename(columns={"EPACode": "epaid", "Val": "Obs", "State": "state_name",
                           "ParamCode": "variable", "SiteCode": "siteid", "Unit": "Units",
                           "Date": "time"}, inplace=True)
        if "Dataset" in df.columns:
            df.drop("Dataset", axis=1, inplace=True)

        # df["time"] = pd.to_datetime(df.time, format="%Y%m%d") # Already parsed by read_csv usually
        df.columns = [i.lower() for i in df.columns]

        if "epaid" in df.columns:
            df["epaid"] = df.epaid.astype(str).str.zfill(9)

        if add_meta:
            monitor_df = read_monitor_file(network="IMPROVE")
            df = df.merge(monitor_df, how="left", left_on="epaid", right_on="siteid")
            df.drop(["siteid_y", "state_name_y"], inplace=True, axis=1)
            df.rename(columns={"siteid_x": "siteid", "state_name_x": "state_name"}, inplace=True)

        try:
            # df.obs.loc[df.obs < df.mdl] = nan # mdl might not exist
            pass
        except:
            pass

        return df.copy()
