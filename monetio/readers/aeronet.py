"""AERONET Reader"""

import warnings
from datetime import datetime
from functools import lru_cache, partial
import numpy as np
import pandas as pd
from .base import PointReader, register_reader
from io import BytesIO

try:
    import dask
    has_dask = True
except ImportError:
    has_dask = False

@register_reader("aeronet")
class AERONETReader(PointReader):
    def open_dataset(self,
                     dates=None,
                     product="AOD15",
                     inv_type=None,
                     latlonbox=None,
                     siteid=None,
                     daily=False,
                     lunar=False,
                     freq=None,
                     detect_dust=False,
                     interp_to_aod_values=None,
                     n_procs=1,
                     verbose=10,
                     files=None,
                     **kwargs):
        """
        Reads AERONET data.
        """
        if files:
            if isinstance(files, str):
                files = [files]

            dfs = []
            for f in files:
                a = AERONET()
                try:
                    with open(f) as fid:
                        if "Inversion" in fid.readline():
                            a.inv_type = True
                except:
                    pass

                a.new_aod_values = interp_to_aod_values
                a.url = f
                a.read_aeronet()

                if freq is not None and not a.df.empty:
                    a.df = (
                        a.df.set_index("time")
                        .groupby("siteid")
                        .resample(freq)
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                if detect_dust:
                    a.dust_detect()
                if a.new_aod_values is not None:
                    a.calc_new_aod_values()

                dfs.append(a.df)

            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs)

        else:
            a = AERONET()
            if interp_to_aod_values is not None:
                interp_to_aod_values = np.asarray(interp_to_aod_values)

            kwargs_inner = dict(
                product=product,
                inv_type=inv_type,
                latlonbox=latlonbox,
                siteid=siteid,
                daily=daily,
                lunar=lunar,
                detect_dust=detect_dust,
                interp_to_aod_values=interp_to_aod_values,
            )

            requested_parallel = n_procs != 1
            dates = pd.to_datetime(dates)

            if dates is not None:
                min_date = dates.min()
                max_date = dates.max()
                time_bounds = pd.date_range(start=min_date, end=max_date, freq="D")
                if max_date not in time_bounds:
                    time_bounds = time_bounds.append(pd.DatetimeIndex([max_date]))
            else:
                time_bounds = []

            if has_dask and requested_parallel and dates is not None and len(time_bounds) > 2:
                tasks = [
                    dask.delayed(_parallel_aeronet_call)(pd.DatetimeIndex([t1, t2]), **kwargs_inner, freq=None)
                    for t1, t2 in zip(time_bounds[:-1], time_bounds[1:])
                ]
                dfs = dask.compute(*tasks, scheduler="processes", num_workers=n_procs)
                df = pd.concat(dfs, ignore_index=True).drop_duplicates()
                if freq is not None:
                    df.index = df.time
                    df = df.groupby("siteid").resample(freq).mean(numeric_only=True).reset_index()
                return df.reset_index(drop=True)
            else:
                return a.add_data(dates=dates, freq=freq, **kwargs_inner)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

@lru_cache(1)
def get_valid_sites():
    from urllib.error import URLError
    try:
        df = pd.read_csv(
            "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt",
            skiprows=1,
        ).rename(
            columns={
                "Site_Name": "siteid",
                "Longitude(decimal_degrees)": "longitude",
                "Latitude(decimal_degrees)": "latitude",
                "Elevation(meters)": "elevation",
            },
        )
    except URLError:
        print("getting valid sites failed")
        return None
    except Exception:
        raise
    return df

def _parallel_aeronet_call(**kwargs):
    a = AERONET()
    return a.add_data(**kwargs)

class AERONET:
    _valid_prod_noninv = (
        "AOD10", "AOD15", "AOD20", "SDA10", "SDA15", "SDA20", "TOT10", "TOT15", "TOT20",
    )
    _valid_prod_inv = (
        "SIZ", "RIN", "CAD", "VOL", "TAB", "AOD", "SSA", "ASY", "FRC", "LID", "FLX",
    )
    _valid_inv_type = ("ALM15", "ALM20", "HYB15", "HYB20")

    def __init__(self):
        self.url = None
        self.df = None
        self.dates = None
        self.prod = None
        self.inv_type = None
        self.daily = None
        self.lunar = None
        self.latlonbox = None
        self.siteid = None
        self.new_aod_values = None

        # Buffer to store downloaded content
        self._content_buffer = None

    def build_url(self):
        assert self.dates is not None, "required parameter"
        d1, d2 = self.dates.min(), self.dates.max()
        sy, sm, sd, sh = d1.strftime(r"%Y"), d1.strftime(r"%m"), d1.strftime(r"%d"), d1.strftime(r"%H")
        ey, em, ed, eh = d2.strftime(r"%Y"), d2.strftime(r"%m"), d2.strftime(r"%d"), d2.strftime(r"%H")
        dates_ = (
            f"year={sy}&month={sm}&day={sd}&hour={sh}"
            f"&year2={ey}&month2={em}&day2={ed}&hour2={eh}"
        )

        assert self.prod is not None, "required parameter"

        if self.inv_type is None:
            if self.prod in self._valid_prod_noninv:
                base_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?"
            else:
                raise ValueError(f"invalid product {self.prod!r}")
            inv_type_ = ""
            product_ = f"&{self.prod}=1"

        elif self.inv_type in self._valid_inv_type:
            if self.prod in self._valid_prod_inv:
                base_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?"
            else:
                raise ValueError(f"invalid product {self.prod!r}")
            inv_type_ = f"&{self.inv_type}=1"
            product_ = f"&product={self.prod}"
        else:
            raise ValueError(f"invalid inv type: {self.inv_type!r}")

        assert self.daily in {10, 20}, "required parameter"
        avg_ = f"&AVG={self.daily}"

        if self.lunar is not None:
            if self.lunar in {0, 1}:
                lunar_ = f"&lunar_merge={self.lunar}"
            else:
                raise ValueError(f"invalid lunar setting {self.lunar!r}")
        else:
            lunar_ = ""

        if self.siteid is not None:
            if self.siteid in get_valid_sites().siteid.values:
                loc_ = f"&site={self.siteid}"
            else:
                raise ValueError(f"invalid site {self.siteid!r}")
        elif self.latlonbox is None:
            loc_ = ""
        else:
            lat1, lon1, lat2, lon2 = map(str, map(float, self.latlonbox))
            loc_ = f"&lat1={lat1}&lat2={lat2}&lon1={lon1}&lon2={lon2}"

        self.url = f"{base_url}{dates_}{product_}{avg_}{lunar_}{inv_type_}{loc_}&if_no_html=1"

    def _get_content(self, timeout=60, retries=3):
        """Robustly fetch content from URL."""
        if not (isinstance(self.url, str) and self.url.startswith("http")):
            return None # Local file handled elsewhere

        if self._content_buffer:
            self._content_buffer.seek(0)
            return self._content_buffer

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        response = session.get(self.url, timeout=timeout)
        response.raise_for_status()

        self._content_buffer = BytesIO(response.content)
        return self._content_buffer

    def _lines_from_url(self, *, n=10):
        from itertools import islice

        if isinstance(self.url, str) and self.url.startswith("http"):
            # Use the robust fetcher
            content = self._get_content()
            # Read first n lines from bytes buffer
            # Need to decode carefully
            content.seek(0)
            # Read a chunk that should contain enough lines?
            # Or just wrap in TextIOWrapper for iteration?
            import io
            wrapper = io.TextIOWrapper(content, encoding='utf-8', errors='replace')
            # iter_lines in requests yields lines without newlines? No, this is TextIOWrapper.
            # We strip to match behavior of 'iter_lines' join logic often used or just to look nice.
            s = "\n".join(line.rstrip('\n') for line in islice(wrapper, n))
            wrapper.detach() # Don't close the BytesIO
            content.seek(0) # Reset
        else:
            with open(self.url) as f:
                s = "\n".join(islice(f, n))
        return s

    def read_aeronet(self):
        print("Reading Aeronet Data...")
        inv = self.inv_type is not None
        skiprows = 5 if not inv else 6

        # This will trigger download if needed
        info = self._lines_from_url(n=skiprows)

        if len(info.splitlines()) == 1:
            raise Exception("valid query but no data found")
        elif info.startswith("<html>"):
            raise Exception("invalid query, open the URL to check the error")

        # Determine source for read_csv
        if self._content_buffer:
            self._content_buffer.seek(0)
            source = self._content_buffer
        else:
            source = self.url

        df = pd.read_csv(
            source,
            engine="python",
            header="infer",
            skiprows=skiprows,
            parse_dates={"time": [1, 2]},
            usecols=None,
            date_parser=lambda x: datetime.strptime(x, r"%d:%m:%Y %H:%M:%S"),
            na_values=-999,
        )
        df.rename(columns=str.lower, inplace=True)
        df.rename(
            columns={
                "aeronet_site": "siteid",
                "aeronet_aeronet_site": "siteid",
                "site_latitude(degrees)": "latitude",
                "site_longitude(degrees)": "longitude",
                "site_elevation(m)": "elevation",
                "latitude(degrees)": "latitude",
                "longitude(degrees)": "longitude",
                "elevation(m)": "elevation",
            },
            inplace=True,
        )
        if df.siteid.unique().size == 1:
            df.set_index("time", inplace=True)
        df.dropna(subset=["latitude", "longitude"], inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        if hasattr(df, "attrs"):
            df.attrs["info"] = info
        self.df = df

    def add_data(
        self,
        dates=None,
        product="AOD15",
        *,
        inv_type=None,
        siteid=None,
        latlonbox=None,
        daily=False,
        lunar=False,
        freq=None,
        detect_dust=False,
        interp_to_aod_values=None,
    ):
        self.latlonbox = latlonbox
        self.siteid = siteid
        if dates is None:
            now = datetime.utcnow()
            self.dates = pd.date_range(start=now.date(), end=now, freq="H")
        else:
            self.dates = pd.DatetimeIndex(dates)

        self.prod = product.upper() if product else product
        self.inv_type = inv_type
        self.daily = 20 if daily else 10
        self.lunar = 1 if lunar else 0
        self.new_aod_values = interp_to_aod_values

        if self.new_aod_values is not None and not self.prod.startswith("AOD"):
            print("`interp_to_aod_values` will be ignored")

        self.build_url()
        try:
            self.read_aeronet()
        except Exception as e:
            raise Exception(f"loading from URL {self.url!r} failed.") from e

        if freq is not None:
            self.df = (
                self.df.set_index("time")
                .groupby("siteid")
                .resample(freq)
                .mean(numeric_only=True)
                .reset_index()
            )

        if detect_dust:
            self.dust_detect()

        if self.new_aod_values is not None:
            self.calc_new_aod_values()

        return self.df

    def dust_detect(self):
        self.df["dust"] = (self.df["aod_1020nm"] > 0.3) & (self.df["440-870_angstrom_exponent"] < 0.6)

    def calc_new_aod_values(self):
        def _tspack_aod_interp(row, new_wv=[440.0, 470.0, 550.0, 670.0, 870.0, 1020.0, 1240.0]):
            import numpy as np
            try:
                import pytspack
            except ImportError as e:
                raise RuntimeError("You must install pytspack before using this function.") from e

            new_wv = np.asarray(new_wv)
            aod_columns = [c for c in row.index if c.startswith("aod_")]
            aods = row[aod_columns]
            wv = [float(c.replace("aod_", "").replace("nm", "")) for c in aod_columns]

            a = pd.DataFrame({"aod": aods}).reset_index()
            a["wv"] = wv
            df_aod_nu = a.dropna()
            df_aod_nu_sorted = df_aod_nu.sort_values(by="wv").dropna()
            if len(df_aod_nu_sorted) < 2:
                return new_wv * np.nan
            else:
                x, y, yp, sigma = pytspack.tspsi(df_aod_nu_sorted.wv.values, df_aod_nu_sorted.aod.values)
                yi = pytspack.hval(self.new_aod_values, x, y, yp, sigma)
                return yi

        out = self.df.apply(_tspack_aod_interp, axis=1, result_type="expand", new_wv=self.new_aod_values)
        names = "aod_" + pd.Series(self.new_aod_values.astype(int).astype(str)) + "nm"
        out.columns = names.values
        dup_names = list(set(self.df) & set(out))
        if dup_names:
            suff = "_orig"
            warnings.warn(f"Renaming duplicate AOD columns {dup_names} by adding suffix '{suff}'.", stacklevel=2)
            for name in dup_names:
                self.df = self.df.rename(columns={name: f"{name}{suff}"})
                if self.daily == 10:
                    wl = name[4:-2]
                    ename = f"exact_wavelengths_of_aod(um)_{wl}nm"
                    ename_new = f"exact_wavelengths_of_aod(um)_{wl}nm{suff}"
                    self.df = self.df.rename(columns={ename: ename_new})
        self.df = pd.concat([self.df, out], axis=1)
