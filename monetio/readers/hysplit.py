"""HYSPLIT Reader"""

import datetime

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("hysplit")
class HYSPLITReader(GriddedReader):
    def open_dataset(
        self,
        files,
        drange=None,
        century=None,
        verbose=False,
        sample_time_stamp="start",
        check_grid=True,
        **kwargs,
    ):
        """
        Reads HYSPLIT binary concentration (cdump) files.
        """
        file_list = FileUtility.expand_paths(files)

        if len(file_list) == 1:
            return open_dataset_hysplit(
                file_list[0],
                drange=drange,
                century=century,
                verbose=verbose,
                sample_time_stamp=sample_time_stamp,
                check_grid=check_grid,
            )
        else:
            blist = [(f, f, "met") for f in file_list]
            return combine_dataset(
                blist,
                drange=drange,
                century=century,
                verbose=verbose,
                sample_time_stamp=sample_time_stamp,
                check_grid=check_grid,
            )

    def harmonize(self, ds):
        return ds


# -----------------------------------------------------------------------------
# HYSPLIT Core Logic Ported
# -----------------------------------------------------------------------------


def open_dataset_hysplit(
    fname,
    drange=None,
    century=None,
    verbose=False,
    sample_time_stamp="start",
    check_grid=True,
):
    binfile = ModelBin(
        fname,
        drange=drange,
        century=century,
        verbose=verbose,
        readwrite="r",
        sample_time_stamp=sample_time_stamp,
    )
    dset = binfile.dset
    if check_grid:
        return fix_grid_continuity(dset)
    else:
        return dset


class ModelBin:
    def __init__(
        self,
        filename,
        drange=None,
        century=None,
        verbose=True,
        readwrite="r",
        sample_time_stamp="start",
    ):
        self.drange = drange
        self.filename = filename
        self.century = century
        self.verbose = verbose
        self.zeroconcdates = []
        self.nonzeroconcdates = []
        self.atthash = {}
        self.atthash["Starting Latitudes"] = []
        self.atthash["Starting Longitudes"] = []
        self.atthash["Starting Heights"] = []
        self.atthash["Source Date"] = []
        self.sample_time_stamp = sample_time_stamp
        self.gridhash = {}
        self.levels = None
        self.dset = xr.Dataset()

        if readwrite == "r":
            if verbose:
                print("reading " + filename)
            self.dataflag = self.readfile(filename, drange, verbose=verbose, century=century)

    @staticmethod
    def define_struct():
        from numpy import dtype

        real4 = ">f"
        int4 = ">i"
        int2 = ">i2"
        char4 = ">a4"

        rec1 = dtype(
            [
                ("pad1", int4),
                ("model_id", char4),
                ("met_year", int4),
                ("met_month", int4),
                ("met_day", int4),
                ("met_hr", int4),
                ("met_fhr", int4),
                ("start_loc", int4),
                ("conc_pack", int4),
                ("pad2", int4),
            ]
        )

        rec2 = dtype(
            [
                ("pad1", int4),
                ("r_year", int4),
                ("r_month", int4),
                ("r_day", int4),
                ("r_hr", int4),
                ("s_lat", real4),
                ("s_lon", real4),
                ("s_ht", real4),
                ("r_min", int4),
                ("pad2", int4),
            ]
        )

        rec3 = dtype(
            [
                ("pad1", int4),
                ("nlat", int4),
                ("nlon", int4),
                ("dlat", real4),
                ("dlon", real4),
                ("llcrnr_lat", real4),
                ("llcrnr_lon", real4),
                ("pad2", int4),
            ]
        )

        rec4a = dtype([("pad1", int4), ("nlev", int4)])
        rec4b = dtype([("levht", int4)])
        rec5a = dtype([("pad1", int4), ("pad2", int4), ("pollnum", int4)])
        rec5b = dtype([("pname", char4)])
        rec5c = dtype([("pad2", int4)])
        rec6 = dtype(
            [
                ("pad1", int4),
                ("oyear", int4),
                ("omonth", int4),
                ("oday", int4),
                ("ohr", int4),
                ("omin", int4),
                ("oforecast", int4),
                ("pad3", int4),
            ]
        )
        rec8a = dtype(
            [
                ("pad1", int4),
                ("poll", char4),
                ("lev", int4),
                ("ne", int4),
            ]
        )
        rec8b = dtype([("indx", int2), ("jndx", int2), ("conc", real4)])
        rec8c = dtype([("pad2", int4)])

        return (rec1, rec2, rec3, rec4a, rec4b, rec5a, rec5b, rec5c, rec6, rec8a, rec8b, rec8c)

    def parse_header(self, hdata1):
        if len(hdata1["start_loc"]) != 1:
            print("WARNING in ModelBin _readfile - number of starting locations incorrect")
        nstartloc = hdata1["start_loc"][0]
        self.atthash["Meteorological Model ID"] = hdata1["model_id"][0].decode("UTF-8")
        self.atthash["Number Start Locations"] = nstartloc
        return nstartloc

    def parse_hdata2(self, hdata2, nstartloc, century):
        for nnn in range(0, nstartloc):
            lat = hdata2["s_lat"][nnn]
            lon = hdata2["s_lon"][nnn]
            hgt = hdata2["s_ht"][nnn]

            self.atthash["Starting Latitudes"].append(lat)
            self.atthash["Starting Longitudes"].append(lon)
            self.atthash["Starting Heights"].append(hgt)

            if century is None:
                if hdata2["r_year"][0] < 50:
                    century = 2000
                else:
                    century = 1900
                print("WARNING: Guessing Century for HYSPLIT concentration file", century)

            sourcedate = datetime.datetime(
                century + hdata2["r_year"][nnn],
                hdata2["r_month"][nnn],
                hdata2["r_day"][nnn],
                hdata2["r_hr"][nnn],
                hdata2["r_min"][nnn],
            )
            self.atthash["Source Date"].append(sourcedate.strftime("%Y%m%d.%H%M%S"))
        return century

    def parse_hdata3(self, hdata3):
        ahash = {}
        ahash["Number Lat Points"] = hdata3["nlat"][0]
        ahash["Number Lon Points"] = hdata3["nlon"][0]
        ahash["Latitude Spacing"] = hdata3["dlat"][0]
        ahash["Longitude Spacing"] = hdata3["dlon"][0]
        ahash["llcrnr longitude"] = hdata3["llcrnr_lon"][0]
        ahash["llcrnr latitude"] = hdata3["llcrnr_lat"][0]
        return ahash

    def parse_hdata4(self, hdata4a, hdata4b):
        self.levels = hdata4b["levht"]
        self.atthash["Number of Levels"] = hdata4a["nlev"][0]
        self.atthash["Level top heights (m)"] = hdata4b["levht"]

    def parse_hdata6and7(self, hdata6, hdata7, century):
        if not hdata6.size:
            return False, None, None
        pdate1 = datetime.datetime(
            century + int(hdata6["oyear"][0]),
            int(hdata6["omonth"][0]),
            int(hdata6["oday"][0]),
            int(hdata6["ohr"][0]),
            int(hdata6["omin"][0]),
        )
        pdate2 = datetime.datetime(
            century + int(hdata7["oyear"][0]),
            int(hdata7["omonth"][0]),
            int(hdata7["oday"][0]),
            int(hdata7["ohr"][0]),
            int(hdata7["omin"][0]),
        )
        dt = pdate2 - pdate1
        sample_dt = dt.days * 24 + dt.seconds / 3600.0
        self.atthash["sample time hours"] = sample_dt
        if self.sample_time_stamp == "end":
            self.atthash["time description"] = "End of sampling time period"
        else:
            self.atthash["time description"] = "start of sampling time period"
        return True, pdate1, pdate2

    @staticmethod
    def parse_hdata8(hdata8a, hdata8b, pdate1):
        lev_name = hdata8a["lev"][0]
        col_name = hdata8a["poll"][0].decode("UTF-8")
        edata = hdata8b.byteswap().newbyteorder()
        concframe = pd.DataFrame.from_records(edata)
        concframe["levels"] = lev_name
        concframe["time"] = pdate1

        names = concframe.columns.values
        names = ["y" if x == "jndx" else x for x in names]
        names = ["x" if x == "indx" else x for x in names]
        names = ["z" if x == "levels" else x for x in names]
        concframe.columns = names
        concframe.set_index(["time", "z", "y", "x"], inplace=True)
        concframe.rename(columns={"conc": col_name}, inplace=True)
        return concframe

    def readfile(self, filename, drange, verbose, century):
        # Use FileUtility to open file (supports S3)
        fs = FileUtility.get_fs(filename)
        fid = fs.open(filename, "rb")

        recs = self.define_struct()
        rec1, rec2, rec3, rec4a = recs[0], recs[1], recs[2], recs[3]
        rec4b, rec5a, rec5b, rec5c = recs[4], recs[5], recs[6], recs[7]
        rec6, rec8a, rec8b, rec8c = recs[8], recs[9], recs[10], recs[11]

        hdata1 = np.fromfile(fid, dtype=rec1, count=1)
        nstartloc = self.parse_header(hdata1)

        hdata2 = np.fromfile(fid, dtype=rec2, count=nstartloc)
        century = self.parse_hdata2(hdata2, nstartloc, century)

        hdata3 = np.fromfile(fid, dtype=rec3, count=1)
        self.gridhash = self.parse_hdata3(hdata3)
        if self.verbose:
            print("Grid specs", self.gridhash)

        hdata4a = np.fromfile(fid, dtype=rec4a, count=1)
        hdata4b = np.fromfile(fid, dtype=rec4b, count=hdata4a["nlev"][0])
        self.parse_hdata4(hdata4a, hdata4b)

        hdata5a = np.fromfile(fid, dtype=rec5a, count=1)
        np.fromfile(fid, dtype=rec5b, count=hdata5a["pollnum"][0])
        np.fromfile(fid, dtype=rec5c, count=1)
        self.atthash["Number of Species"] = hdata5a["pollnum"][0]
        self.atthash["Species ID"] = []

        iimax = 0
        iii = 0
        imax = 1e8
        testf = True

        while testf:
            hdata6 = np.fromfile(fid, dtype=rec6, count=1)
            hdata7 = np.fromfile(fid, dtype=rec6, count=1)
            check, pdate1, pdate2 = self.parse_hdata6and7(hdata6, hdata7, century)
            if not check:
                break

            testf, savedata = check_drange(drange, pdate1, pdate2)
            if verbose:
                print("sample time", pdate1, " to ", pdate2)

            inc_iii = False
            for _ in range(self.atthash["Number of Levels"]):
                for _ in range(self.atthash["Number of Species"]):
                    hdata8a = np.fromfile(fid, dtype=rec8a, count=1)
                    if hdata8a["ne"] >= 1:
                        self.atthash["Species ID"].append(hdata8a["poll"][0].decode("UTF-8"))
                        hdata8b = np.fromfile(fid, dtype=rec8b, count=hdata8a["ne"][0])
                        self.nonzeroconcdates.append(pdate1)
                    else:
                        self.zeroconcdates.append(pdate1)

                    np.fromfile(fid, dtype=rec8c, count=1)

                    if savedata and hdata8a["ne"] >= 1:
                        self.nonzeroconcdates.append(pdate1)
                        inc_iii = True
                        if self.sample_time_stamp == "end":
                            concframe = self.parse_hdata8(hdata8a, hdata8b, pdate2)
                        else:
                            concframe = self.parse_hdata8(hdata8a, hdata8b, pdate1)
                        dset = xr.Dataset.from_dataframe(concframe)
                        if not self.dset.any():
                            self.dset = dset
                        else:
                            self.dset = xr.merge([self.dset, dset])
                        iimax += 1
            if iimax > imax:
                testf = False
            if inc_iii:
                iii += 1

        fid.close()

        self.atthash.update(self.gridhash)
        self.atthash["Species ID"] = list(set(self.atthash["Species ID"]))
        self.atthash["Coordinate time description"] = "Beginning of sampling time"

        if not self.dset.any():
            return False
        if self.dset.variables:
            self.dset.attrs = self.atthash
            mgrid = get_latlongrid(self.gridhash, self.dset.coords["x"], self.dset.coords["y"])
            self.dset = self.dset.assign_coords(longitude=(("y", "x"), mgrid[0]))
            self.dset = self.dset.assign_coords(latitude=(("y", "x"), mgrid[1]))
            self.dset = self.dset.reset_coords()
            self.dset = self.dset.set_coords(["time", "latitude", "longitude"])
        if iii == 0 and verbose:
            print("Warning: ModelBin class _readfile method: no data in the date range found")
            return False
        return True


def check_drange(drange, pdate1, pdate2):
    savedata = True
    testf = True
    if drange is None:
        savedata = True
    elif pdate1 >= drange[0] and pdate1 <= drange[1] and pdate2 <= drange[1]:
        savedata = True
    elif pdate1 > drange[1] or pdate2 > drange[1]:
        testf = False
        savedata = False
    else:
        savedata = False
    return testf, savedata


def fix_grid_continuity(dset):
    if not dset.any():
        return dset
    if check_grid_continuity(dset):
        return dset
    xvv = dset.x.values
    yvv = dset.y.values
    xlim = [xvv[0], xvv[-1]]
    ylim = [yvv[0], yvv[-1]]
    xindx = np.arange(xlim[0], xlim[1] + 1)
    yindx = np.arange(ylim[0], ylim[1] + 1)
    mgrid = get_latlongrid(dset.attrs, xindx, yindx)
    conc = np.zeros_like(mgrid[0])
    dummy = xr.DataArray(conc, dims=["y", "x"])
    dummy = dummy.assign_coords(latitude=(("y", "x"), mgrid[1]))
    dummy = dummy.assign_coords(longitude=(("y", "x"), mgrid[0]))
    dummy = dummy.assign_coords(x=(("x"), xindx))
    dummy = dummy.assign_coords(y=(("y"), yindx))
    cdset, dummy2 = xr.align(dset, dummy, join="outer")
    cdset = cdset.assign_coords(latitude=(("y", "x"), mgrid[1]))
    cdset = cdset.assign_coords(longitude=(("y", "x"), mgrid[0]))
    return cdset.fillna(0)


def check_grid_continuity(dset):
    xvv = dset.x.values
    yvv = dset.y.values
    tt1 = np.array([xvv[i] - xvv[i - 1] for i in np.arange(1, len(xvv))])
    tt2 = np.array([yvv[i] - yvv[i - 1] for i in np.arange(1, len(yvv))])
    if np.any(tt1 != 1):
        return False
    if np.any(tt2 != 1):
        return False
    return True


def get_latlongrid(attrs, xindx, yindx):
    xindx = np.array(xindx)
    yindx = np.array(yindx)
    if np.any(xindx <= 0):
        raise Exception("HYSPLIT grid error xindex <=0")
    if np.any(yindx <= 0):
        raise Exception("HYSPLIT grid error yindex <=0")
    lat, lon = getlatlon(attrs)
    success = True
    try:
        lonlist = [lon[x - 1] for x in xindx]
    except Exception:
        success = False
    try:
        latlist = [lat[x - 1] for x in yindx]
    except Exception:
        success = False
    if not success:
        return None
    mgrid = np.meshgrid(lonlist, latlist)
    return mgrid


def getlatlon(attrs):
    lon_tolerance = 0.001
    llcrnr_lat = attrs["llcrnr latitude"]
    llcrnr_lon = attrs["llcrnr longitude"]
    nlat = attrs["Number Lat Points"]
    nlon = attrs["Number Lon Points"]
    dlat = attrs["Latitude Spacing"]
    dlon = attrs["Longitude Spacing"]
    lastlon = llcrnr_lon + (nlon - 1) * dlon
    lastlat = llcrnr_lat + (nlat - 1) * dlat
    lat = np.linspace(llcrnr_lat, lastlat, num=int(nlat))
    lon = np.linspace(llcrnr_lon, lastlon, num=int(nlon))
    lon = np.array([x - 360 if x >= 180 + lon_tolerance else x for x in lon])
    return lat, lon


def combine_dataset(
    blist,
    drange=None,
    species=None,
    century=None,
    verbose=False,
    sample_time_stamp="start",
    check_grid=True,
):
    import sys

    mlat_p = mlon_p = None
    ylist = []
    dtlist = []
    splist = []
    sourcelist = []

    aaa = sorted(blist, key=lambda x: x[1])
    blist_dict = {}
    for val in aaa:
        if val[1] in blist_dict.keys():
            blist_dict[val[1]].append((val[0], val[2]))
        else:
            blist_dict[val[1]] = [(val[0], val[2])]

    xlist = []
    sourcelist = []
    enslist = []
    for iii, key in enumerate(blist_dict):
        xsublist = []
        for fname in blist_dict[key]:
            if drange:
                century = int(drange[0].year / 100) * 100
                hxr = open_dataset_hysplit(
                    fname[0],
                    drange=drange,
                    century=century,
                    verbose=verbose,
                    sample_time_stamp=sample_time_stamp,
                    check_grid=False,
                )
            else:
                hxr = open_dataset_hysplit(
                    fname[0],
                    century=century,
                    verbose=verbose,
                    sample_time_stamp=sample_time_stamp,
                    check_grid=False,
                )
            try:
                mlat, mlon = getlatlon(hxr.attrs)
            except Exception:
                print("WARNING Cannot open " + fname[0])
            if iii > 0:
                tt1 = np.array_equal(mlat, mlat_p)
                tt2 = np.array_equal(mlon, mlon_p)
                if not tt1 or not tt2:
                    print("WARNING: grids are not the same. cannot combine")
                    sys.exit()
            mlat_p = mlat
            mlon_p = mlon

            xrash = add_species(hxr, species=species)
            xsublist.append(xrash)
            enslist.append(fname[1])
            dtlist.append(hxr.attrs["sample time hours"])
            splist.extend(xrash.attrs["Species ID"])
            if iii == 0:
                xnew = xrash.copy()
            else:
                aaa, xnew = xr.align(xrash, xnew, join="outer")
                xnew = xnew.fillna(0)
        sourcelist.append(key)
        xlist.append(xsublist)

    ylist = []
    slist = []
    for jjj, sublist in enumerate(xlist):
        hlist = []
        for iii, temp in enumerate(sublist):
            aaa, bbb = xr.align(temp, xnew, join="outer")
            aaa = aaa.fillna(0)
            bbb = bbb.fillna(0)
            aaa.expand_dims("ens")
            aaa["ens"] = enslist[iii]
            hlist.append(aaa)
        new = xr.concat(hlist, "ens")
        ylist.append(new)
        slist.append(sourcelist[jjj])

    if dtlist:
        dtlist = list(set(dtlist))
        dt = dtlist[0]

    newhxr = xr.concat(ylist, "source")
    newhxr["source"] = slist
    newhxr = newhxr.assign_attrs({"sample time hours": dt})
    newhxr = newhxr.assign_attrs({"Species ID": list(set(splist))})
    newhxr.attrs.update(hxr.attrs)

    newhxr = reset_latlon_coords(newhxr)
    if check_grid:
        rval = fix_grid_continuity(newhxr)
    else:
        rval = newhxr
    return rval


def add_species(dset, species=None):
    sflist = []
    splist = dset.attrs["Species ID"]
    if not species:
        species = dset.attrs["Species ID"]

    sss = 0
    tmp = []
    while sss < len(splist):
        if splist[sss] in species:
            tmp.append(dset[splist[sss]].fillna(0))
            sflist.append(splist[sss])
        sss += 1

    total_par = tmp[0]
    ppp = 1
    while ppp < len(tmp):
        total_par = total_par + tmp[ppp]
        ppp += 1
    atthash = dset.attrs
    atthash["Species ID"] = sflist
    total_par = total_par.assign_attrs(atthash)
    return total_par


def reset_latlon_coords(hxr):
    mgrid = get_latlongrid(hxr.attrs, hxr.x.values, hxr.y.values)
    if "latitude" in hxr.coords:
        hxr = hxr.drop("longitude")
    if "longitude" in hxr.coords:
        hxr = hxr.drop("latitude")
    hxr = hxr.assign_coords(latitude=(("y", "x"), mgrid[1]))
    hxr = hxr.assign_coords(longitude=(("y", "x"), mgrid[0]))
    return hxr
