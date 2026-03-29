import pytest
import datetime
from monetio.readers.rrfs import RRFSReader

def test_rrfs_build_urls():
    reader = RRFSReader()
    urls = reader.build_urls(dates="2026-03-28", hour=0, lead_time=1, product="prslev.3km", domain="conus")
    assert len(urls) == 1
    assert "rrfs.20260328" in urls[0]
    assert "rrfs.t00z.prslev.3km.f001.conus.grib2" in urls[0]
    assert "noaa-rrfs-pds" in urls[0]
