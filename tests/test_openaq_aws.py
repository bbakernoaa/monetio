from monetio.obs.openaq_aws import (
    _build_urls,
    get_locations,
    get_paths,
    get_provider_countries,
    get_providers,
    read,
)


def test_read():
    url = (
        "s3://openaq-data-archive/records/csv.gz/"
        "locationid=2178/year=2022/month=05/"
        "location-2178-20220503.csv.gz"
    )
    df = read(url)
    assert len(df) > 0


def test_get_providers():
    providers = get_providers()
    assert len(providers) > 0
    providers_set = set(providers)
    assert len(providers_set) == len(providers), "unique"
    assert {"aqdc", "airnow"} <= providers_set


def test_get_provider_countries():
    countries = get_provider_countries("aqdc")
    assert len(countries) > 0
    assert {"us", "mobile"} <= set(countries)


def test_get_locations():
    df = get_locations(provider="aqdc", country="mobile")
    assert len(df) > 0
    assert df.provider.eq("aqdc").all()
    assert df.country.eq("mobile").all()


def test_get_paths_vs_build():
    date = "2019-08-01"
    siteid = "10000"
    paths = get_paths(date, siteid=siteid)
    naive_urls = _build_urls(date, siteid)
    assert len(paths) == len(naive_urls) == 1
    assert all(u.endswith(p) for p, u in zip(paths, naive_urls))
