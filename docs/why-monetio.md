# Overview: Why MONETIO?

## Features

Retrieving, loading, and combining data and putting into a common format
is the core of MONETIO. MONETIO uses the [pandas](https://pandas.pydata.org) and [xarray](https://xarray.pydata.org) data formats for data
analysis.

- **Open point observations in a common format**: [pandas] excels at working with tabular data or point measurements. It is used for time series analysis and statistical measures.
- **Open model and satellite data in a common format**: [xarray] is used when N-dimensional arrays are needed.
- **Retrieving observational datasets**: Easy access to datasets for given time and space.
- **Efficiently combine/interpolate**: Methods to align model and observational datasets.
- **Lazy loading**: Built on top of [dask](https://dask.org) for handling large datasets.

## Gallery

![Time Series](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/pm2.5_timeseries.jpg)
_Time Series_

![Time Series of RMSE](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/pm2.5_timeseries_rmse.jpg)
_Time Series of RMSE_

![Spatial Plots](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/ozone_spatial.jpg)
_Spatial Plots_

![Scatter Plots](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/no2_scatter.jpg)
![PDFS Plots](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/no2_pdf.jpg)
![Difference Scatter Plots](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/no2_diffscatter.jpg)
![Difference PDFS Plots](https://raw.githubusercontent.com/noaa-oar-arl/MONET/master/sample_figures/no2_diffpdf.jpg)

[pandas]: https://pandas.pydata.org
[xarray]: https://xarray.pydata.org
