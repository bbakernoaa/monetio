"""WRF-Chem variable specifications and diagnostic definitions."""

from typing import Dict

from .base import DiagnosticSpec

# GOCART Species Groups
GOCART_PM25 = [
    "P25",
    "BC1",
    "BC2",
    "OC1",
    "OC2",
    "SULF",
    "D1",
    "D2",
    "S1",
    "S2",
]

GOCART_PM10 = [
    "P10",
    "BC1",
    "BC2",
    "OC1",
    "OC2",
    "SULF",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "S1",
    "S2",
    "S3",
    "S4",
]

# Diagnostic definitions
DIAGNOSTICS: Dict[str, DiagnosticSpec] = {
    "PM25": DiagnosticSpec(
        variables=GOCART_PM25,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="PM2.5",
        name="PM25",
    ),
    "PM10": DiagnosticSpec(
        variables=GOCART_PM10,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="Particulate Matter < 10 microns",
        name="PM10",
    ),
    "NOx": DiagnosticSpec(
        variables=["NO", "NO2"],
        weights=None,
        units="ppbV",
        long_name="Nitrogen Oxides",
        name="NOx",
    ),
}
