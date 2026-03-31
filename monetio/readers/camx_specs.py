"""CAMx variable specifications and diagnostic definitions."""

from typing import Dict

from .base import DiagnosticSpec

# Core species groups
# Ported from monetio/readers/camx.py
COARSE = ["CPRM", "CCRS"]
FINE = [
    "NA",
    "PSO4",
    "PNO3",
    "PNH4",
    "PH2O",
    "PCL",
    "PEC",
    "FPRM",
    "FCRS",
    "SOA1",
    "SOA2",
    "SOA3",
    "SOA4",
]
NOY_GAS = [
    "NO",
    "NO2",
    "NO3",
    "N2O5",
    "HONO",
    "HNO3",
    "PAN",
    "PANX",
    "PNA",
    "NTR",
    "CRON",
    "CRN2",
    "CRNO",
    "CRPX",
    "OPAN",
]
POC = ["SOA1", "SOA2", "SOA3", "SOA4"]

# Diagnostic definitions
DIAGNOSTICS: Dict[str, DiagnosticSpec] = {
    "PM25": DiagnosticSpec(
        variables=FINE,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="PM2.5",
        name="PM2.5",
    ),
    "PM10": DiagnosticSpec(
        variables=FINE + COARSE,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="Particulate Matter < 10 microns",
        name="PM10",
    ),
    "PM_COARSE": DiagnosticSpec(
        variables=COARSE,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="Coarse Mode Particulate Matter",
        name="PM_COARSE",
    ),
    "NOy": DiagnosticSpec(
        variables=NOY_GAS,
        weights=None,
        units="ppbV",
        long_name="NOy",
        name="NOy",
    ),
    "NOx": DiagnosticSpec(
        variables=["NO", "NOX"],
        weights=None,
        units="ppbV",
        long_name="NOx",
        name="NOx",
    ),
}
