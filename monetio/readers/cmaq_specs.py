"""CMAQ variable specifications and diagnostic definitions."""

from typing import Dict, List, NamedTuple, Optional


class DiagnosticSpec(NamedTuple):
    """Specification for a derived diagnostic variable."""

    variables: List[str]
    weights: Optional[List[float]] = None
    units: str = "unknown"
    long_name: str = "unknown"
    name: str = "unknown"


# Core species groups
AITKEN = [
    "ACLI",
    "AECI",
    "ANAI",
    "ANH4I",
    "ANO3I",
    "AOTHRI",
    "APNCOMI",
    "APOCI",
    "ASO4I",
    "AORGAI",
    "AORGPAI",
    "AORGBI",
]

ACCUMULATION = [
    "AALJ",
    "AALK1J",
    "AALK2J",
    "ABNZ1J",
    "ABNZ2J",
    "ABNZ3J",
    "ACAJ",
    "ACLJ",
    "AECJ",
    "AFEJ",
    "AISO1J",
    "AISO2J",
    "AISO3J",
    "AKJ",
    "AMGJ",
    "AMNJ",
    "ANAJ",
    "ANH4J",
    "ANO3J",
    "AOLGAJ",
    "AOLGBJ",
    "AORGCJ",
    "AOTHRJ",
    "APAH1J",
    "APAH2J",
    "APAH3J",
    "APNCOMJ",
    "APOCJ",
    "ASIJ",
    "ASO4J",
    "ASQTJ",
    "ATIJ",
    "ATOL1J",
    "ATOL2J",
    "ATOL3J",
    "ATRP1J",
    "ATRP2J",
    "AXYL1J",
    "AXYL2J",
    "AXYL3J",
    "AORGAJ",
    "AORGPAJ",
    "AORGBJ",
]

COARSE = ["ACLK", "ACORS", "ANH4K", "ANO3K", "ASEACAT", "ASO4K", "ASOIL"]

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

# Diagnostic definitions
DIAGNOSTICS: Dict[str, DiagnosticSpec] = {
    "PM25": DiagnosticSpec(
        variables=AITKEN + ACCUMULATION + COARSE,
        weights=[1.0] * len(AITKEN + ACCUMULATION) + [0.2] * len(COARSE),
        units=r"$\mu g m^{-3}$",
        long_name="PM2.5",
        name="PM2.5",
    ),
    "PM10": DiagnosticSpec(
        variables=AITKEN + ACCUMULATION + COARSE,
        weights=None,  # All weights = 1.0
        units=r"$\mu g m^{-3}$",
        long_name="Particulate Matter < 10 microns",
        name="PM10",
    ),
    "PM_COURSE": DiagnosticSpec(
        variables=COARSE,
        weights=None,
        units=r"$\mu g m^{-3}$",
        long_name="Course Mode Particulate Matter",
        name="PM_COURSE",
    ),
    "CLf": DiagnosticSpec(
        variables=["ACLI", "ACLJ", "ACLK"],
        weights=[1.0, 1.0, 0.2],
        units=r"$\mu g m^{-3}$",
        long_name="Fine Mode particulate Cl",
        name="CLf",
    ),
    "CAf": DiagnosticSpec(
        variables=["ACAI", "ACAJ", "ASEACAT", "ASOIL", "ACORS"],
        weights=[
            1.0,
            1.0,
            0.2 * 32.0 / 1000.0,
            0.2 * 83.8 / 1000.0,
            0.2 * 56.2 / 1000.0,
        ],
        units=r"$\mu g m^{-3}$",
        long_name="Fine Mode particulate CA",
        name="CAf",
    ),
    "NAf": DiagnosticSpec(
        variables=["ANAI", "ANAJ", "ASEACAT", "ASOIL", "ACORS"],
        weights=[
            1.0,
            1.0,
            0.2 * 837.3 / 1000.0,
            0.2 * 62.6 / 1000.0,
            0.2 * 2.3 / 1000.0,
        ],
        units=r"$\mu g m^{-3}$",
        long_name="NAf",
        name="NAf",
    ),
    "SO4f": DiagnosticSpec(
        variables=["ASO4I", "ASO4J", "ASO4K"],
        weights=[1.0, 1.0, 0.2],
        units=r"$\mu g m^{-3}$",
        long_name="SO4f",
        name="SO4f",
    ),
    "NH4f": DiagnosticSpec(
        variables=["ANH4I", "ANH4J", "ANH4K"],
        weights=[1.0, 1.0, 0.2],
        units=r"$\mu g m^{-3}$",
        long_name="NH4f",
        name="NH4f",
    ),
    "NO3f": DiagnosticSpec(
        variables=["ANO3I", "ANO3J", "ANO3K"],
        weights=[1.0, 1.0, 0.2],
        units=r"$\mu g m^{-3}$",
        long_name="NO3f",
        name="NO3f",
    ),
    "NOy": DiagnosticSpec(
        variables=NOY_GAS,
        weights=None,
        units="ppbV",  # Assuming gas
        long_name="NOy",
        name="NOy",
    ),
    "NOx": DiagnosticSpec(
        variables=["NO", "NO2"],
        weights=None,
        units="ppbV",  # Assuming gas
        long_name="NOx",
        name="NOx",
    ),
}
