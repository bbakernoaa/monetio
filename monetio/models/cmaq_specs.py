"""CMAQ Species and Diagnostic Variable Definitions

This file contains dictionaries defining CMAQ species and the formulas for
derived diagnostic variables. This centralized approach allows for easier
maintenance and extensibility.

"""

# Variable lists for different aerosol modes
accumulation = [
    "AALJ", "AALK1J", "AALK2J", "ABNZ1J", "ABNZ2J", "ABNZ3J", "ACAJ", "ACLJ", "AECJ",
    "AFEJ", "AISO1J", "AISO2J", "AISO3J", "AKJ", "AMGJ", "AMNJ", "ANAJ", "ANH4J",
    "ANO3J", "AOLGAJ", "AOLGBJ", "AORGCJ", "AOTHRJ", "APAH1J", "APAH2J", "APAH3J",
    "APNCOMJ", "APOCJ", "ASIJ", "ASO4J", "ASQTJ", "ATIJ", "ATOL1J", "ATOL2J",
    "ATOL3J", "ATRP1J", "ATRP2J", "AXYL1J", "AXYL2J", "AXYL3J", "AORGAJ",
    "AORGPAJ", "AORGBJ",
]
aitken = [
    "ACLI", "AECI", "ANAI", "ANH4I", "ANO3I", "AOTHRI", "APNCOMI", "APOCI", "ASO4I",
    "AORGAI", "AORGPAI", "AORGBI",
]
coarse = ["ACLK", "ACORS", "ANH4K", "ANO3K", "ASEACAT", "ASO4K", "ASOIL"]
noy_gas = [
    "NO", "NO2", "NO3", "N2O5", "HONO", "HNO3", "PAN", "PANX", "PNA", "NTR",
    "CRON", "CRN2", "CRNO", "CRPX", "OPAN",
]

# Dictionary defining derived diagnostic variables
CMAQ_SPECIES = {
    "PM25": {
        "vars": aitken + accumulation + coarse,
        "weights": [1.0] * (len(aitken) + len(accumulation)) + [0.2] * len(coarse),
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "PM2.5",
            "long_name": "PM2.5"
        },
        "alt_names": ["PM25_TOT"],
    },
    "PM10": {
        "vars": aitken + accumulation + coarse,
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "PM10",
            "long_name": "Particulate Matter < 10 microns",
        },
        "alt_names": ["PMC_TOT"],
    },
    "PM_COURSE": {
        "vars": coarse,
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "PM_COURSE",
            "long_name": "Course Mode Particulate Matter",
        }
    },
    "CLf": {
        "vars": ["ACLI", "ACLJ", "ACLK"],
        "weights": [1.0, 1.0, 0.2],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "CLf",
            "long_name": "Fine Mode particulate Cl"
        }
    },
    "NAf": {
        "vars": ["ANAI", "ANAJ", "ASEACAT", "ASOIL", "ACORS"],
        "weights": [1.0, 1.0, 0.2 * 837.3 / 1000.0, 0.2 * 62.6 / 1000.0, 0.2 * 2.3 / 1000.0],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "NAf",
            "long_name": "NAf"
        }
    },
    "CAf": {
        "vars": ["ACAI", "ACAJ", "ASEACAT", "ASOIL", "ACORS"],
        "weights": [1.0, 1.0, 0.2 * 32.0 / 1000.0, 0.2 * 83.8 / 1000.0, 0.2 * 56.2 / 1000.0],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "CAf",
            "long_name": "Fine Mode particulate CA"
        }
    },
    "SO4f": {
        "vars": ["ASO4I", "ASO4J", "ASO4K"],
        "weights": [1.0, 1.0, 0.2],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "SO4f",
            "long_name": "SO4f"
        }
    },
    "NH4f": {
        "vars": ["ANH4I", "ANH4J", "ANH4K"],
        "weights": [1.0, 1.0, 0.2],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "NH4f",
            "long_name": "NH4f"
        }
    },
    "NO3f": {
        "vars": ["ANO3I", "ANO3J", "ANO3K"],
        "weights": [1.0, 1.0, 0.2],
        "attrs": {
            "units": r"$\mu g m^{-3}$",
            "name": "NO3f",
            "long_name": "NO3f"
        }
    },
    "NOy": {
        "vars": noy_gas,
        "attrs": {"name": "NOy", "long_name": "NOy"}
    },
    "NOx": {
        "vars": ["NO", "NO2"],
        "attrs": {"name": "NOx", "long_name": "NOx"}
    }
}
