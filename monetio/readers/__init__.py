"""
Readers module initialization.

This module provides a unified interface for reading various data formats
and sources through a plugin-based reader architecture.
"""

# Import all reader classes to register them in the READER_REGISTRY
from .base import READER_REGISTRY
from .nesdis_avhrr_aot_aws_gridded import NESDISAVHRRAOTAWSGriddedReader

# Import specific readers
from .nesdis_eps_viirs_aod_nrt import NESDISEPSVIIRSAODNRTReader
from .nesdis_viirs_aod_aws_gridded import NESDISVIIRSAODAWSGriddedReader
from .nesdis_viirs_ndvi_aws_gridded import NESDISVIIRSNDVIAWSGriddedReader

__all__ = [
    "READER_REGISTRY",
    "NESDISEPSVIIRSAODNRTReader",
    "NESDISVIIRSAODAWSGriddedReader",
    "NESDISVIIRSNDVIAWSGriddedReader",
    "NESDISAVHRRAOTAWSGriddedReader",
]
