"""
Copyright (c) 2023 Aryaman Jeendgar. All rights reserved.

UHI-serialization: An implementation of the UHI serialization format
"""


from __future__ import annotations
from .hdf5_serialization import write_hdf5_schema, read_hdf5_schema

__version__ = "0.1.0"

__all__ = ("__version__",)
