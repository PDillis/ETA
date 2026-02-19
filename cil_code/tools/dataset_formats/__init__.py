"""
Dataset format adapters for configurable dataset creation.

This package provides adapters for different dataset formats (Bench2Drive, Play2Drive, etc.)
to enable the generic dataset creation pipeline to work with different directory structures,
annotation formats, and action encodings.
"""

from .base import DatasetFormatAdapter
from .bench2drive import Bench2DriveAdapter
from .play2drive import Play2DriveAdapter

__all__ = [
    'DatasetFormatAdapter',
    'Bench2DriveAdapter',
    'Play2DriveAdapter',
]
