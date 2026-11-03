"""Project-specific exception hierarchy.

Keeping a small hierarchy makes it possible for callers (including the
dashboard) to distinguish configuration problems from data problems without
parsing error strings.
"""

from __future__ import annotations


class GeoAIError(Exception):
    """Base class for all project-specific errors."""


class ConfigurationError(GeoAIError):
    """Raised when configuration or environment values are invalid."""


class DataValidationError(GeoAIError):
    """Raised when user-supplied files, paths, or values fail validation."""


class RasterError(GeoAIError):
    """Raised for raster reading, writing, or processing failures."""


class VectorError(GeoAIError):
    """Raised for vector reading, writing, or processing failures."""


class ModelError(GeoAIError):
    """Raised for model training, loading, or prediction failures."""
