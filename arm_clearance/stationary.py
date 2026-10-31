"""Compatibility exports; the ordinary entry uses the feature architecture."""
from .features.stationary import (
    _runs,
    stationary_spans,
    StationaryControls,
    transition_basis,
    MIN_HOLD_FRAMES,
    ROTATION_CHORD_TOLERANCE,
    LOCATION_TOLERANCE,
)
