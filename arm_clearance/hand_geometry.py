"""Compatibility exports; the ordinary entry uses the feature architecture."""
from .features.collision.hand_geometry import (
    _descendants,
    _palm_labels,
    _palm_surface_labels,
    _skin,
    smooth_radius_upper_envelope,
    HandGeometry,
    build_hand_geometry,
    _rotation_vectors,
    refit_corrected_hand_envelopes,
)
