"""Compatibility exports; the ordinary entry uses the feature architecture."""
from .adapters.action_io import (
    channels,
    _bezier_points,
    _split_points,
    _safe_segments,
    curve_boundary,
    _extrapolation_slope,
    _select_source_slot,
    _channelbag,
    _validate_curve,
    _key_snapshot,
    _write_curve,
    bake_action,
    defaultdict_list,
    trim_terminal_bake_guard,
    _DEFAULT_END_GUARD,
)
