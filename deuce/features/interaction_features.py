"""Interaction feature name list (re-export for tooling / inspection)."""

from __future__ import annotations

# Mirrors ``model_features.INTERACTION_FEATURES`` — single list for imports from deuce.features.
INTERACTION_FEATURES = [
    "int_serve_surface_speed",
    "int_return_slow_surface",
    "int_pressure_round",
    "int_serve_altitude",
    "int_return_climate",
    "int_lefty_surface",
]
