"""Compatibility wrapper for imports that still use the old root-level module path."""

from deuce.player_db import (
    DEFAULT_DB_PATH,
    PRESSURE_MAP,
    RANKINGS_MAP,
    RETURN_MAP,
    SERVE_MAP,
    STAT_KEYS,
    TDI_MAP,
    WINLOSS_MAP,
    fix_player_names_and_save,
    get_player_stats,
    list_players,
    load_db,
    save_db,
    update_db_from_atp_datasets,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "PRESSURE_MAP",
    "RANKINGS_MAP",
    "RETURN_MAP",
    "SERVE_MAP",
    "STAT_KEYS",
    "TDI_MAP",
    "WINLOSS_MAP",
    "fix_player_names_and_save",
    "get_player_stats",
    "list_players",
    "load_db",
    "save_db",
    "update_db_from_atp_datasets",
]
