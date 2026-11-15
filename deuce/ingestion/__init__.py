"""Pluggable historical data sources and import quality checks."""

from deuce.ingestion.adapters import (
    DataSourceAdapter,
    LocalCsvMatchAdapter,
    get_adapter,
    register_adapter,
)

__all__ = [
    "DataSourceAdapter",
    "LocalCsvMatchAdapter",
    "get_adapter",
    "register_adapter",
]
