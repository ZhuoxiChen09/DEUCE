"""Rolling / segmented research reports."""

from deuce.research.report import (
    ResearchReport,
    report_to_json_dict,
    research_report_from_csv,
    research_report_from_rows,
)

__all__ = [
    "ResearchReport",
    "report_to_json_dict",
    "research_report_from_csv",
    "research_report_from_rows",
]
