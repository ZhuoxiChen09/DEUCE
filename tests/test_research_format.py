"""research_format: safe coercion and scale detection for walk-forward metrics."""

from deuce.presentation.research_format import (
    as_float,
    format_avg_edge,
    format_hit_rate,
    format_roi_or_yield,
)


def test_as_float_coercion():
    assert as_float(None) is None
    assert as_float("") is None
    assert as_float(True) is None
    assert as_float("0.12") == 0.12
    assert as_float(12.0) == 12.0


def test_format_roi_proportion_vs_percent():
    s, note = format_roi_or_yield(0.12)
    assert "12.00%" in s
    assert "proportion" in note
    s2, note2 = format_roi_or_yield(12.0)
    assert "12.00%" in s2
    assert "0–100" in note2


def test_format_hit_rate_proportion_vs_percent():
    s, n = format_hit_rate(0.512)
    assert "51.2%" in s
    assert "proportion" in n
    s2, n2 = format_hit_rate(51.2)
    assert "51.2%" in s2
    assert "0–100" in n2


def test_format_avg_edge():
    s, n = format_avg_edge(0.03)
    assert "3.00 pp" in s
    s2, n2 = format_avg_edge(3.0)
    assert "3.00 pp" in s2
