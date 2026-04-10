import pytest

from deuce.edge.signals import compute_edge


def test_edge_symmetry():
    s = compute_edge(0.6, 0.5, market_prob_b=0.5, use_fair=True)
    assert s.edge_a == pytest.approx(-s.edge_b, abs=1e-6)
