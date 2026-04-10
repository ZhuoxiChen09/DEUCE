from deuce.name_utils import match_player, normalize_player_name


def test_normalize():
    assert "carlos" in normalize_player_name("Carlos Alcaraz").lower()


def test_match_player():
    cands = ["Jannik Sinner", "Taylor Fritz"]
    assert match_player("sinner", cands) == "Jannik Sinner"
