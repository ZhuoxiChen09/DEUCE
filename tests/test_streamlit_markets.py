import html as html_lib

from deuce.presentation.live_scoreboard_utils import live_score_source_label


def test_live_score_source_label_maps_atp_feed():
    assert live_score_source_label("atp-official-public") == "ATP Tour live feed"


def test_live_score_source_label_can_be_used_as_footer_fallback():
    footer = html_lib.escape(live_score_source_label("atp-official-public"))
    assert footer == "ATP Tour live feed"
