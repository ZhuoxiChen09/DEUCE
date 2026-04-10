"""Map live_scores normalized dict rows into domain MatchConfig / MatchState."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from deuce.domain.enums import MatchFormat, MatchStatus, ServerSide, SetType, Surface
from deuce.domain.models import GameState, MatchConfig, MatchState, Player, PointState, SetState, Tournament
from deuce.domain.scoring import ScoringEngine

_SET_GAMES_RE = re.compile(r"^(\d+)(?:\([^)]+\))?$")


def _stable_match_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("match_id") or "").strip()
    if explicit:
        return explicit
    players = row.get("players") or []
    blob = "|".join(
        [
            str(row.get("date") or ""),
            str(row.get("tournament") or ""),
            str(row.get("source") or ""),
        ]
        + [str((p or {}).get("name") or "") for p in players[:2]]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _tour_from_draw(draw: str) -> str:
    d = (draw or "").lower()
    if "women" in d or d == "wta":
        return "WTA"
    if "men" in d or d == "atp":
        return "ATP"
    return ""


def infer_server_side(row: dict[str, Any]) -> ServerSide:
    """Map provider serve hint to ServerSide (PLAYER_A = first row in `players`)."""
    raw = str(row.get("serve") or "").strip()
    if not raw:
        return ServerSide.PLAYER_A
    src = str(row.get("source") or "").lower()
    try:
        n = int(raw)
    except ValueError:
        return ServerSide.PLAYER_A
    if "atp" in src:
        return ServerSide.PLAYER_A if n == 0 else ServerSide.PLAYER_B
    return ServerSide.PLAYER_A if n == 1 else ServerSide.PLAYER_B


def _parse_set_games_token(tok: str) -> int:
    tok = str(tok).strip()
    m = _SET_GAMES_RE.match(tok)
    if m:
        return int(m.group(1))
    return int(tok)


def _is_completed_set_games(ga: int, gb: int) -> bool:
    if (ga, gb) in ((7, 6), (6, 7)):
        return True
    if ga == gb:
        return False
    if ga > gb:
        return ga >= 6 and (ga - gb) >= 2
    return gb >= 6 and (gb - ga) >= 2


def _partition_live_set_columns(
    pairs: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Split set game columns into finished sets vs the in-progress set (may be 0-0)."""
    completed: list[tuple[int, int]] = []
    for ga, gb in pairs:
        if _is_completed_set_games(ga, gb):
            completed.append((ga, gb))
        else:
            return completed, (ga, gb)
    return completed, (0, 0)


def _completed_match_set_tuples(tokens_a: list[str], tokens_b: list[str]) -> list[tuple[int, int]]:
    if len(tokens_a) != len(tokens_b):
        raise ValueError("Mismatched score token lengths")
    return [
        (_parse_set_games_token(tokens_a[i]), _parse_set_games_token(tokens_b[i]))
        for i in range(len(tokens_a))
    ]


def _points_to_server_frame(
    pa: str,
    pb: str,
    server: ServerSide,
) -> tuple[str, str]:
    if server is ServerSide.PLAYER_A:
        return (pa, pb)
    return (pb, pa)


def match_config_from_provider_dict(
    row: dict[str, Any],
    *,
    match_id: str | None = None,
    match_format: MatchFormat = MatchFormat.BEST_OF_3,
) -> MatchConfig | None:
    players = row.get("players") or []
    if len(players) < 2:
        return None
    mid = match_id or _stable_match_id(row)
    draw = str(row.get("draw") or "")
    t = Tournament(
        name=str(row.get("tournament") or ""),
        surface=Surface.HARD,
        tour=_tour_from_draw(draw),
    )
    p0 = players[0] or {}
    p1 = players[1] or {}
    return MatchConfig(
        match_id=mid,
        tournament=t,
        player_a=Player(
            player_id=str(p0.get("player_id") or p0.get("name") or "a"),
            name=str(p0.get("name") or "Player A"),
            country_code=str(p0.get("country_code") or ""),
        ),
        player_b=Player(
            player_id=str(p1.get("player_id") or p1.get("name") or "b"),
            name=str(p1.get("name") or "Player B"),
            country_code=str(p1.get("country_code") or ""),
        ),
        match_format=match_format,
        round_name=str(row.get("round") or ""),
        court=str(row.get("court") or ""),
    )


def match_state_from_provider_dict(row: dict[str, Any]) -> MatchState | None:
    """Best-effort bridge from a normalized live_scores row to domain MatchState."""
    cfg = match_config_from_provider_dict(row)
    if cfg is None:
        return None
    if row.get("is_scheduled"):
        return ScoringEngine.start_match(cfg)
    players = row.get("players") or []
    ta = [str(x) for x in (players[0] or {}).get("score_tokens") or []]
    tb = [str(x) for x in (players[1] or {}).get("score_tokens") or []]
    server = infer_server_side(row)
    is_live = bool(row.get("is_live"))
    is_completed = bool(row.get("is_completed"))
    if not ta and not tb:
        return ScoringEngine.start_match(cfg)
    if len(ta) != len(tb):
        return None

    if is_completed:
        set_scores = _completed_match_set_tuples(ta, tb)
        state = ScoringEngine.set_state_from_scores(cfg, set_scores, server=server)
        if not state.is_complete:
            return state.model_copy(update={"status": MatchStatus.COMPLETED})
        return state

    if is_live:
        if len(ta) == 1:
            completed_pairs: list[tuple[int, int]] = []
            cur_games = (0, 0)
            raw_points = (ta[0], tb[0])
        else:
            game_pairs = [
                (_parse_set_games_token(ta[i]), _parse_set_games_token(tb[i]))
                for i in range(len(ta) - 1)
            ]
            completed_pairs, cur_games = _partition_live_set_columns(game_pairs)
            raw_points = (ta[-1], tb[-1])
        base = ScoringEngine.set_state_from_scores(cfg, completed_pairs, server=server)
        cur_set_num = len(completed_pairs) + 1
        in_tb = cur_games[0] == 6 and cur_games[1] == 6
        sp, rp = _points_to_server_frame(raw_points[0], raw_points[1], server)
        game = GameState(
            game_number=cur_games[0] + cur_games[1] + 1,
            server=server,
            point_state=PointState(server_points=sp, returner_points=rp),
            set_type=SetType.TIEBREAK if in_tb else SetType.REGULAR,
        )
        cset = SetState(
            set_number=cur_set_num,
            games_a=cur_games[0],
            games_b=cur_games[1],
            is_tiebreak=in_tb,
        )
        return base.model_copy(
            update={
                "current_set": cset,
                "current_game": game,
                "server": server,
                "status": MatchStatus.LIVE,
            }
        )

    # Fallback: treat like completed reconstruction without status coercion
    set_scores_fb = _completed_match_set_tuples(ta, tb)
    return ScoringEngine.set_state_from_scores(cfg, set_scores_fb, server=server)
