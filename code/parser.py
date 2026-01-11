from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests

# ----------------------------
# Config (edit if you want)
# ----------------------------

BASE_URL = "https://api.football-data.org/v4"
TOKEN_ENV = "FOOTBALL_DATA_API_TOKEN"

# Only this competition is exported (as requested)
COMPETITION_ID = int(os.getenv("AIA_COMPETITION_ID", "2014"))

# Whether to ask the API to include a "goals" array in matches responses.
# football-data.org supports this via the "X-Unfold-Goals: true" header.
UNFOLD_GOALS = (os.getenv("AIA_UNFOLD_GOALS", "true").strip().lower() in {"1", "true", "yes", "y"})

# Try seasons starting from this year; older seasons may be blocked by API plan (403)
MIN_YEAR = int(os.getenv("AIA_MIN_YEAR", "2020"))

MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("AIA_MIN_SECONDS_BETWEEN_CALLS", "8.0"))
MAX_RETRIES_429 = int(os.getenv("AIA_MAX_RETRIES_429", "6"))

OUT_DIR = Path(os.getenv("AIA_OUT_DIR", "data"))
CACHE_DIR = Path(os.getenv("AIA_CACHE_DIR", "data/_api_cache"))

OUT_MATCHES = Path(os.getenv("AIA_OUT_MATCHES", str(OUT_DIR / "matches.csv")))
OUT_STANDINGS = Path(os.getenv("AIA_OUT_STANDINGS", str(OUT_DIR / "standings.csv")))
OUT_TEAMS = Path(os.getenv("AIA_OUT_TEAMS", str(OUT_DIR / "teams.csv")))

OUT_SCORERS = Path(os.getenv("AIA_OUT_SCORERS", str(OUT_DIR / "scorers.csv")))
OUT_MATCH_GOALS = Path(os.getenv("AIA_OUT_MATCH_GOALS", str(OUT_DIR / "match_goals.csv")))
OUT_MATCH_BOOKINGS = Path(os.getenv("AIA_OUT_MATCH_BOOKINGS", str(OUT_DIR / "match_bookings.csv")))

SCORERS_LIMIT = int(os.getenv("AIA_SCORERS_LIMIT", "50"))

# Events export behavior:
# - Some subscriptions / endpoints may not include "goals"/"bookings" arrays in the season-wide
#   /competitions/{id}/matches response even with X-Unfold-Goals.
# - As a fallback, we can fetch matches per matchday (far fewer requests than per match).
EVENTS_MATCHDAY_FALLBACK = (os.getenv("AIA_EVENTS_MATCHDAY_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "y"})
MAX_MATCHDAY_FALLBACK = int(os.getenv("AIA_MAX_MATCHDAY", "38"))

# Expensive fallback: fetch /matches/{id} for each match (very slow with strict rate limits).
EVENTS_FETCH_MATCH_DETAILS = (os.getenv("AIA_EVENTS_FETCH_MATCH_DETAILS", "true").strip().lower() in {"1", "true", "yes", "y"})
EVENTS_MATCH_STATUS_FOR_DETAILS = os.getenv("AIA_EVENTS_MATCH_STATUS_FOR_DETAILS", "FINISHED").strip().upper()


# ----------------------------
# Small utils
# ----------------------------


def load_dotenv_simple(*, env_path: Path, override: bool = False) -> None:
    """Load key/value pairs from a .env file into os.environ (no deps)."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if override or k not in os.environ:
            os.environ[k] = v


def _s(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def season_years(competition_payload: dict, *, min_year: int) -> List[int]:
    years: set[int] = set()
    for s in competition_payload.get("seasons") or []:
        start = (s or {}).get("startDate")  # yyyy-mm-dd
        if isinstance(start, str) and len(start) >= 4 and start[:4].isdigit():
            y = int(start[:4])
            if y >= min_year:
                years.add(y)
    if not years:
        years = set(range(min_year, date.today().year + 1))
    return sorted(years)


# ----------------------------
# API client (rate-limited + cached)
# ----------------------------


@dataclass
class RateLimitConfig:
    min_seconds_between_calls: float = 8.0
    max_retries_429: int = 6


class FootballDataClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str = BASE_URL,
        cache_dir: Optional[Path] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        timeout_seconds: float = 60.0,
        unfold_goals: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limit = rate_limit
        self._last_call_ts = 0.0

        self.session = requests.Session()
        headers = {"X-Auth-Token": token}
        if unfold_goals:
            headers["X-Unfold-Goals"] = "true"
        self.session.headers.update(headers)

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, path: str, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return path
        items = sorted((str(k), str(v)) for k, v in params.items())
        return f"{path}?{urlencode(items)}"

    def _cache_paths(self, path: str, params: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Path]]:
        key = self._cache_key(path, params)
        if not self.cache_dir:
            return key, None
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return key, (self.cache_dir / f"{h}.json")

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        key, cache_path = self._cache_paths(path, params)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if self.rate_limit:
            now = time.time()
            wait = self.rate_limit.min_seconds_between_calls - (now - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)

        url = f"{self.base_url}{path}"

        for attempt in range((self.rate_limit.max_retries_429 if self.rate_limit else 0) + 1):
            r = self.session.get(url, params=params, timeout=self.timeout_seconds)
            if r.status_code == 429 and self.rate_limit:
                retry_after = r.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after else (self.rate_limit.min_seconds_between_calls * (attempt + 1))
                except Exception:
                    sleep_s = self.rate_limit.min_seconds_between_calls * (attempt + 1)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            data = r.json()
            if cache_path:
                cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if self.rate_limit:
                self._last_call_ts = time.time()
            return data

        raise RuntimeError(f"Failed after retries: {key}")


# ----------------------------
# Flatteners + exporters
# ----------------------------


def flatten_match(match: Dict[str, Any], *, competition: Dict[str, Any]) -> Dict[str, Any]:
    score = match.get("score") or {}
    odds = match.get("odds") or {}
    return {
        "competition_id": _s(competition, "id"),
        "competition_code": _s(competition, "code"),
        "competition_name": _s(competition, "name"),
        "area_name": _s(competition, "area", "name"),
        "match_id": match.get("id"),
        "utcDate": match.get("utcDate"),
        "status": match.get("status"),
        "minute": match.get("minute"),
        "injuryTime": match.get("injuryTime"),
        "attendance": match.get("attendance"),
        "venue": match.get("venue"),
        "matchday": match.get("matchday"),
        "stage": match.get("stage"),
        "group": match.get("group"),
        "lastUpdated": match.get("lastUpdated"),
        "season_startDate": _s(match, "season", "startDate"),
        "season_endDate": _s(match, "season", "endDate"),
        "homeTeam_id": _s(match, "homeTeam", "id"),
        "homeTeam_name": _s(match, "homeTeam", "name"),
        "homeTeam_shortName": _s(match, "homeTeam", "shortName"),
        "homeTeam_tla": _s(match, "homeTeam", "tla"),
        "homeTeam_crest": _s(match, "homeTeam", "crest"),
        "awayTeam_id": _s(match, "awayTeam", "id"),
        "awayTeam_name": _s(match, "awayTeam", "name"),
        "awayTeam_shortName": _s(match, "awayTeam", "shortName"),
        "awayTeam_tla": _s(match, "awayTeam", "tla"),
        "awayTeam_crest": _s(match, "awayTeam", "crest"),
        "score_winner": score.get("winner"),
        "score_duration": score.get("duration"),
        "score_fullTime_home": _s(score, "fullTime", "home"),
        "score_fullTime_away": _s(score, "fullTime", "away"),
        "score_halfTime_home": _s(score, "halfTime", "home"),
        "score_halfTime_away": _s(score, "halfTime", "away"),
        "score_regularTime_home": _s(score, "regularTime", "home"),
        "score_regularTime_away": _s(score, "regularTime", "away"),
        "score_extraTime_home": _s(score, "extraTime", "home"),
        "score_extraTime_away": _s(score, "extraTime", "away"),
        "score_penalties_home": _s(score, "penalties", "home"),
        "score_penalties_away": _s(score, "penalties", "away"),
        "odds_homeWin": odds.get("homeWin"),
        "odds_draw": odds.get("draw"),
        "odds_awayWin": odds.get("awayWin"),
        "referees_json": json.dumps(match.get("referees") or [], ensure_ascii=False),
        "goals_count": len(match.get("goals") or []),
        "bookings_count": len(match.get("bookings") or []),
        "raw_json": json.dumps(match, ensure_ascii=False, separators=(",", ":")),
    }


def export_matches(
    *,
    client: FootballDataClient,
    competition_id: int,
    seasons: Iterable[int],
    out_csv: Path,
) -> int:
    competition = client.get(f"/competitions/{competition_id}")
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        try:
            payload = client.get(f"/competitions/{competition_id}/matches", params={"season": y})
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise
        matches = payload.get("matches") or []
        rows.extend(flatten_match(m, competition=competition) for m in matches)
    _write_csv(out_csv, rows)
    return len(rows)


def flatten_match_goal(*, competition: Dict[str, Any], match: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "competition_id": _s(competition, "id"),
        "competition_code": _s(competition, "code"),
        "competition_name": _s(competition, "name"),
        "match_id": match.get("id"),
        "utcDate": match.get("utcDate"),
        "season_startDate": _s(match, "season", "startDate"),
        "season_endDate": _s(match, "season", "endDate"),
        "matchday": match.get("matchday"),
        "stage": match.get("stage"),
        "group": match.get("group"),
        "minute": goal.get("minute"),
        "injuryTime": goal.get("injuryTime"),
        "type": goal.get("type"),
        "team_id": _s(goal, "team", "id"),
        "team_name": _s(goal, "team", "name"),
        "scorer_id": _s(goal, "scorer", "id"),
        "scorer_name": _s(goal, "scorer", "name"),
        "assist_id": _s(goal, "assist", "id"),
        "assist_name": _s(goal, "assist", "name"),
        "score_home": _s(goal, "score", "home"),
        "score_away": _s(goal, "score", "away"),
        "raw_goal_json": json.dumps(goal, ensure_ascii=False, separators=(",", ":")),
    }


def flatten_match_booking(*, competition: Dict[str, Any], match: Dict[str, Any], booking: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "competition_id": _s(competition, "id"),
        "competition_code": _s(competition, "code"),
        "competition_name": _s(competition, "name"),
        "match_id": match.get("id"),
        "utcDate": match.get("utcDate"),
        "season_startDate": _s(match, "season", "startDate"),
        "season_endDate": _s(match, "season", "endDate"),
        "matchday": match.get("matchday"),
        "stage": match.get("stage"),
        "group": match.get("group"),
        "minute": booking.get("minute"),
        "team_id": _s(booking, "team", "id"),
        "team_name": _s(booking, "team", "name"),
        "player_id": _s(booking, "player", "id"),
        "player_name": _s(booking, "player", "name"),
        "card": booking.get("card"),
        "raw_booking_json": json.dumps(booking, ensure_ascii=False, separators=(",", ":")),
    }


def export_match_events(
    *,
    client: FootballDataClient,
    competition_id: int,
    seasons: Iterable[int],
    out_goals_csv: Path,
    out_bookings_csv: Path,
) -> Tuple[int, int]:
    """
    Exports match-level events from /competitions/{id}/matches.
    Goals are only present if the API returns a "goals" array (see X-Unfold-Goals).
    """
    competition = client.get(f"/competitions/{competition_id}")
    goal_rows: List[Dict[str, Any]] = []
    booking_rows: List[Dict[str, Any]] = []

    for y in seasons:
        try:
            payload = client.get(f"/competitions/{competition_id}/matches", params={"season": y})
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise

        matches = payload.get("matches") or []

        season_goal_before = len(goal_rows)
        season_booking_before = len(booking_rows)

        def _ingest_match(m: Dict[str, Any]) -> None:
            nonlocal goal_rows, booking_rows
            for g in (m.get("goals") or []):
                goal_rows.append(flatten_match_goal(competition=competition, match=m, goal=g))
            for b in (m.get("bookings") or []):
                booking_rows.append(flatten_match_booking(competition=competition, match=m, booking=b))

        for m in matches:
            _ingest_match(m)

        # If season-wide response contains no events, try matchday queries (cheap-ish fallback).
        season_goal_after = len(goal_rows)
        season_booking_after = len(booking_rows)
        if (
            EVENTS_MATCHDAY_FALLBACK
            and season_goal_after == season_goal_before
            and season_booking_after == season_booking_before
        ):
            max_md = max((md for md in (m.get("matchday") for m in matches) if isinstance(md, int)), default=MAX_MATCHDAY_FALLBACK)
            if max_md <= 0:
                max_md = MAX_MATCHDAY_FALLBACK
            for md in range(1, max_md + 1):
                try:
                    md_payload = client.get(
                        f"/competitions/{competition_id}/matches",
                        params={"season": y, "matchday": md},
                    )
                except requests.HTTPError as e:
                    resp = getattr(e, "response", None)
                    if getattr(resp, "status_code", None) == 403:
                        break
                    raise

                for m in (md_payload.get("matches") or []):
                    _ingest_match(m)

        # Optional last resort: fetch per-match details (very slow; off by default).
        if EVENTS_FETCH_MATCH_DETAILS:
            for m in matches:
                mid = m.get("id")
                if not isinstance(mid, int):
                    continue
                if EVENTS_MATCH_STATUS_FOR_DETAILS and (m.get("status") != EVENTS_MATCH_STATUS_FOR_DETAILS):
                    continue
                # Only fetch if we didn't already get any events on this match object.
                if (m.get("goals") or m.get("bookings")):
                    continue
                try:
                    detail = client.get(f"/matches/{mid}")
                except requests.HTTPError as e:
                    resp = getattr(e, "response", None)
                    if getattr(resp, "status_code", None) == 403:
                        continue
                    raise
                _ingest_match(detail)

    _write_csv(out_goals_csv, goal_rows)
    _write_csv(out_bookings_csv, booking_rows)
    return (len(goal_rows), len(booking_rows))


def flatten_scorer(
    *,
    competition: Dict[str, Any],
    season: Dict[str, Any],
    scorer: Dict[str, Any],
) -> Dict[str, Any]:
    player = scorer.get("player") or {}
    team = scorer.get("team") or {}
    return {
        "competition_id": competition.get("id"),
        "competition_code": competition.get("code"),
        "competition_name": competition.get("name"),
        "season_id": season.get("id"),
        "season_startDate": season.get("startDate"),
        "season_endDate": season.get("endDate"),
        "player_id": player.get("id"),
        "player_name": player.get("name"),
        "player_firstName": player.get("firstName"),
        "player_lastName": player.get("lastName"),
        "player_dateOfBirth": player.get("dateOfBirth"),
        "player_nationality": player.get("nationality"),
        "player_position": player.get("position"),
        "player_shirtNumber": player.get("shirtNumber"),
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "team_shortName": team.get("shortName"),
        "team_tla": team.get("tla"),
        "team_crest": team.get("crest"),
        "goals": scorer.get("goals"),
        "assists": scorer.get("assists"),
        "penalties": scorer.get("penalties"),
        "raw_player_json": json.dumps(player, ensure_ascii=False, separators=(",", ":")),
        "raw_team_json": json.dumps(team, ensure_ascii=False, separators=(",", ":")),
        "raw_scorer_json": json.dumps(scorer, ensure_ascii=False, separators=(",", ":")),
    }


def export_scorers(
    *,
    client: FootballDataClient,
    competition_id: int,
    seasons: Iterable[int],
    out_csv: Path,
    limit: int = 50,
) -> int:
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        try:
            payload = client.get(f"/competitions/{competition_id}/scorers", params={"season": y, "limit": limit})
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise

        competition = payload.get("competition") or {"id": competition_id}
        season_obj = payload.get("season") or {"startDate": f"{y}-01-01"}
        for s in (payload.get("scorers") or []):
            rows.append(flatten_scorer(competition=competition, season=season_obj, scorer=s))

    _write_csv(out_csv, rows)
    return len(rows)


def flatten_standings_table_row(
    *,
    competition: Dict[str, Any],
    season: Dict[str, Any],
    standing: Dict[str, Any],
    row: Dict[str, Any],
    matchday: Optional[int],
) -> Dict[str, Any]:
    team = row.get("team") or {}
    return {
        "competition_id": competition.get("id"),
        "competition_code": competition.get("code"),
        "competition_name": competition.get("name"),
        "area_id": _s(competition, "area", "id"),
        "area_name": _s(competition, "area", "name"),
        "season_id": season.get("id"),
        "season_startDate": season.get("startDate"),
        "season_endDate": season.get("endDate"),
        "season_currentMatchday": season.get("currentMatchday"),
        "matchday": matchday,
        "standing_stage": standing.get("stage"),
        "standing_type": standing.get("type"),
        "position": row.get("position"),
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "team_shortName": team.get("shortName"),
        "team_tla": team.get("tla"),
        "team_crest": team.get("crest"),
        "playedGames": row.get("playedGames"),
        "form": row.get("form"),
        "won": row.get("won"),
        "draw": row.get("draw"),
        "lost": row.get("lost"),
        "points": row.get("points"),
        "goalsFor": row.get("goalsFor"),
        "goalsAgainst": row.get("goalsAgainst"),
        "goalDifference": row.get("goalDifference"),
        "raw_team_json": json.dumps(team, ensure_ascii=False, separators=(",", ":")),
        "raw_row_json": json.dumps(row, ensure_ascii=False, separators=(",", ":")),
    }


def export_standings(
    *,
    client: FootballDataClient,
    competition_id: int,
    seasons: Iterable[int],
    out_csv: Path,
    table_type: str = "TOTAL",
) -> int:
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        # First: get currentMatchday for that season. Then: ask standings at that matchday.
        try:
            payload = client.get(f"/competitions/{competition_id}/standings", params={"season": y})
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise

        season_obj = payload.get("season") or {}
        matchday = season_obj.get("currentMatchday")
        if isinstance(matchday, int):
            try:
                payload = client.get(
                    f"/competitions/{competition_id}/standings",
                    params={"season": y, "matchday": matchday},
                )
            except requests.HTTPError as e:
                resp = getattr(e, "response", None)
                if getattr(resp, "status_code", None) == 403:
                    continue
                raise

        competition = payload.get("competition") or {"id": competition_id}
        season_obj = payload.get("season") or season_obj
        standings = payload.get("standings") or []

        for st in standings:
            if table_type and (st.get("type") != table_type):
                continue
            for r in (st.get("table") or []):
                rows.append(
                    flatten_standings_table_row(
                        competition=competition,
                        season=season_obj,
                        standing=st,
                        row=r,
                        matchday=matchday if isinstance(matchday, int) else None,
                    )
                )

    _write_csv(out_csv, rows)
    return len(rows)


def flatten_team(*, competition: Dict[str, Any], season: Dict[str, Any], team: Dict[str, Any]) -> Dict[str, Any]:
    area = team.get("area") or {}
    coach = team.get("coach") or {}
    contract = coach.get("contract") or {}
    return {
        "competition_id": competition.get("id"),
        "competition_code": competition.get("code"),
        "competition_name": competition.get("name"),
        "season_id": season.get("id"),
        "season_startDate": season.get("startDate"),
        "season_endDate": season.get("endDate"),
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "team_shortName": team.get("shortName"),
        "team_tla": team.get("tla"),
        "team_crest": team.get("crest"),
        "team_founded": team.get("founded"),
        "team_venue": team.get("venue"),
        "team_website": team.get("website"),
        "team_address": team.get("address"),
        "team_clubColors": team.get("clubColors"),
        "area_id": area.get("id"),
        "area_name": area.get("name"),
        "area_code": area.get("code"),
        "coach_id": coach.get("id"),
        "coach_name": coach.get("name"),
        "coach_nationality": coach.get("nationality"),
        "coach_contract_start": contract.get("start"),
        "coach_contract_until": contract.get("until"),
        "marketValue": team.get("marketValue"),
        "runningCompetitions_json": json.dumps(team.get("runningCompetitions") or [], ensure_ascii=False),
        "raw_json": json.dumps(team, ensure_ascii=False, separators=(",", ":")),
    }


def export_teams(
    *,
    client: FootballDataClient,
    competition_id: int,
    seasons: Iterable[int],
    out_csv: Path,
) -> int:
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        try:
            payload = client.get(f"/competitions/{competition_id}/teams", params={"season": y})
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) == 403:
                continue
            raise
        competition = payload.get("competition") or {"id": competition_id}
        season_obj = payload.get("season") or {"startDate": f"{y}-01-01"}
        teams = payload.get("teams") or []
        rows.extend(flatten_team(competition=competition, season=season_obj, team=t) for t in teams)
    _write_csv(out_csv, rows)
    return len(rows)


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    # repo root is one level above /code
    root = Path(__file__).resolve().parent.parent
    load_dotenv_simple(env_path=(root / ".env"))

    token = (os.getenv(TOKEN_ENV) or "").strip()
    if not token:
        raise SystemExit(f"Missing {TOKEN_ENV}. Put it in .env or export it.")

    client = FootballDataClient(
        token=token,
        cache_dir=(root / CACHE_DIR),
        rate_limit=RateLimitConfig(
            min_seconds_between_calls=MIN_SECONDS_BETWEEN_CALLS,
            max_retries_429=MAX_RETRIES_429,
        ),
        unfold_goals=UNFOLD_GOALS,
    )

    competition = client.get(f"/competitions/{COMPETITION_ID}")
    years = season_years(competition, min_year=MIN_YEAR)

    matches_path = (root / OUT_MATCHES).resolve()
    standings_path = (root / OUT_STANDINGS).resolve()
    teams_path = (root / OUT_TEAMS).resolve()
    scorers_path = (root / OUT_SCORERS).resolve()
    goals_path = (root / OUT_MATCH_GOALS).resolve()
    bookings_path = (root / OUT_MATCH_BOOKINGS).resolve()

    match_rows = export_matches(
        client=client,
        competition_id=COMPETITION_ID,
        seasons=years,
        out_csv=matches_path,
    )
    print(f"Wrote {match_rows} match rows -> {matches_path}")

    goals_rows, bookings_rows = export_match_events(
        client=client,
        competition_id=COMPETITION_ID,
        seasons=years,
        out_goals_csv=goals_path,
        out_bookings_csv=bookings_path,
    )
    print(f"Wrote {goals_rows} goal rows -> {goals_path}")
    print(f"Wrote {bookings_rows} booking rows -> {bookings_path}")

    standings_rows = export_standings(
        client=client,
        competition_id=COMPETITION_ID,
        seasons=years,
        out_csv=standings_path,
    )
    print(f"Wrote {standings_rows} standings rows -> {standings_path}")

    team_rows = export_teams(
        client=client,
        competition_id=COMPETITION_ID,
        seasons=years,
        out_csv=teams_path,
    )
    print(f"Wrote {team_rows} team rows -> {teams_path}")

    scorers_rows = export_scorers(
        client=client,
        competition_id=COMPETITION_ID,
        seasons=years,
        out_csv=scorers_path,
        limit=SCORERS_LIMIT,
    )
    print(f"Wrote {scorers_rows} scorer rows -> {scorers_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

