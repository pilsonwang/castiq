"""US state reference data: name cleaning, centroids, and a tile-grid layout for maps.

The company file is free-text. "Tx Texas", "D.C.", "York" and "District Of Columbia" all appear.
`normalize_state` maps every value it can to one canonical name and returns None for the rest, so the
data-quality report can count what was not mapped.
"""

from __future__ import annotations

import difflib
import re

# name: (abbr, lat, lon, tile_row, tile_col)
STATES: dict[str, tuple[str, float, float, int, int]] = {
    "Alabama": ("AL", 32.8, -86.8, 6, 6),
    "Alaska": ("AK", 64.7, -152.0, 0, 0),
    "Arizona": ("AZ", 34.3, -111.7, 5, 1),
    "Arkansas": ("AR", 34.9, -92.4, 5, 4),
    "California": ("CA", 37.2, -119.4, 4, 0),
    "Colorado": ("CO", 39.0, -105.5, 4, 2),
    "Connecticut": ("CT", 41.6, -72.7, 3, 9),
    "Delaware": ("DE", 39.0, -75.5, 4, 9),
    "District of Columbia": ("DC", 38.9, -77.0, 5, 8),
    "Florida": ("FL", 28.6, -82.4, 7, 8),
    "Georgia": ("GA", 32.7, -83.4, 6, 7),
    "Hawaii": ("HI", 20.8, -156.3, 7, 0),
    "Idaho": ("ID", 44.4, -114.6, 2, 1),
    "Illinois": ("IL", 40.0, -89.2, 2, 5),
    "Indiana": ("IN", 39.9, -86.3, 3, 5),
    "Iowa": ("IA", 42.1, -93.5, 3, 4),
    "Kansas": ("KS", 38.5, -98.4, 5, 3),
    "Kentucky": ("KY", 37.5, -85.3, 4, 5),
    "Louisiana": ("LA", 31.1, -92.0, 6, 4),
    "Maine": ("ME", 45.4, -69.2, 0, 10),
    "Maryland": ("MD", 39.0, -76.8, 4, 8),
    "Massachusetts": ("MA", 42.3, -71.8, 2, 9),
    "Michigan": ("MI", 44.3, -85.4, 2, 6),
    "Minnesota": ("MN", 46.3, -94.3, 2, 4),
    "Mississippi": ("MS", 32.7, -89.7, 6, 5),
    "Missouri": ("MO", 38.4, -92.5, 4, 4),
    "Montana": ("MT", 47.0, -109.6, 2, 2),
    "Nebraska": ("NE", 41.5, -99.8, 4, 3),
    "Nevada": ("NV", 39.3, -116.6, 3, 1),
    "New Hampshire": ("NH", 43.7, -71.6, 1, 10),
    "New Jersey": ("NJ", 40.2, -74.7, 3, 8),
    "New Mexico": ("NM", 34.4, -106.1, 5, 2),
    "New York": ("NY", 42.9, -75.5, 2, 8),
    "North Carolina": ("NC", 35.6, -79.4, 5, 6),
    "North Dakota": ("ND", 47.5, -100.5, 2, 3),
    "Ohio": ("OH", 40.3, -82.8, 3, 6),
    "Oklahoma": ("OK", 35.6, -97.5, 6, 3),
    "Oregon": ("OR", 43.9, -120.6, 3, 0),
    "Pennsylvania": ("PA", 40.9, -77.8, 3, 7),
    "Rhode Island": ("RI", 41.7, -71.5, 3, 10),
    "South Carolina": ("SC", 33.9, -80.9, 5, 7),
    "South Dakota": ("SD", 44.4, -100.2, 3, 3),
    "Tennessee": ("TN", 35.9, -86.4, 5, 5),
    "Texas": ("TX", 31.5, -99.3, 7, 3),
    "Utah": ("UT", 39.3, -111.7, 4, 1),
    "Vermont": ("VT", 44.1, -72.7, 1, 9),
    "Virginia": ("VA", 37.5, -78.9, 4, 7),
    "Washington": ("WA", 47.4, -120.5, 2, 0),
    "West Virginia": ("WV", 38.6, -80.6, 4, 6),
    "Wisconsin": ("WI", 44.6, -89.9, 1, 5),
    "Wyoming": ("WY", 43.0, -107.6, 3, 2),
}

# Land borders between states (lower 48 + DC). Alaska and Hawaii have no land border; for territory
# design they are linked to the nearest mainland state (AK-WA, HI-CA) so they join a West territory.
_ADJ_ABBR: dict[str, str] = {
    "AL": "FL GA MS TN", "AZ": "CA CO NM NV UT", "AR": "LA MO MS OK TN TX", "CA": "AZ NV OR HI",
    "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI", "DE": "MD NJ PA", "DC": "MD VA", "FL": "AL GA",
    "GA": "AL FL NC SC TN", "ID": "MT NV OR UT WA WY", "IL": "IN IA KY MO WI", "IN": "IL KY MI OH",
    "IA": "IL MN MO NE SD WI", "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV", "LA": "AR MS TX",
    "ME": "NH", "MD": "DE PA VA WV DC", "MA": "CT NH NY RI VT", "MI": "IN OH WI", "MN": "IA ND SD WI",
    "MS": "AL AR LA TN", "MO": "AR IL IA KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY",
    "NV": "AZ CA ID OR UT", "NH": "ME MA VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX UT",
    "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA", "ND": "MN MT SD", "OH": "IN KY MI PA WV",
    "OK": "AR CO KS MO NM TX", "OR": "CA ID NV WA", "PA": "DE MD NJ NY OH WV", "RI": "CT MA",
    "SC": "GA NC", "SD": "IA MN MT ND NE WY", "TN": "AL AR GA KY MS MO NC VA", "TX": "AR LA NM OK",
    "UT": "AZ CO ID NV NM WY", "VT": "MA NH NY", "VA": "KY MD NC TN WV DC", "WA": "ID OR AK",
    "WV": "KY MD OH PA VA", "WI": "IL IA MI MN", "WY": "CO ID MT NE SD UT", "AK": "WA", "HI": "CA",
}

_ABBR_TO_NAME = {v[0]: k for k, v in STATES.items()}

ADJACENCY: dict[str, set[str]] = {
    _ABBR_TO_NAME[a]: {_ABBR_TO_NAME[b] for b in nbrs.split()} for a, nbrs in _ADJ_ABBR.items()
}
_LOWER_TO_NAME = {k.lower(): k for k in STATES}
_ALIASES = {
    "d.c.": "District of Columbia",
    "dc": "District of Columbia",
    "washington dc": "District of Columbia",
    "washington d.c.": "District of Columbia",
    "district of columbia": "District of Columbia",
    "york": "New York",  # the source drops "New " for some records
}


def normalize_state(raw: object) -> str | None:
    """Return the canonical US state name, or None when the value cannot be mapped safely."""
    if raw is None or not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.strip())
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    low = text.lower()
    if low in _LOWER_TO_NAME:
        return _LOWER_TO_NAME[low]
    if low in _ALIASES:
        return _ALIASES[low]
    if text.upper() in _ABBR_TO_NAME:
        return _ABBR_TO_NAME[text.upper()]
    # "Tx Texas", "Texas State", "State of Ohio": accept when exactly one state name is inside.
    hits = {name for lname, name in _LOWER_TO_NAME.items() if re.search(rf"\b{re.escape(lname)}\b", low)}
    # "West Virginia" contains "Virginia", "New York" etc. - keep the longest match only.
    if hits:
        longest = max(hits, key=len)
        hits = {h for h in hits if h == longest or h.lower() not in longest.lower()}
        if len(hits) == 1:
            return hits.pop()
    # Typos such as "Flrida" or "Pensylvania": accept only a very close match.
    close = difflib.get_close_matches(low, list(_LOWER_TO_NAME), n=1, cutoff=0.88)
    if close:
        return _LOWER_TO_NAME[close[0]]
    return None


def abbr(state: str) -> str:
    return STATES[state][0]


def centroid(state: str) -> tuple[float, float]:
    _, lat, lon, _, _ = STATES[state]
    return lat, lon


def tile(state: str) -> tuple[int, int]:
    _, _, _, row, col = STATES[state]
    return row, col
