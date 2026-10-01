"""Feature engineering: text fields, structured sets, and rating buckets."""

from __future__ import annotations

import pandas as pd


def _pipe_to_space(value: str) -> str:
    return value.replace("|", " ").strip()


def parse_pipe_list(value: str) -> set[str]:
    return {x.strip() for x in str(value).split("|") if x.strip()}


def parse_genres(genres: str) -> set[str]:
    return parse_pipe_list(genres)


def rating_bucket(rating: float) -> str:
    """Discretize rating so similar scores share a TF-IDF token."""
    if rating >= 8.5:
        return "rating_tier_excellent"
    if rating >= 7.5:
        return "rating_tier_good"
    if rating >= 6.5:
        return "rating_tier_average"
    return "rating_tier_low"


def build_overview_document(row: pd.Series, *, keyword_repeat: int = 4) -> str:
    overview = str(row["overview"]).strip()
    keywords = _pipe_to_space(str(row.get("keywords", "")))
    keyword_block = " ".join([keywords] * keyword_repeat) if keywords else ""
    return " ".join(part for part in (overview, keyword_block) if part)


def build_metadata_document(row: pd.Series, *, genre_repeat: int = 3) -> str:
    """Genres are repeated so TF-IDF emphasizes overlap without drowning plot text."""
    genres = _pipe_to_space(str(row["genres"]))
    genre_block = " ".join([genres] * genre_repeat)
    cast = _pipe_to_space(str(row["cast"]))
    director = str(row.get("director", "")).strip()
    rating_token = rating_bucket(float(row["rating"]))
    parts = [genre_block, cast, rating_token]
    if director:
        parts.append(f"director_{director.replace(' ', '_')}")
    return " ".join(parts)
