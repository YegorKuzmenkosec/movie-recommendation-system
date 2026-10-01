#!/usr/bin/env python3
"""CLI: content-based and item–item collaborative movie recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recommender.collaborative import ItemItemCollaborativeRecommender
from recommender.content_based import ContentBasedRecommender

ROOT = Path(__file__).resolve().parent
DEFAULT_MOVIES = ROOT / "data" / "movies.csv"
DEFAULT_RATINGS = ROOT / "data" / "ratings.csv"


def _print_content(title: str, recs) -> None:
    print(f"User likes:\n  {title}\n")
    print("Recommendations (content-based):")
    for i, rec in enumerate(recs, start=1):
        print(
            f"  {i}. {rec.title}  "
            f"(similarity={rec.score:.3f}, rating={rec.rating}, genres={rec.genres})"
        )


def _print_cf_movie(title: str, recs) -> None:
    print(f"Users who liked:\n  {title}\n")
    print("Also liked (item–item collaborative):")
    for i, rec in enumerate(recs, start=1):
        print(f"  {i}. {rec.title}  (similarity={rec.score:.3f})")


def _print_cf_user(user_id: str, recs) -> None:
    print(f"User:\n  {user_id}\n")
    print("Recommendations (collaborative):")
    for i, rec in enumerate(recs, start=1):
        extra = (
            f", predicted={rec.predicted_rating:.2f}"
            if rec.predicted_rating is not None
            else ""
        )
        print(f"  {i}. {rec.title}  (score={rec.score:.3f}{extra})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Movie recommendations: TF-IDF cosine similarity, plus item–item CF."
    )
    parser.add_argument(
        "title",
        nargs="?",
        default="Interstellar",
        help="Movie title the user likes (default: Interstellar)",
    )
    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=5,
        help="Number of recommendations (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=("content", "collaborative", "both"),
        default="content",
        help="Recommendation engine (default: content)",
    )
    parser.add_argument(
        "--user",
        help="For collaborative mode: recommend for this user_id instead of a seed movie",
    )
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    args = parser.parse_args()

    if args.mode in {"content", "both"}:
        movies = pd.read_csv(args.movies)
        engine = ContentBasedRecommender(movies)
        _print_content(args.title, engine.recommend(args.title, top_n=args.top))
        if args.mode == "both":
            print()

    if args.mode in {"collaborative", "both"}:
        ratings = pd.read_csv(args.ratings)
        cf = ItemItemCollaborativeRecommender(ratings)
        if args.user:
            _print_cf_user(args.user, cf.recommend_for_user(args.user, top_n=args.top))
        else:
            _print_cf_movie(args.title, cf.recommend_for_movie(args.title, top_n=args.top))


if __name__ == "__main__":
    main()
