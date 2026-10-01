"""Content-based recommendations via TF-IDF and cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender.features import (
    build_metadata_document,
    build_overview_document,
    parse_genres,
    parse_pipe_list,
)


@dataclass
class Recommendation:
    title: str
    score: float
    rating: float
    genres: str


def _jaccard_matrix(item_sets: list[set[str]]) -> np.ndarray:
    n = len(item_sets)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            a, b = item_sets[i], item_sets[j]
            union = a | b
            s = (len(a & b) / len(union)) if union else 0.0
            sim[i, j] = s
            sim[j, i] = s
    return sim


class ContentBasedRecommender:
    def __init__(
        self,
        movies: pd.DataFrame,
        *,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        structured_weight: float = 0.18,
    ) -> None:
        required = {"title", "overview", "genres", "cast", "rating"}
        missing = required - set(movies.columns)
        if missing:
            raise ValueError(f"movies DataFrame missing columns: {sorted(missing)}")

        self.structured_weight = structured_weight
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["overview_doc"] = self.movies.apply(build_overview_document, axis=1)
        self.movies["metadata_doc"] = self.movies.apply(build_metadata_document, axis=1)

        overview_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        metadata_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features // 2,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )

        overview_matrix = overview_vectorizer.fit_transform(self.movies["overview_doc"])
        metadata_matrix = metadata_vectorizer.fit_transform(self.movies["metadata_doc"])
        # Plot/keywords dominate; metadata (genres, cast, director, rating tier) is a lighter signal.
        tfidf_matrix = hstack(
            [overview_matrix, metadata_matrix.multiply(0.35)], format="csr"
        )
        tfidf_sim = cosine_similarity(tfidf_matrix)

        genre_sim = _jaccard_matrix([parse_genres(g) for g in self.movies["genres"]])
        cast_sim = _jaccard_matrix([parse_pipe_list(c) for c in self.movies["cast"]])
        if "director" in self.movies.columns:
            director_sets = [
                {str(d).strip()} if str(d).strip() else set()
                for d in self.movies["director"]
            ]
            director_sim = _jaccard_matrix(director_sets)
        else:
            director_sim = np.zeros_like(genre_sim)

        structured_sim = 0.50 * genre_sim + 0.25 * cast_sim + 0.25 * director_sim
        w = self.structured_weight
        self.similarity_matrix = (1.0 - w) * tfidf_sim + w * structured_sim
        self.overview_vectorizer = overview_vectorizer
        self.metadata_vectorizer = metadata_vectorizer

    def _index_for_title(self, title: str) -> int:
        matches = self.movies.index[
            self.movies["title"].str.casefold() == title.casefold()
        ].tolist()
        if not matches:
            available = ", ".join(self.movies["title"].head(5).tolist())
            raise KeyError(
                f"Movie '{title}' not found. Examples in catalog: {available}..."
            )
        return matches[0]

    def recommend(self, title: str, top_n: int = 5) -> list[Recommendation]:
        idx = self._index_for_title(title)
        scores = self.similarity_matrix[idx]
        ranked = np.argsort(scores)[::-1]

        results: list[Recommendation] = []
        for j in ranked:
            if j == idx:
                continue
            row = self.movies.iloc[j]
            results.append(
                Recommendation(
                    title=row["title"],
                    score=float(scores[j]),
                    rating=float(row["rating"]),
                    genres=str(row["genres"]),
                )
            )
            if len(results) >= top_n:
                break
        return results

    def similar_pairs(self, min_score: float = 0.15, limit: int = 20) -> pd.DataFrame:
        """Debug/explore: top similar movie pairs in the catalog."""
        n = len(self.movies)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                s = self.similarity_matrix[i, j]
                if s >= min_score:
                    pairs.append(
                        {
                            "title_a": self.movies.at[i, "title"],
                            "title_b": self.movies.at[j, "title"],
                            "cosine_similarity": s,
                        }
                    )
        pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return pd.DataFrame(pairs[:limit])
