"""Item–item collaborative filtering on a user–movie rating matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CollaborativeRecommendation:
    title: str
    score: float
    predicted_rating: float | None = None


class ItemItemCollaborativeRecommender:
    """Adjusted cosine similarity between movies in user–rating space.

    Missing ratings stay missing (not zero). Each user's ratings are mean-centered
    so that cosine compares preference patterns, not raw score scale.
    """

    def __init__(self, ratings: pd.DataFrame) -> None:
        required = {"user_id", "title", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings DataFrame missing columns: {sorted(missing)}")

        self.ratings = ratings.copy()
        self.user_item = self.ratings.pivot_table(
            index="user_id", columns="title", values="rating", aggfunc="mean"
        )
        self.titles = list(self.user_item.columns)
        self.title_index = {title: i for i, title in enumerate(self.titles)}

        user_mean = self.user_item.mean(axis=1)
        centered = self.user_item.sub(user_mean, axis=0).fillna(0.0)
        self.item_similarity = cosine_similarity(centered.T)
        np.fill_diagonal(self.item_similarity, 1.0)

    def _index_for_title(self, title: str) -> int:
        key = next((t for t in self.titles if t.casefold() == title.casefold()), None)
        if key is None:
            raise KeyError(f"Movie '{title}' has no ratings in the interaction matrix.")
        return self.title_index[key]

    def recommend_for_movie(self, title: str, top_n: int = 5) -> list[CollaborativeRecommendation]:
        idx = self._index_for_title(title)
        scores = self.item_similarity[idx]
        ranked = np.argsort(scores)[::-1]
        results: list[CollaborativeRecommendation] = []
        for j in ranked:
            if j == idx:
                continue
            results.append(
                CollaborativeRecommendation(title=self.titles[j], score=float(scores[j]))
            )
            if len(results) >= top_n:
                break
        return results

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> list[CollaborativeRecommendation]:
        if user_id not in self.user_item.index:
            raise KeyError(f"Unknown user_id '{user_id}'.")

        user_ratings = self.user_item.loc[user_id]
        seen_mask = user_ratings.notna().to_numpy()
        ratings_vec = user_ratings.fillna(0.0).to_numpy(dtype=np.float64)

        sims = self.item_similarity.copy()
        sims[sims < 0] = 0.0
        liked = np.where(seen_mask, ratings_vec, 0.0)
        affinity = sims @ liked
        affinity[seen_mask] = -np.inf

        weighted = sims[:, seen_mask] @ ratings_vec[seen_mask]
        normalizer = sims[:, seen_mask].sum(axis=1)
        predicted = np.divide(
            weighted,
            normalizer,
            out=np.zeros_like(weighted),
            where=normalizer > 1e-9,
        )

        ranked = np.argsort(affinity)[::-1]
        results: list[CollaborativeRecommendation] = []
        for j in ranked:
            if not np.isfinite(affinity[j]):
                continue
            results.append(
                CollaborativeRecommendation(
                    title=self.titles[j],
                    score=float(affinity[j]),
                    predicted_rating=float(predicted[j]),
                )
            )
            if len(results) >= top_n:
                break
        return results
