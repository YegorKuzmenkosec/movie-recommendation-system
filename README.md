# Movie Recommender

Content-based movie recommendations with TF-IDF vectors and cosine similarity, plus optional item–item collaborative filtering.

## What this shows

- Feature engineering: plot text, keywords, genres, cast, director, rating buckets
- Vectorization with scikit-learn `TfidfVectorizer`
- Pairwise **cosine similarity** over sparse feature matrices
- Jaccard overlap on structured fields (genres, cast, director)
- Item–item collaborative filtering on a user–movie rating matrix

Example (content-based):

```
User likes:
  Interstellar

Recommendations:
  The Martian
  Inception
  Ad Astra
  Gravity
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Content-based (default):

```bash
python main.py Interstellar
python main.py "The Dark Knight" -n 5
```

Collaborative filtering (users who liked the same film):

```bash
python main.py Interstellar --mode collaborative
python main.py --mode collaborative --user bob
```

Both engines:

```bash
python main.py Interstellar --mode both
```

## Data

| File | Role |
| --- | --- |
| `data/movies.csv` | Catalog: overview, genres, cast, director, rating, keywords |
| `data/ratings.csv` | Synthetic user–item ratings for collaborative filtering |

Swap in your own CSVs as long as the column names match.

## How content-based scoring works

1. Build an **overview document** (plot + repeated keywords) and a **metadata document** (genres, cast, director token, rating tier).
2. Fit two TF-IDF vectorizers and stack the sparse matrices.
3. Compute cosine similarity between every pair of movies.
4. Blend with Jaccard similarities on genres, cast, and director.

The result is a catalog-sized similarity matrix; recommendations are the nearest neighbors of the seed title.
