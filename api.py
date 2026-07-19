"""
API de recommandation de films.

Lancement :  python -m uvicorn api:app --reload
Doc auto   :  http://localhost:8000/docs

Pour l'instant les films/notes viennent des fichiers ml-100k directement.
Étape suivante (Jour 2 du plan) : remplacer ça par PostgreSQL.

Affiches réelles : nécessite une clé TMDB gratuite (themoviedb.org/settings/api).
Mets-la dans la variable d'environnement TMDB_API_KEY avant de lancer l'API :
  export TMDB_API_KEY="ta_clé_ici"
  python -m uvicorn api:app --reload
Sans clé, l'API fonctionne quand même (poster_url sera vide, le frontend
retombe automatiquement sur les cartes colorées par genre).
"""
import os
import re
from typing import List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from surprise import dump

app = FastAPI(title="Movie Recommender API")

# Autorise le frontend (fichier HTML servi en local ou ailleurs) à appeler l'API
# ⚠️ allow_origins=["*"] = OK en dev/démo locale, à restreindre absolument
# à ton vrai domaine avant toute mise en ligne publique (sinon n'importe quel
# site peut appeler ton API depuis le navigateur d'un visiteur).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Chargement du modèle entraîné + des données films au démarrage
# ------------------------------------------------------------------
_, algo = dump.load("model.pkl")

genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]
item_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')

ratings = pd.read_csv(
    'ml-100k/u.data', sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

# Films les plus populaires, utilisés pour le "cold start"
# (un nouvel utilisateur sans historique reçoit les films les mieux notés/plus populaires)
popular_movies = (
    ratings.groupby('item_id')
    .agg(n_ratings=('rating', 'count'), avg_rating=('rating', 'mean'))
    .query('n_ratings >= 50')  # évite les films avec 1-2 notes de 5
    .sort_values('avg_rating', ascending=False)
    .reset_index()
)

all_item_ids = movies['item_id'].tolist()
known_user_ids = set(ratings['user_id'].unique())


def get_primary_genre(item_id: int) -> str:
    """Renvoie le premier genre actif d'un film (utilisé pour l'affichage en front)."""
    row = movies.loc[movies.item_id == item_id, genre_cols]
    if row.empty:
        return "unknown"
    active = row.iloc[0]
    active_genres = [g for g in genre_cols if active[g] == 1]
    return active_genres[0] if active_genres else "unknown"


MAIN_GENRES = [
    'Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi',
    'Thriller', 'Adventure', "Children's", 'Horror'
]


# ------------------------------------------------------------------
# Affiches réelles via TMDB (optionnel — fonctionne sans clé, juste sans image)
# ------------------------------------------------------------------
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

_poster_cache: dict[int, Optional[str]] = {}


def get_poster_url(item_id: int, title: str) -> Optional[str]:
    """
    Cherche l'affiche du film sur TMDB à partir de son titre (les titres
    ml-100k sont au format "Toy Story (1995)" -> on extrait le titre et
    l'année pour affiner la recherche). Résultat mis en cache en mémoire
    pour éviter de re-questionner TMDB à chaque requête sur le même film.
    """
    if not TMDB_API_KEY:
        return None
    if item_id in _poster_cache:
        return _poster_cache[item_id]

    match = re.match(r"^(.*)\s\((\d{4})\)$", title.strip())
    clean_title, year = (match.group(1), match.group(2)) if match else (title, None)

    try:
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if year:
            params["year"] = year
        resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=3)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        poster_path = results[0]["poster_path"] if results else None
        url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except Exception:
        url = None  # une affiche manquante ne doit jamais faire planter l'API

    _poster_cache[item_id] = url
    return url


# ------------------------------------------------------------------
# Schémas de réponse
# ------------------------------------------------------------------
class MovieRecommendation(BaseModel):
    item_id: int
    title: str
    predicted_rating: float
    genre: str
    poster_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id: int
    is_new_user: bool
    recommendations: List[MovieRecommendation]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Recommender API"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, n: int = 10):
    """
    Retourne les N films recommandés pour un utilisateur.
    Si l'utilisateur est inconnu du modèle (cold start), on renvoie
    les films populaires les mieux notés à la place.
    """
    is_new_user = user_id not in known_user_ids

    if is_new_user:
        top = popular_movies.head(n)
        recs = []
        for row in top.itertuples():
            title = movies.loc[movies.item_id == row.item_id, 'title'].values[0]
            recs.append(MovieRecommendation(
                item_id=int(row.item_id),
                title=title,
                predicted_rating=round(row.avg_rating, 2),
                genre=get_primary_genre(row.item_id),
                poster_url=get_poster_url(row.item_id, title),
            ))
        return RecommendationResponse(user_id=user_id, is_new_user=True, recommendations=recs)

    # Films déjà notés par l'utilisateur -> à exclure des recommandations
    already_rated = set(ratings.loc[ratings.user_id == user_id, 'item_id'])
    candidates = [iid for iid in all_item_ids if iid not in already_rated]

    predictions = [algo.predict(user_id, iid) for iid in candidates]
    predictions.sort(key=lambda p: p.est, reverse=True)
    top_n = predictions[:n]

    recs = []
    for p in top_n:
        title = movies.loc[movies.item_id == p.iid, 'title'].values[0]
        recs.append(MovieRecommendation(
            item_id=int(p.iid),
            title=title,
            predicted_rating=round(p.est, 2),
            genre=get_primary_genre(p.iid),
            poster_url=get_poster_url(p.iid, title),
        ))
    return RecommendationResponse(user_id=user_id, is_new_user=False, recommendations=recs)


@app.get("/genres")
def list_genres():
    """Liste des genres principaux, utilisée par le front pour organiser l'affichage en rangées."""
    return {"genres": MAIN_GENRES}


@app.get("/movies/by_genre/{genre}")
def movies_by_genre(genre: str, limit: int = 15):
    """Films d'un genre donné, triés par popularité (pour les rangées façon Netflix)."""
    if genre not in genre_cols:
        raise HTTPException(status_code=404, detail=f"Genre inconnu : {genre}")

    genre_movies = movies[movies[genre] == 1][['item_id', 'title']]
    merged = genre_movies.merge(popular_movies, on='item_id', how='left')
    merged['n_ratings'] = merged['n_ratings'].fillna(0)
    merged = merged.sort_values('n_ratings', ascending=False).head(limit)

    result = [
        {
            "item_id": int(r.item_id),
            "title": r.title,
            "avg_rating": round(r.avg_rating, 2) if pd.notna(r.avg_rating) else None,
            "poster_url": get_poster_url(int(r.item_id), r.title),
        }
        for r in merged.itertuples()
    ]
    return {"genre": genre, "movies": result}


@app.get("/movies")
def list_movies(search: str = "", limit: int = 20):
    """Recherche de films par titre (pour que le front propose une liste à noter)."""
    df = movies
    if search:
        df = df[df.title.str.contains(search, case=False, na=False)]
    result = df[['item_id', 'title']].head(limit).to_dict(orient='records')
    return {"movies": result}


@app.get("/user/{user_id}/exists")
def user_exists(user_id: int):
    return {"user_id": user_id, "exists": user_id in known_user_ids}