"""
Exploration du dataset MovieLens 100k.
Objectif : comprendre la distribution des notes, la sparsité,
et la popularité des films avant d'entraîner un modèle.
"""
import pandas as pd

# --- Chargement des notes ---
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

# --- Chargement des films (titre + genres) ---
genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]
item_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
movies = pd.read_csv(
    'ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1'
)

print("=" * 50)
print("APERÇU DES DONNÉES")
print("=" * 50)
print(ratings.head())
print()

n_users = ratings.user_id.nunique()
n_items = ratings.item_id.nunique()
n_ratings = len(ratings)

print(f"Utilisateurs : {n_users}")
print(f"Films        : {n_items}")
print(f"Notes        : {n_ratings}")

# --- Sparsité de la matrice user-item ---
# % de la matrice qui est "vide" (pas de note)
sparsity = 1 - (n_ratings / (n_users * n_items))
print(f"Sparsité     : {sparsity:.2%}  (matrice quasi vide, normal pour du CF)")

# --- Distribution des notes ---
print("\n" + "=" * 50)
print("DISTRIBUTION DES NOTES")
print("=" * 50)
print(ratings.rating.value_counts().sort_index())
print(f"Moyenne globale : {ratings.rating.mean():.2f}")

# --- Popularité des films (nombre de notes reçues) ---
print("\n" + "=" * 50)
print("TOP 10 FILMS LES PLUS NOTÉS")
print("=" * 50)
popularity = ratings.groupby('item_id').size().sort_values(ascending=False)
top10 = popularity.head(10).reset_index(name='n_ratings')
top10 = top10.merge(movies[['item_id', 'title']], on='item_id')
print(top10[['title', 'n_ratings']].to_string(index=False))

# --- Activité des utilisateurs ---
print("\n" + "=" * 50)
print("ACTIVITÉ DES UTILISATEURS")
print("=" * 50)
user_activity = ratings.groupby('user_id').size()
print(f"Notes par utilisateur : min={user_activity.min()}, "
      f"médiane={user_activity.median():.0f}, max={user_activity.max()}")
print("=> important pour gérer le 'cold start' : certains users ont peu noté")